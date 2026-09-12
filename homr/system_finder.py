"""Propose printed-system bounds from homr's own staff and barline detection.

This is deliberately a proposal, not an editor: callers decide whether to save,
change, or reject the returned bounds. Choral pages are not rectangular tables,
so the useful question here is where each printed system begins and ends, not
which logical singer owns each detected staff.

The grouping rule is the measured rule previously maintained by the choir app.
Adjacent staves belong to one printed system when their interior barlines line
up. Vertical whitespace is only a veto/fallback because real choir pages have
overlapping within-system and between-system gap ranges.
"""

from __future__ import annotations

import math
import os
import statistics
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from typing import TypedDict

import cv2
import numpy as np
import pypdfium2 as pdfium

from homr import color_adjust
from homr.bar_line_detection import detect_bar_lines
from homr.debug import Debug
from homr.image_prediction import get_predictions, predict_symbols
from homr.noise_filtering import filter_predictions
from homr.note_detection import combine_noteheads_with_stems
from homr.resize import resize_image
from homr.staff_detection import break_wide_fragments, detect_staff, make_lines_stronger
from homr.system_crops import SystemBounds
from homr.type_definitions import NDArray

FIND_DPI = 200
TOL_X = 0.008
EDGE_X = 0.02
AGREE = 0.6
SLACK = 1.001

Logger = Callable[[str], None]


class NormalizedBox(TypedDict):
    top: float
    bottom: float
    left: float
    right: float


class PageGeometry(TypedDict):
    width: int
    height: int
    staves: list[NormalizedBox]
    bar_lines: list[NormalizedBox]


def _noop(_message: str) -> None:
    pass


def _box(
    top: float,
    bottom: float,
    left: float,
    right: float,
    height: int,
    width: int,
) -> NormalizedBox:
    return {
        "top": float(top) / height,
        "bottom": float(bottom) / height,
        "left": float(left) / width,
        "right": float(right) / width,
    }


def detect_page_geometry(
    image: NDArray,
    source_name: str = "system-finder-page",
    use_gpu: bool = False,
) -> PageGeometry:
    """Return normalized staff and barline boxes without decoding any music."""
    image = resize_image(image)
    preprocessed = color_adjust.apply_clahe(image)
    predictions = get_predictions(image, preprocessed, source_name, False, use_gpu)
    debug = Debug(predictions.original, source_name, False)
    predictions = filter_predictions(predictions, debug)
    predictions.staff = make_lines_stronger(predictions.staff, (1, 2))

    symbols = predict_symbols(debug, predictions)
    symbols.staff_fragments = break_wide_fragments(symbols.staff_fragments)

    noteheads_with_stems = combine_noteheads_with_stems(symbols.noteheads, symbols.stems_rest)
    if not noteheads_with_stems:
        raise ValueError(f"No noteheads found in {source_name}")
    notehead_heights = np.asarray(
        [float(note.notehead.size[1]) for note in noteheads_with_stems], dtype=float
    )
    average_notehead_height = float(np.median(notehead_heights))
    all_noteheads = [note.notehead for note in noteheads_with_stems]
    all_stems = [note.stem for note in noteheads_with_stems if note.stem is not None]
    bar_lines_or_rests = [
        line
        for line in symbols.bar_lines
        if not line.is_overlapping_with_any(all_noteheads)
        and not line.is_overlapping_with_any(all_stems)
    ]
    bar_line_boxes = detect_bar_lines(bar_lines_or_rests, average_notehead_height)
    staves = detect_staff(
        debug,
        predictions.staff,
        symbols.staff_fragments,
        symbols.clefs_keys,
        bar_line_boxes,
    )

    height, width = predictions.staff.shape[:2]
    return {
        "width": int(width),
        "height": int(height),
        "staves": [
            _box(staff.min_y, staff.max_y, staff.min_x, staff.max_x, height, width)
            for staff in staves
        ],
        "bar_lines": [
            _box(
                bar.center[1] - bar.size[1] / 2,
                bar.center[1] + bar.size[1] / 2,
                bar.center[0] - bar.size[0] / 2,
                bar.center[0] + bar.size[0] / 2,
                height,
                width,
            )
            for bar in bar_line_boxes
        ],
    }


def _interior_barlines(staff: NormalizedBox, bar_lines: Sequence[NormalizedBox]) -> list[float]:
    """Barline x positions on one staff, excluding the system's two ends."""
    found: list[float] = []
    for bar in bar_lines:
        if bar["bottom"] < staff["top"] - 0.005 or bar["top"] > staff["bottom"] + 0.005:
            continue
        x = (bar["left"] + bar["right"]) / 2
        if x < staff["left"] + EDGE_X or x > staff["right"] - EDGE_X:
            continue
        found.append(x)
    return sorted(found)


def _agreement(a: Sequence[float], b: Sequence[float]) -> float | None:
    """How much of either barline set has a counterpart on the other staff."""
    if not a or not b:
        return None
    hits_a = sum(1 for x in a if any(abs(x - y) <= TOL_X for y in b))
    hits_b = sum(1 for y in b if any(abs(x - y) <= TOL_X for x in a))
    return max(hits_a / len(a), hits_b / len(b))


def _gap_threshold(gaps: Sequence[float]) -> float:
    """Split at the largest multiplicative step in the sorted gaps."""
    ordered = sorted(gaps)
    if len(ordered) < 2:
        return ordered[0] + 1.0
    steps = [
        (ordered[index + 1] / max(ordered[index], 1e-6), index) for index in range(len(ordered) - 1)
    ]
    _, cut = max(steps)
    return (ordered[cut] + ordered[cut + 1]) / 2


def group_staves(
    staves: Sequence[NormalizedBox], bar_lines: Sequence[NormalizedBox]
) -> list[list[NormalizedBox]]:
    """Group page-order staves into printed systems using barline agreement."""
    ordered_staves = sorted(staves, key=lambda staff: staff["top"])
    if len(ordered_staves) < 2:
        return [list(ordered_staves)] if ordered_staves else []

    bars = [_interior_barlines(staff, bar_lines) for staff in ordered_staves]
    gaps = [
        ordered_staves[index + 1]["top"] - ordered_staves[index]["bottom"]
        for index in range(len(ordered_staves) - 1)
    ]
    scores = [_agreement(bars[index], bars[index + 1]) for index in range(len(bars) - 1)]
    same: list[bool | None] = [None if score is None else score >= AGREE for score in scores]

    inside = [gap for gap, is_same in zip(gaps, same, strict=True) if is_same]
    if inside:
        ordinary = statistics.median(inside)
        same = [
            True if is_same is False and gap < ordinary / SLACK else is_same
            for is_same, gap in zip(same, gaps, strict=True)
        ]
        same = [
            is_same if is_same is not None else gap <= ordinary * SLACK
            for is_same, gap in zip(same, gaps, strict=True)
        ]
    else:
        threshold = _gap_threshold(gaps)
        same = [gap < threshold for gap in gaps]

    systems: list[list[NormalizedBox]] = [[ordered_staves[0]]]
    for keep, staff in zip(same, ordered_staves[1:], strict=True):
        if keep:
            systems[-1].append(staff)
        else:
            systems.append([staff])
    return systems


def bands_for_page(
    page: int,
    staves: Sequence[NormalizedBox],
    bar_lines: Sequence[NormalizedBox],
) -> list[SystemBounds]:
    """Return contiguous proposal bands for one page, indexed from one."""
    systems = group_staves(staves, bar_lines)
    if not systems:
        return []
    tops = [system[0]["top"] for system in systems]
    bottoms = [system[-1]["bottom"] for system in systems]
    between = [tops[index + 1] - bottoms[index] for index in range(len(systems) - 1)]
    room = statistics.median(between) if between else 0.05

    bands: list[SystemBounds] = []
    for index in range(len(systems)):
        top = max(0.0, tops[index] - room) if index == 0 else (bottoms[index - 1] + tops[index]) / 2
        bottom = (
            min(1.0, bottoms[index] + room)
            if index == len(systems) - 1
            else (bottoms[index] + tops[index + 1]) / 2
        )
        bands.append(SystemBounds(index=index + 1, page=page, top=top, bottom=bottom))
    return bands


def _render_pdf_page(pdf_path: str, page_number: int, dpi: int, workdir: str) -> NDArray:
    """Raster one PDF page through Poppler, matching the measured choir proposal seam."""
    stem = os.path.join(workdir, f"page-{page_number:03d}")
    try:
        # The executable is fixed on the host's trusted PATH. PDF paths are argv
        # entries, never shell input; no shell is started here.
        subprocess.run(  # noqa: S603
            [  # noqa: S607
                "pdftoppm",
                "-r",
                str(dpi),
                "-f",
                str(page_number),
                "-l",
                str(page_number),
                "-png",
                "-singlefile",
                pdf_path,
                stem,
            ],
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise OSError(f"Could not render PDF page {page_number}: {error}") from error
    image = cv2.imread(stem + ".png")
    if image is None:
        raise OSError(f"Could not read rendered PDF page {page_number}")
    return image


def find_system_bounds(
    pdf_path: str,
    use_gpu: bool = False,
    dpi: int = FIND_DPI,
    log: Logger = _noop,
    page: int | None = None,
) -> list[SystemBounds]:
    """Propose printed-system bands in a PDF, optionally for one page only.

    Full-PDF results are indexed across the score. A single-page request is
    intentionally indexed from one because its caller may hold one compute slot
    per page and reindex while accumulating the page results.
    """
    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError("system-bound proposal requires a PDF input")
    if dpi < 1 or not math.isfinite(float(dpi)):
        raise ValueError("system-bound proposal dpi must be at least 1")
    try:
        document = pdfium.PdfDocument(pdf_path)
    except Exception as error:
        raise ValueError(f"could not open PDF {pdf_path}: {error}") from error

    result: list[SystemBounds] = []
    try:
        page_count = len(document)
        if page is not None and not 1 <= page <= page_count:
            raise ValueError(
                f"system-bound proposal page {page} does not exist; PDF has {page_count} page(s)"
            )
        page_numbers = [page] if page is not None else list(range(1, page_count + 1))
        with tempfile.TemporaryDirectory(prefix="homr-system-finder-") as workdir:
            for page_number in page_numbers:
                log(f"Looking for systems on page {page_number} of {page_count}")
                image = _render_pdf_page(pdf_path, page_number, dpi, workdir)
                geometry = detect_page_geometry(
                    image,
                    source_name=f"{pdf_path}#page-{page_number}",
                    use_gpu=use_gpu,
                )
                page_bands = bands_for_page(page_number, geometry["staves"], geometry["bar_lines"])
                log(
                    f"Page {page_number}: {len(geometry['staves'])} staves in "
                    f"{len(page_bands)} system(s)"
                )
                for band in page_bands:
                    result.append(
                        SystemBounds(
                            index=len(result) + 1,
                            page=page_number,
                            top=band.top,
                            bottom=band.bottom,
                        )
                    )
    finally:
        document.close()
    return result


def bounds_payload(
    bounds: Sequence[SystemBounds],
) -> dict[str, list[dict[str, int | float]]]:
    """Stable JSON schema matching the choir app's ``SystemBounds.to_dict()``."""
    return {
        "systems": [
            {
                "index": bound.index,
                "page": bound.page,
                "top": bound.top,
                "bottom": bound.bottom,
                "measure_start": bound.measure_start,
                "measure_end": bound.measure_end,
            }
            for bound in bounds
        ]
    }
