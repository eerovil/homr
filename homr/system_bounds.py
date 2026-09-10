"""Propose printed-system bands from homr's staff and barline detections.

This is deliberately a proposal API: it detects and groups what is printed but never
writes or approves a caller's systems file. Coordinates are fractions of the original
PDF page and indices are one-based across the whole score, matching ``SystemBounds``.

The grouping rule is the measured choir-page rule moved from musescore-choir-plugins:
adjacent staves belong to one system when their interior barlines agree; page geometry
is only a veto/fallback when barline evidence is missing. No music decoding happens.
"""

from __future__ import annotations

import json
import os
import statistics
import tempfile
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium

from homr import color_adjust
from homr.bar_line_detection import detect_bar_lines
from homr.debug import Debug
from homr.main import get_predictions, predict_symbols
from homr.noise_filtering import filter_predictions
from homr.note_detection import combine_noteheads_with_stems
from homr.resize import resize_image
from homr.staff_detection import break_wide_fragments, detect_staff, make_lines_stronger
from homr.system_crops import DEFAULT_SYSTEM_DPI, SystemBounds
from homr.type_definitions import NDArray

# Keep these values stable: they are measured against the choir benchmark and are part of
# the proposal contract, not tuning knobs for individual pages.
TOL_X = 0.008
EDGE_X = 0.02
AGREE = 0.6
SLACK = 1.001
DEFAULT_PROPOSAL_DPI = DEFAULT_SYSTEM_DPI  # 200 dpi

Box = dict[str, float]


def _box(
    top: float,
    bottom: float,
    left: float,
    right: float,
    height: int,
    width: int,
) -> Box:
    return {
        "top": float(top) / height,
        "bottom": float(bottom) / height,
        "left": float(left) / width,
        "right": float(right) / width,
    }


def detect_staff_layout(image_path: str, segnet_use_gpu: bool = False) -> dict[str, object]:
    """Return staff and barline boxes for one page image without decoding music."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read {image_path}")

    # PDF pages are already page-aligned. Do not autocrop: proposal coordinates must remain
    # fractions of the original page so a caller can use them directly as system bounds.
    image = resize_image(image)
    preprocessed = color_adjust.apply_clahe(image)
    predictions = get_predictions(image, preprocessed, image_path, False, segnet_use_gpu)
    debug = Debug(predictions.original, image_path, False)
    predictions = filter_predictions(predictions, debug)
    predictions.staff = make_lines_stronger(predictions.staff, (1, 2))

    symbols = predict_symbols(debug, predictions)
    symbols.staff_fragments = break_wide_fragments(symbols.staff_fragments)

    noteheads_with_stems = combine_noteheads_with_stems(symbols.noteheads, symbols.stems_rest)
    if not noteheads_with_stems:
        raise ValueError(f"No noteheads found in {image_path}")
    average_note_head_height = float(
        np.median([note.notehead.size[1] for note in noteheads_with_stems])
    )
    all_noteheads = [note.notehead for note in noteheads_with_stems]
    all_stems = [note.stem for note in noteheads_with_stems if note.stem is not None]
    bar_lines_or_rests = [
        line
        for line in symbols.bar_lines
        if not line.is_overlapping_with_any(all_noteheads)
        and not line.is_overlapping_with_any(all_stems)
    ]
    bar_line_boxes = detect_bar_lines(bar_lines_or_rests, average_note_head_height)
    staves = detect_staff(
        debug,
        predictions.staff,
        symbols.staff_fragments,
        symbols.clefs_keys,
        bar_line_boxes,
    )

    height, width = predictions.staff.shape[:2]
    return {
        "width": width,
        "height": height,
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


def _interior_barlines(staff: Box, bar_lines: Sequence[Box]) -> list[float]:
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
    if not a or not b:
        return None
    hits_a = sum(1 for x in a if any(abs(x - y) <= TOL_X for y in b))
    hits_b = sum(1 for y in b if any(abs(x - y) <= TOL_X for x in a))
    return max(hits_a / len(a), hits_b / len(b))


def _gap_threshold(gaps: Sequence[float]) -> float:
    ordered = sorted(gaps)
    if len(ordered) < 2:
        return ordered[0] + 1.0
    steps = [
        (ordered[index + 1] / max(ordered[index], 1e-6), index)
        for index in range(len(ordered) - 1)
    ]
    _, cut = max(steps)
    return (ordered[cut] + ordered[cut + 1]) / 2


def group_staves(staves: Sequence[Box], bar_lines: Sequence[Box]) -> list[list[Box]]:
    """Group page-order staves into printed systems using barline agreement."""
    ordered_staves = sorted(staves, key=lambda staff: staff["top"])
    if len(ordered_staves) < 2:
        return [ordered_staves] if ordered_staves else []

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

    systems: list[list[Box]] = [[ordered_staves[0]]]
    for keep, staff in zip(same, ordered_staves[1:], strict=True):
        if keep:
            systems[-1].append(staff)
        else:
            systems.append([staff])
    return systems


def bands_for_page(page: int, staves: Sequence[Box], bar_lines: Sequence[Box]) -> list[SystemBounds]:
    """Return contiguous system bands for one page, indexed from one on that page."""
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


def _render_pdf_page(page: pdfium.PdfPage, path: str, dpi: int) -> None:
    bitmap = page.render(scale=dpi / 72.0)
    rgb = np.array(bitmap.to_pil().convert("RGB"))
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(path, image):
        raise OSError(f"Could not write temporary page image {path}")


def propose_system_bounds(
    pdf_path: str,
    *,
    dpi: int = DEFAULT_PROPOSAL_DPI,
    segnet_use_gpu: bool = False,
) -> list[SystemBounds]:
    """Detect and group every printed system in a PDF without saving a proposal."""
    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError("system-bound proposal requires a PDF input")
    if dpi < 1:
        raise ValueError("proposal dpi must be at least 1")

    pdf = pdfium.PdfDocument(pdf_path)
    if not pdf:
        raise ValueError(f"invalid PDF {pdf_path}")
    try:
        result: list[SystemBounds] = []
        with tempfile.TemporaryDirectory(prefix="homr-system-bounds-") as scratch:
            for page_index in range(len(pdf)):
                image_path = os.path.join(scratch, f"page-{page_index + 1:03d}.png")
                _render_pdf_page(pdf[page_index], image_path, dpi)
                layout = detect_staff_layout(image_path, segnet_use_gpu=segnet_use_gpu)
                staves = layout.get("staves", [])
                bar_lines = layout.get("bar_lines", [])
                if not isinstance(staves, list) or not isinstance(bar_lines, list):
                    raise ValueError("staff detector returned an invalid layout")
                page_bands = bands_for_page(page_index + 1, staves, bar_lines)
                for band in page_bands:
                    result.append(
                        SystemBounds(
                            index=len(result) + 1,
                            page=band.page,
                            top=band.top,
                            bottom=band.bottom,
                        )
                    )
        return result
    finally:
        pdf.close()


def proposal_json(bounds: Sequence[SystemBounds], dpi: int = DEFAULT_PROPOSAL_DPI) -> str:
    """Stable machine-readable proposal output accepted by the existing systems schema."""
    payload = {
        "version": 1,
        "dpi": dpi,
        "systems": [
            {
                "index": bound.index,
                "page": bound.page,
                "top": bound.top,
                "bottom": bound.bottom,
            }
            for bound in bounds
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
