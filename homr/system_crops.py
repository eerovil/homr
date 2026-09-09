"""Crop explicitly supplied printed systems from a PDF, in stable score order.

Choral pages are not rectangular tables: a voice that rests through a system may
simply not be printed.  Homr's normal whole-page parser has to reconcile staff
row N across every system, which is therefore the wrong question for such a
page.  This module does not guess the missing logical voice.  It accepts the
system bands already known by the caller and makes each band an independent
image, so the ordinary one-system recognition path can answer only what the
pixels actually contain.

The JSON schema intentionally matches musescore-choir-plugins' ``.systems.json``:
``top`` and ``bottom`` are fractions of the original PDF page height and ``page``
and ``index`` are one-based.  Measure ranges are accepted as provenance but are
not used by recognition.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium

from homr.type_definitions import NDArray

DEFAULT_SYSTEM_DPI = 200


@dataclass(frozen=True)
class SystemBounds:
    """One printed system in the source PDF."""

    index: int
    page: int
    top: float
    bottom: float
    measure_start: int = 0
    measure_end: int = 0


@dataclass(frozen=True)
class SystemCrop:
    """A rendered system image and the bounds it came from."""

    bounds: SystemBounds
    path: str

    @property
    def index(self) -> int:
        return self.bounds.index


def _as_int(value: object, field: str) -> int:
    if type(value) is not int:  # bool is not a valid index
        raise ValueError(f"system {field} must be an integer")
    return value


def _as_fraction(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"system {field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"system {field} must be finite")
    return result


def _from_json(item: object) -> SystemBounds:
    if not isinstance(item, dict):
        raise ValueError("every system bound must be an object")
    missing = {name for name in ("index", "page", "top", "bottom") if name not in item}
    if missing:
        raise ValueError("system bound is missing " + ", ".join(sorted(missing)))
    bound = SystemBounds(
        index=_as_int(item["index"], "index"),
        page=_as_int(item["page"], "page"),
        top=_as_fraction(item["top"], "top"),
        bottom=_as_fraction(item["bottom"], "bottom"),
        measure_start=_as_int(item.get("measure_start", 0), "measure_start"),
        measure_end=_as_int(item.get("measure_end", 0), "measure_end"),
    )
    if bound.index < 1:
        raise ValueError("system index must be at least 1")
    if bound.page < 1:
        raise ValueError(f"system {bound.index}: page must be at least 1")
    if not 0 <= bound.top < bound.bottom <= 1:
        raise ValueError(
            f"system {bound.index}: require 0 <= top < bottom <= 1, "
            f"got {bound.top}..{bound.bottom}"
        )
    return bound


def validate_system_bounds(bounds: list[SystemBounds]) -> list[SystemBounds]:
    """Validate score order without silently sorting or dropping any band."""
    if not bounds:
        raise ValueError("system bounds contain no systems")
    previous: SystemBounds | None = None
    for bound in bounds:
        if previous is not None:
            if bound.index <= previous.index:
                raise ValueError("system indices must be strictly increasing in score order")
            if bound.page < previous.page or (
                bound.page == previous.page and bound.top < previous.top
            ):
                raise ValueError("system bounds must be listed in page/top order")
        previous = bound
    return bounds


def load_system_bounds(path: str) -> list[SystemBounds]:
    """Read the choir app's ``.systems.json`` format."""
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read system bounds {path}: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("systems"), list):
        raise ValueError("system bounds must be a JSON object with a 'systems' list")
    return validate_system_bounds([_from_json(item) for item in data["systems"]])


def select_system(bounds: list[SystemBounds], index: int | None) -> list[SystemBounds]:
    """Select one score-wide system index, or retain the complete ordered list."""
    if index is None:
        return bounds
    if index < 1:
        raise ValueError("system index must be at least 1")
    selected = [bound for bound in bounds if bound.index == index]
    if not selected:
        raise ValueError(f"system index {index} is not present in the bounds file")
    return selected


def _render_page(page: object, dpi: int) -> NDArray:
    # pypdfium2's page object is intentionally kept out of the public type
    # surface: its concrete wrapper type has changed between releases.
    bitmap = page.render(scale=dpi / 72.0)  # type: ignore[attr-defined]
    rgb = np.array(bitmap.to_pil().convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _crop(page: NDArray, bound: SystemBounds) -> NDArray:
    """Use the same fractional-to-pixel rounding as the choir app."""
    height = page.shape[0]
    top = max(0, min(height - 1, int(height * bound.top)))
    band = max(1, min(height - top, int(height * bound.bottom) - top))
    return page[top : top + band, :]


def render_system_crops(
    pdf_path: str,
    bounds: list[SystemBounds],
    out_dir: str,
    dpi: int = DEFAULT_SYSTEM_DPI,
) -> list[SystemCrop]:
    """Render only the requested PDF pages and write one PNG per printed system."""
    if dpi < 1:
        raise ValueError("system dpi must be at least 1")
    bounds = validate_system_bounds(bounds)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(pdf_path)
    if not document:
        raise ValueError(f"invalid PDF {pdf_path}")
    pages: dict[int, NDArray] = {}
    result: list[SystemCrop] = []
    try:
        for bound in bounds:
            if bound.page > len(document):
                raise ValueError(
                    f"system {bound.index}: page {bound.page} does not exist; "
                    f"PDF has {len(document)} page(s)"
                )
            if bound.page not in pages:
                pages[bound.page] = _render_page(document[bound.page - 1], dpi)
            image = _crop(pages[bound.page], bound)
            path = os.path.join(out_dir, f"system-{bound.index:03d}@{dpi}.png")
            if not cv2.imwrite(path, image):
                raise OSError(f"could not write system crop {path}")
            result.append(SystemCrop(bound, path))
    finally:
        document.close()
    return result
