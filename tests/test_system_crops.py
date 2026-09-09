from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from homr.system_crops import (
    SystemBounds,
    load_system_bounds,
    render_system_crops,
    select_system,
    validate_system_bounds,
)


def _pdf(path: Path) -> None:
    first = Image.new("RGB", (120, 200), "red")
    second = Image.new("RGB", (120, 200), "white")
    draw = ImageDraw.Draw(second)
    draw.rectangle((0, 0, 119, 99), fill="green")
    draw.rectangle((0, 100, 119, 199), fill="blue")
    first.save(path, "PDF", save_all=True, append_images=[second], resolution=72.0)


def test_loads_choir_systems_json_and_preserves_score_order(tmp_path: Path) -> None:
    path = tmp_path / ".systems.json"
    path.write_text(
        json.dumps(
            {
                "systems": [
                    {
                        "index": 3,
                        "page": 1,
                        "top": 0.1,
                        "bottom": 0.4,
                        "measure_start": 9,
                        "measure_end": 12,
                    },
                    {"index": 7, "page": 2, "top": 0.2, "bottom": 0.5},
                ]
            }
        )
    )

    assert load_system_bounds(str(path)) == [
        SystemBounds(3, 1, 0.1, 0.4, 9, 12),
        SystemBounds(7, 2, 0.2, 0.5),
    ]


@pytest.mark.parametrize(
    "systems, message",
    [
        ([], "no systems"),
        ([{"index": 1, "page": 1, "top": 0.4, "bottom": 0.4}], "top < bottom"),
        ([{"index": 0, "page": 1, "top": 0.1, "bottom": 0.2}], "index"),
        ([{"index": 1, "page": 0, "top": 0.1, "bottom": 0.2}], "page"),
        (
            [
                {"index": 2, "page": 1, "top": 0.1, "bottom": 0.2},
                {"index": 1, "page": 1, "top": 0.3, "bottom": 0.4},
            ],
            "strictly increasing",
        ),
        (
            [
                {"index": 1, "page": 2, "top": 0.1, "bottom": 0.2},
                {"index": 2, "page": 1, "top": 0.3, "bottom": 0.4},
            ],
            "page/top order",
        ),
    ],
)
def test_invalid_bounds_are_refused(tmp_path: Path, systems: list[dict], message: str) -> None:
    path = tmp_path / "bounds.json"
    path.write_text(json.dumps({"systems": systems}))
    with pytest.raises(ValueError, match=message):
        load_system_bounds(str(path))


def test_selection_uses_score_wide_index_without_reordering() -> None:
    bounds = [SystemBounds(3, 1, 0.1, 0.2), SystemBounds(8, 2, 0.3, 0.4)]
    assert select_system(bounds, None) is bounds
    assert select_system(bounds, 8) == [bounds[1]]
    with pytest.raises(ValueError, match="not present"):
        select_system(bounds, 4)


def test_render_uses_requested_pdf_page_and_fractional_band(tmp_path: Path) -> None:
    pdf = tmp_path / "two-pages.pdf"
    _pdf(pdf)
    bounds = [
        SystemBounds(1, 2, 0.0, 0.5),
        SystemBounds(2, 2, 0.5, 1.0),
    ]

    crops = render_system_crops(str(pdf), bounds, str(tmp_path / "out"), dpi=72)

    assert [crop.index for crop in crops] == [1, 2]
    images = [cv2.imread(crop.path) for crop in crops]
    assert all(image is not None for image in images)
    upper, lower = images
    assert upper is not None and lower is not None
    # Pillow's PDF writer may round a page by a pixel; the bounds must still
    # divide the actual rendered page into two equal-ish halves at full width.
    assert abs(upper.shape[0] - lower.shape[0]) <= 1
    assert upper.shape[1] == lower.shape[1]
    assert upper.shape[0] >= 95
    # BGR: the upper half is green and the lower half blue. This proves page 2
    # was selected and the vertical fractions were not applied to the page stack.
    assert np.mean(upper[:, :, 1]) > np.mean(upper[:, :, 0]) + 40
    assert np.mean(lower[:, :, 0]) > np.mean(lower[:, :, 1]) + 40


def test_missing_pdf_page_is_an_error_not_a_skipped_system(tmp_path: Path) -> None:
    pdf = tmp_path / "two-pages.pdf"
    _pdf(pdf)
    with pytest.raises(ValueError, match="does not exist"):
        render_system_crops(
            str(pdf), [SystemBounds(1, 3, 0.1, 0.2)], str(tmp_path / "out"), dpi=72
        )


def test_validate_does_not_sort_a_misordered_page(tmp_path: Path) -> None:
    bounds = [SystemBounds(1, 1, 0.7, 0.8), SystemBounds(2, 1, 0.2, 0.3)]
    with pytest.raises(ValueError, match="page/top order"):
        validate_system_bounds(bounds)
