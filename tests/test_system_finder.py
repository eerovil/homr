from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from homr import main, system_finder
from homr.system_crops import SystemBounds
from homr.system_finder import NormalizedBox


def staff(top: float, bottom: float, left: float = 0.1, right: float = 0.9) -> NormalizedBox:
    return {"top": top, "bottom": bottom, "left": left, "right": right}


def barline(staff_box: NormalizedBox, x: float) -> NormalizedBox:
    return {
        "top": staff_box["top"],
        "bottom": staff_box["bottom"],
        "left": x - 0.001,
        "right": x + 0.001,
    }


def page(
    *rows: tuple[float, float, list[float]]
) -> tuple[list[NormalizedBox], list[NormalizedBox]]:
    staves: list[NormalizedBox] = []
    bars: list[NormalizedBox] = []
    for top, bottom, xs in rows:
        one = staff(top, bottom)
        staves.append(one)
        bars.extend(barline(one, x) for x in xs)
    return staves, bars


def test_staves_carrying_the_same_bars_are_one_system() -> None:
    staves, bars = page(
        (0.10, 0.14, [0.3, 0.5, 0.7]),
        (0.22, 0.26, [0.3, 0.5, 0.7]),
    )
    assert len(system_finder.group_staves(staves, bars)) == 1


def test_staves_carrying_different_bars_are_different_systems() -> None:
    staves, bars = page(
        (0.10, 0.14, [0.3, 0.5, 0.7]),
        (0.22, 0.26, [0.3, 0.5, 0.7]),
        (0.34, 0.38, [0.4, 0.6]),
        (0.46, 0.50, [0.4, 0.6]),
    )
    assert [len(system) for system in system_finder.group_staves(staves, bars)] == [2, 2]


def test_opening_and_closing_lines_are_not_evidence() -> None:
    ends, bars = page((0.10, 0.14, [0.1, 0.9]))
    assert system_finder._interior_barlines(ends[0], bars) == []

    staves, bars = page(
        (0.10, 0.14, [0.1, 0.5, 0.9]),
        (0.22, 0.26, [0.1, 0.5, 0.9]),
        (0.34, 0.38, [0.1, 0.3, 0.9]),
        (0.46, 0.50, [0.1, 0.3, 0.9]),
    )
    assert [len(system) for system in system_finder.group_staves(staves, bars)] == [2, 2]


def test_a_break_needs_the_white_to_go_with_it() -> None:
    staves, bars = page(
        (0.10, 0.14, [0.3, 0.5]),
        (0.22, 0.26, [0.3, 0.5]),
        (0.42, 0.46, [0.35, 0.55]),
        (0.53, 0.57, [0.62]),
    )
    assert [len(system) for system in system_finder.group_staves(staves, bars)] == [2, 2]


def test_a_staff_with_no_barline_is_decided_by_the_gap() -> None:
    staves, bars = page(
        (0.10, 0.14, [0.3, 0.5]),
        (0.21, 0.25, []),
        (0.50, 0.54, [0.4]),
        (0.62, 0.66, [0.4]),
    )
    assert [len(system) for system in system_finder.group_staves(staves, bars)] == [2, 2]


def test_one_staff_on_the_page_is_one_system() -> None:
    staves, bars = page((0.10, 0.14, [0.3]))
    assert [len(system) for system in system_finder.group_staves(staves, bars)] == [1]


def test_a_boundary_is_halfway_between_two_systems() -> None:
    staves, bars = page(
        (0.10, 0.14, [0.3, 0.5]),
        (0.22, 0.26, [0.3, 0.5]),
        (0.40, 0.44, [0.6]),
        (0.52, 0.56, [0.6]),
    )
    bands = system_finder.bands_for_page(2, staves, bars)
    assert [band.index for band in bands] == [1, 2]
    assert all(band.page == 2 for band in bands)
    assert bands[0].bottom == pytest.approx(0.33)
    assert bands[1].top == pytest.approx(0.33)


def test_outer_edges_are_given_room_and_clamped() -> None:
    staves, bars = page(
        (0.02, 0.06, [0.3, 0.5]),
        (0.14, 0.18, [0.3, 0.5]),
        (0.80, 0.84, [0.6]),
        (0.92, 0.96, [0.6]),
    )
    bands = system_finder.bands_for_page(1, staves, bars)
    assert bands[0].top == 0.0
    assert bands[-1].bottom == 1.0


def test_page_with_no_staves_proposes_nothing() -> None:
    assert system_finder.bands_for_page(1, [], []) == []


def test_payload_matches_system_bounds_schema() -> None:
    payload = system_finder.bounds_payload([SystemBounds(index=1, page=2, top=0.1, bottom=0.4)])
    assert payload == {
        "systems": [
            {
                "index": 1,
                "page": 2,
                "top": 0.1,
                "bottom": 0.4,
                "measure_start": 0,
                "measure_end": 0,
            }
        ]
    }


def _two_page_pdf(path: Path) -> None:
    first = Image.new("RGB", (120, 200), "white")
    first_draw = ImageDraw.Draw(first)
    first_draw.rectangle((0, 0, 119, 99), fill="red")
    second = Image.new("RGB", (120, 200), "white")
    second_draw = ImageDraw.Draw(second)
    second_draw.rectangle((0, 100, 119, 199), fill="blue")
    first.save(path, "PDF", save_all=True, append_images=[second], resolution=72.0)


def _fake_geometry(
    image: np.ndarray, source_name: str = "", use_gpu: bool = False
) -> system_finder.PageGeometry:
    one = staff(0.15, 0.20)
    two = staff(0.35, 0.40)
    bars = [barline(one, 0.3), barline(two, 0.3)]
    return {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "staves": [one, two],
        "bar_lines": bars,
    }


def test_pdf_pages_are_proposed_independently_in_score_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "two-pages.pdf"
    _two_page_pdf(pdf)
    calls: list[tuple[str, bool]] = []

    def detect(
        image: np.ndarray, source_name: str = "", use_gpu: bool = False
    ) -> system_finder.PageGeometry:
        calls.append((source_name, use_gpu))
        return _fake_geometry(image, source_name, use_gpu)

    monkeypatch.setattr(system_finder, "detect_page_geometry", detect)
    found = system_finder.find_system_bounds(str(pdf), use_gpu=True, dpi=72)

    assert len(calls) == 2
    assert all(use_gpu for _, use_gpu in calls)
    assert calls[0][0].endswith("#page-1")
    assert calls[1][0].endswith("#page-2")
    assert [(bound.index, bound.page) for bound in found] == [(1, 1), (2, 2)]


def test_one_page_can_be_requested_for_short_compute_leases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "two-pages.pdf"
    _two_page_pdf(pdf)
    calls: list[str] = []

    def detect(
        image: np.ndarray, source_name: str = "", use_gpu: bool = False
    ) -> system_finder.PageGeometry:
        calls.append(source_name)
        return _fake_geometry(image, source_name, use_gpu)

    monkeypatch.setattr(system_finder, "detect_page_geometry", detect)
    found = system_finder.find_system_bounds(str(pdf), dpi=72, page=2)

    assert len(calls) == 1 and calls[0].endswith("#page-2")
    assert [(bound.index, bound.page) for bound in found] == [(1, 2)]
    with pytest.raises(ValueError, match="does not exist"):
        system_finder.find_system_bounds(str(pdf), dpi=72, page=3)


def test_system_bound_cli_writes_json_only_to_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pdf = tmp_path / "score.pdf"
    pdf.write_bytes(b"pdf")
    monkeypatch.setattr(main, "download_weights", lambda *args: None)
    monkeypatch.setattr(
        main,
        "find_system_bounds",
        lambda *args, **kwargs: [SystemBounds(index=1, page=1, top=0.1, bottom=0.9)],
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["homer", str(pdf), "--gpu", "no", "--find-system-bounds"],
    )

    main.main()

    output = capsys.readouterr()
    assert json.loads(output.out) == {
        "systems": [
            {
                "index": 1,
                "page": 1,
                "top": 0.1,
                "bottom": 0.9,
                "measure_start": 0,
                "measure_end": 0,
            }
        ]
    }
    assert "Result was written" not in output.out


def test_invalid_proposal_input_is_refused() -> None:
    with pytest.raises(ValueError, match="PDF"):
        system_finder.find_system_bounds("page.png")
    with pytest.raises(ValueError, match="dpi"):
        system_finder.find_system_bounds("score.pdf", dpi=0)
