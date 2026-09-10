from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from homr import system_bounds
from homr.system_crops import SystemBounds


def _staff(top: float, bottom: float) -> system_bounds.Box:
    return {"top": top, "bottom": bottom, "left": 0.1, "right": 0.9}


def _bar(x: float, top: float, bottom: float) -> system_bounds.Box:
    return {"top": top, "bottom": bottom, "left": x - 0.001, "right": x + 0.001}


def test_barline_agreement_groups_staves_before_gap_size() -> None:
    staves = [
        _staff(0.10, 0.14),
        _staff(0.20, 0.24),
        _staff(0.31, 0.35),
        _staff(0.42, 0.46),
    ]
    bars = [
        *[_bar(x, 0.10, 0.24) for x in (0.30, 0.50, 0.70)],
        *[_bar(x, 0.31, 0.46) for x in (0.25, 0.55, 0.78)],
    ]

    systems = system_bounds.group_staves(staves, bars)

    assert [len(system) for system in systems] == [2, 2]


def test_small_gap_veto_keeps_a_staff_when_its_barlines_disagree() -> None:
    staves = [
        _staff(0.10, 0.14),
        _staff(0.22, 0.26),
        _staff(0.42, 0.46),
        _staff(0.53, 0.57),
    ]
    bars = [
        *[_bar(x, 0.10, 0.26) for x in (0.30, 0.50)],
        *[_bar(x, 0.42, 0.46) for x in (0.35, 0.55)],
        _bar(0.62, 0.53, 0.57),
    ]

    systems = system_bounds.group_staves(staves, bars)

    assert [len(system) for system in systems] == [2, 2]


def test_gap_fallback_uses_the_largest_step_when_no_barlines_exist() -> None:
    staves = [
        _staff(0.10, 0.14),
        _staff(0.19, 0.23),
        _staff(0.38, 0.42),
        _staff(0.47, 0.51),
    ]

    systems = system_bounds.group_staves(staves, [])

    assert [len(system) for system in systems] == [2, 2]


def test_band_edges_are_contiguous_and_page_relative() -> None:
    staves = [
        _staff(0.10, 0.14),
        _staff(0.20, 0.24),
        _staff(0.40, 0.44),
        _staff(0.50, 0.54),
    ]
    bars = [
        *[_bar(x, 0.10, 0.24) for x in (0.30, 0.50, 0.70)],
        *[_bar(x, 0.40, 0.54) for x in (0.25, 0.55, 0.78)],
    ]

    bands = system_bounds.bands_for_page(3, staves, bars)

    assert [band.page for band in bands] == [3, 3]
    assert [band.index for band in bands] == [1, 2]
    assert bands[0].bottom == pytest.approx(bands[1].top)
    assert 0 <= bands[0].top < bands[0].bottom < bands[1].bottom <= 1


class _FakePdf:
    def __init__(self, _path: str) -> None:
        self.pages = [object(), object()]
        self.closed = False

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, index: int):
        return self.pages[index]

    def __bool__(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


def test_pdf_proposal_preserves_page_order_and_uses_score_wide_indices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rendered: list[tuple[str, int, str, int]] = []
    layouts = iter(
        [
            {"staves": [_staff(0.1, 0.2)], "bar_lines": []},
            {"staves": [_staff(0.2, 0.3), _staff(0.6, 0.7)], "bar_lines": []},
        ]
    )
    fake_pdf = _FakePdf("unused")
    monkeypatch.setattr(system_bounds.pdfium, "PdfDocument", lambda _path: fake_pdf)
    monkeypatch.setattr(
        system_bounds,
        "_render_pdf_page",
        lambda pdf, page, path, dpi: rendered.append((pdf, page, path, dpi)),
    )
    monkeypatch.setattr(
        system_bounds,
        "detect_staff_layout",
        lambda *_args, **_kwargs: next(layouts),
    )

    pdf_path = str(tmp_path / "score.pdf")
    bounds = system_bounds.propose_system_bounds(pdf_path)

    assert [(bound.index, bound.page) for bound in bounds] == [(1, 1), (2, 2)]
    assert [(item[0], item[1], item[3]) for item in rendered] == [
        (pdf_path, 1, 200),
        (pdf_path, 2, 200),
    ]
    assert fake_pdf.closed


def test_machine_json_is_stable_and_matches_system_bounds_schema() -> None:
    text = system_bounds.proposal_json(
        [SystemBounds(index=1, page=2, top=0.125, bottom=0.5)], dpi=200
    )

    assert text == (
        '{"dpi":200,"systems":[{"bottom":0.5,"index":1,"page":2,"top":0.125}],"version":1}'
    )
    assert json.loads(text)["systems"][0] == {
        "index": 1,
        "page": 2,
        "top": 0.125,
        "bottom": 0.5,
    }


def test_module_cli_writes_only_json_to_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        system_bounds,
        "propose_system_bounds",
        lambda *_args, **_kwargs: [SystemBounds(1, 1, 0.1, 0.9)],
    )
    monkeypatch.setattr(sys, "argv", ["homr.system_bounds", "score.pdf"])

    system_bounds.main()

    captured = capsys.readouterr()
    assert json.loads(captured.out)["systems"] == [
        {"index": 1, "page": 1, "top": 0.1, "bottom": 0.9}
    ]
    assert captured.err == ""
