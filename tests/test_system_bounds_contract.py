from __future__ import annotations

import pytest

from homr import system_bounds


def _staff(top: float, bottom: float) -> system_bounds.Box:
    return {"top": top, "bottom": bottom, "left": 0.1, "right": 0.9}


def _bar(staff: system_bounds.Box, x: float) -> system_bounds.Box:
    return {
        "top": staff["top"],
        "bottom": staff["bottom"],
        "left": x - 0.001,
        "right": x + 0.001,
    }


def test_opening_and_closing_barlines_are_not_grouping_evidence() -> None:
    first = _staff(0.10, 0.14)
    assert system_bounds._interior_barlines(first, [_bar(first, 0.1), _bar(first, 0.9)]) == []

    staves = [_staff(0.10, 0.14), _staff(0.22, 0.26), _staff(0.34, 0.38), _staff(0.46, 0.50)]
    bars = [
        *[_bar(staves[0], x) for x in (0.1, 0.5, 0.9)],
        *[_bar(staves[1], x) for x in (0.1, 0.5, 0.9)],
        *[_bar(staves[2], x) for x in (0.1, 0.3, 0.9)],
        *[_bar(staves[3], x) for x in (0.1, 0.3, 0.9)],
    ]
    assert [len(group) for group in system_bounds.group_staves(staves, bars)] == [2, 2]


def test_staff_with_no_interior_barline_is_decided_by_gap() -> None:
    staves = [_staff(0.10, 0.14), _staff(0.21, 0.25), _staff(0.50, 0.54), _staff(0.62, 0.66)]
    bars = [
        *[_bar(staves[0], x) for x in (0.3, 0.5)],
        *[_bar(staves[2], x) for x in (0.4,)],
        *[_bar(staves[3], x) for x in (0.4,)],
    ]
    assert [len(group) for group in system_bounds.group_staves(staves, bars)] == [2, 2]


def test_zero_and_one_staff_pages_keep_the_existing_contract() -> None:
    assert system_bounds.bands_for_page(1, [], []) == []
    only = _staff(0.10, 0.14)
    assert [len(group) for group in system_bounds.group_staves([only], [_bar(only, 0.3)])] == [1]


def test_outer_band_edges_get_room_and_clamp_to_page() -> None:
    staves = [_staff(0.02, 0.06), _staff(0.14, 0.18), _staff(0.80, 0.84), _staff(0.92, 0.96)]
    bars = [
        *[_bar(staves[0], x) for x in (0.3, 0.5)],
        *[_bar(staves[1], x) for x in (0.3, 0.5)],
        *[_bar(staves[2], x) for x in (0.6,)],
        *[_bar(staves[3], x) for x in (0.6,)],
    ]

    bands = system_bounds.bands_for_page(1, staves, bars)

    assert bands[0].top == pytest.approx(0.0)
    assert bands[-1].bottom == pytest.approx(1.0)
    assert bands[0].bottom == pytest.approx(bands[1].top)
