# ruff: noqa: S101

"""Regrouping may preserve detected music or reject it, never silently trim it."""

from itertools import product
from unittest.mock import Mock

import numpy as np
import pytest

from homr import staff_parsing as parsing
from homr.errors import IncompleteRecognitionError
from homr.model import MultiStaff, Staff
from tests.test_staff_parsing import make_row, make_staff


def flat_staffs(rows: list[MultiStaff]) -> list[Staff]:
    return [staff for row in rows for staff in row.staffs]


@pytest.mark.parametrize(
    ("layout", "leading", "trailing"),
    [
        ("FTTTT", 1, 0),
        ("TTTTF", 0, 1),
        ("FTTTTF", 1, 1),
        ("FFFFTF", 0, 2),
        ("FTFFFF", 2, 0),
    ],
)
def test_omissions_are_reported_without_mutating_the_detected_music(
    layout: str, leading: int, trailing: int
) -> None:
    rows = [make_row(i, flag == "T") for i, flag in enumerate(layout)]
    original_staffs = flat_staffs(rows)
    original_rows = [list(row.staffs) for row in rows]
    for staff in original_staffs:
        # Opaque note evidence must stay attached to its original staff.
        staff.symbols.append(Mock())
    original_symbols = [list(staff.symbols) for staff in original_staffs]

    message = f"{leading} leading and {trailing} trailing"
    with pytest.raises(IncompleteRecognitionError, match=message) as error:
        parsing._ensure_same_number_of_staffs(rows)

    assert f"out of {len(layout)}" in str(error.value)
    assert "scan each system separately" in str(error.value)
    assert [row.staffs for row in rows] == original_rows
    assert [staff.symbols for staff in original_staffs] == original_symbols
    assert [id(staff) for staff in flat_staffs(rows)] == [id(staff) for staff in original_staffs]


@pytest.mark.parametrize("layout", ["FTTTT", "TTTTF", "FTTTTF"])
def test_a_lossy_layout_stops_before_staff_recognition(
    monkeypatch: pytest.MonkeyPatch, layout: str
) -> None:
    rows = [make_row(i, flag == "T") for i, flag in enumerate(layout)]
    read = Mock(side_effect=AssertionError("Recognition must not run on a rejected layout"))
    regions = Mock()
    monkeypatch.setattr(parsing, "parse_staff_image", read)
    monkeypatch.setattr(parsing, "StaffRegions", regions)

    with pytest.raises(IncompleteRecognitionError, match="Staff layout would discard"):
        parsing.parse_staffs(Mock(), rows, np.zeros((1, 1)), Mock())

    read.assert_not_called()
    regions.assert_not_called()


@pytest.mark.parametrize("premerged", [False, True])
def test_lossless_regrouping_preserves_staff_identity_and_order(premerged: bool) -> None:
    staffs = [make_staff(i, is_grandstaff=bool(i % 2)) for i in range(8)]
    rows = [MultiStaff([staff], []) for staff in staffs]
    if premerged:
        rows = [MultiStaff(staffs[:2], []), *rows[2:]]

    result = parsing._ensure_same_number_of_staffs(rows)

    assert [len(row.staffs) for row in result] == [2, 2, 2, 2]
    assert [id(staff) for staff in flat_staffs(result)] == [id(staff) for staff in staffs]
    assert [id(staff) for staff in flat_staffs(rows)] == [id(staff) for staff in staffs]


def test_uniform_independent_parts_keep_the_original_system_grouping() -> None:
    rows = [MultiStaff([make_staff(4 * row + i) for i in range(4)], []) for row in range(2)]
    assert parsing._ensure_same_number_of_staffs(rows) is rows


def test_small_layouts_either_preserve_every_staff_or_explicitly_fail() -> None:
    # Exhaust all 510 nonempty binary signatures of up to eight detected groups.
    # This checks preservation, not whether an inferred part assignment is correct.
    for length in range(1, 9):
        for flags in product((False, True), repeat=length):
            rows = [make_row(i, flag) for i, flag in enumerate(flags)]
            before = [id(staff) for staff in flat_staffs(rows)]
            try:
                result = parsing._ensure_same_number_of_staffs(rows)
            except IncompleteRecognitionError:
                core = parsing._find_periodic_core(flat_staffs(rows))
                assert core is not None and (core[1] or core[2]), flags
            else:
                assert [id(staff) for staff in flat_staffs(result)] == before, flags
            assert [id(staff) for staff in flat_staffs(rows)] == before, flags
