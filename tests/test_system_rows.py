"""Staff groups printed as one system under two brackets are read as one system.

eerovil/musescore-choir-plugins#274: Finlandia (arr. Morgan) prints its eight
staves as one system with the tenors and the basses under separate brackets.
Nothing spans the break, so the bracket rules left two groups of four, and two
groups are read as two systems in a row: four parts of ten bars, the basses'
five bars appended to the tenors'. These pin the join on barlines, and the
refusals that keep it from fusing two systems that merely look alike.
"""

import unittest

import numpy as np

from homr.bounding_boxes import RotatedBoundingBox
from homr.brace_dot_detection import join_rows_sharing_bar_lines, record_bar_lines
from homr.model import MultiStaff, Staff, StaffPoint

LEFT, RIGHT = 300.0, 1700.0


def staff(top: float) -> Staff:
    lines = [top + 10 * i for i in range(5)]
    return Staff([StaffPoint(float(x), lines, 0) for x in np.arange(LEFT, RIGHT + 1, 50.0)])


def bar(x: float, top: float, bottom: float) -> RotatedBoundingBox:
    return RotatedBoundingBox(((x, (top + bottom) / 2), (3, bottom - top), 0), np.empty((0,)))


def bars_on(staves: list[Staff], xs: list[float]) -> list[RotatedBoundingBox]:
    return [bar(x, s.min_y, s.max_y) for s in staves for x in xs]


def join(rows: list[MultiStaff], lines: list[RotatedBoundingBox]) -> list[MultiStaff]:
    record_bar_lines([s for row in rows for s in row.staffs], lines)
    return join_rows_sharing_bar_lines(rows)


class TestJoinRowsSharingBarLines(unittest.TestCase):
    def test_two_bracket_groups_with_the_same_bars_are_one_system(self) -> None:
        tenors = [staff(100 + 120 * i) for i in range(4)]
        basses = [staff(650 + 120 * i) for i in range(4)]
        lines = bars_on(tenors + basses, [634, 890, 1110, 1374, 1636])
        rows = join([MultiStaff(basses, []), MultiStaff(tenors, [])], lines)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].staffs, tenors + basses)

    def test_a_double_bar_line_is_one_place(self) -> None:
        tenors = [staff(100)]
        basses = [staff(300)]
        lines = bars_on(tenors + basses, [634, 890, 1636]) + bars_on(tenors, [1642])
        rows = join([MultiStaff(tenors, []), MultiStaff(basses, [])], lines)
        self.assertEqual(len(rows), 1)

    def test_two_systems_with_different_bars_stay_apart(self) -> None:
        first = [staff(100 + 120 * i) for i in range(4)]
        second = [staff(650 + 120 * i) for i in range(4)]
        lines = bars_on(first, [634, 890, 1110, 1374]) + bars_on(second, [700, 1000, 1300, 1550])
        rows = join([MultiStaff(first, []), MultiStaff(second, [])], lines)
        self.assertEqual([r.staffs for r in rows], [first, second])

    def test_a_majority_of_shared_bar_lines_is_not_enough(self) -> None:
        first = [staff(100)]
        second = [staff(300)]
        lines = bars_on(first, [600, 900, 1200]) + bars_on(second, [600, 900, 1250])
        rows = join([MultiStaff(first, []), MultiStaff(second, [])], lines)
        self.assertEqual(len(rows), 2)

    def test_a_different_number_of_bars_stays_apart(self) -> None:
        first = [staff(100)]
        second = [staff(300)]
        lines = bars_on(first, [600, 900]) + bars_on(second, [600, 900, 1200])
        rows = join([MultiStaff(first, []), MultiStaff(second, [])], lines)
        self.assertEqual(len(rows), 2)

    def test_one_shared_bar_line_is_too_little_evidence(self) -> None:
        first = [staff(100)]
        second = [staff(300)]
        lines = bars_on(first + second, [900])
        rows = join([MultiStaff(first, []), MultiStaff(second, [])], lines)
        self.assertEqual(len(rows), 2)

    def test_the_closing_bar_lines_do_not_count(self) -> None:
        first = [staff(100)]
        second = [staff(300)]
        lines = bars_on(first + second, [LEFT + 5, RIGHT - 5]) + bars_on(first, [800, 1200])
        rows = join([MultiStaff(first, []), MultiStaff(second, [])], lines)
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
