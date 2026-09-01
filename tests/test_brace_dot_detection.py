import unittest

import numpy as np

from homr.bounding_boxes import RotatedBoundingBox
from homr.brace_dot_detection import _create_grandstaffs
from homr.model import MultiStaff, Staff, StaffPoint


def make_staff(number: int) -> Staff:
    y_points = [10 * i + 100 * float(number) for i in range(5)]
    return Staff([StaffPoint(0.0, y_points, 0)])


def make_brace(upper_staff_index: int, lower_staff_index: int) -> RotatedBoundingBox:
    min_y = upper_staff_index * 100
    max_y = lower_staff_index * 100 + 40  # a staff is 40px high, see make_staff()
    rect = ((0.0, (min_y + max_y) / 2), (10.0, max_y - min_y), 0.0)
    return RotatedBoundingBox(rect, np.empty((0, 0)))


def make_system(*numbers: int) -> MultiStaff:
    return MultiStaff([make_staff(number) for number in numbers], [])


def staff_counts(systems: list[MultiStaff]) -> list[int]:
    return [len(system.staffs) for system in systems]


class TestCreateGrandstaffs(unittest.TestCase):
    def test_a_pairing_that_makes_the_systems_disagree_is_dropped(self) -> None:
        """B5: four staffs a system, and a brace found in only one of them."""
        systems = [make_system(0, 1, 2, 3), make_system(4, 5, 6, 7), make_system(8, 9, 10, 11)]

        result = _create_grandstaffs(systems, [make_brace(4, 5)])

        self.assertEqual(staff_counts(result), [4, 4, 4])
        self.assertFalse(any(staff.is_grandstaff for system in result for staff in system.staffs))

    def test_pairing_every_system_the_same_way_is_kept(self) -> None:
        """The two-staff engravings this repertoire is made of: one grand staff a system."""
        systems = [make_system(0, 1), make_system(2, 3), make_system(4, 5)]

        result = _create_grandstaffs(
            systems, [make_brace(0, 1), make_brace(2, 3), make_brace(4, 5)]
        )

        self.assertEqual(staff_counts(result), [1, 1, 1])
        self.assertTrue(all(staff.is_grandstaff for system in result for staff in system.staffs))

    def test_pairing_is_dropped_on_a_page_the_brackets_already_read_as_ragged(self) -> None:
        """B4: the staffs are reported as printed rather than welded into pairs."""
        systems = [make_system(0, 1), make_system(2, 3, 4)]

        result = _create_grandstaffs(systems, [make_brace(0, 1)])

        self.assertEqual(staff_counts(result), [2, 3])
        self.assertFalse(any(staff.is_grandstaff for system in result for staff in system.staffs))

    def test_no_systems(self) -> None:
        self.assertEqual(_create_grandstaffs([], [make_brace(0, 1)]), [])
