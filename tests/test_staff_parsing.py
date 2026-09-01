import unittest
from unittest import mock

import numpy as np

from homr.debug import Debug
from homr.model import MultiStaff, Staff, StaffPoint
from homr.staff_parsing import parse_staffs
from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol


def make_staff(number: int) -> Staff:
    y_points = [10 * i + 100 * float(number) for i in range(5)]
    return Staff([StaffPoint(float(x), [y + x for y in y_points], 0) for x in range(0, 100, 10)])


def make_system(*numbers: int) -> MultiStaff:
    return MultiStaff([make_staff(number) for number in numbers], [])


class FakeParse:
    """Stands in for the transformer: one note per staff, saying which staff it read."""

    def __init__(self) -> None:
        self.calls: list[Staff] = []

    def __call__(
        self, debug: Debug, index: int, staff: Staff, image: object, regions: object, config: object
    ) -> list[EncodedSymbol]:
        self.calls.append(staff)
        return [EncodedSymbol("note_4", "C4", "_", "_", "_", "upper"), EncodedSymbol("barline")]


def parse(systems: list[MultiStaff], selected_staff: int = -1) -> tuple[object, FakeParse]:
    fake = FakeParse()
    image = np.zeros((1000, 200, 3), dtype=np.uint8)
    with mock.patch("homr.staff_parsing.parse_staff_image", fake):
        parsed = parse_staffs(
            mock.MagicMock(spec=Debug),
            systems,
            image,
            config=Config(),
            selected_staff=selected_staff,
        )
    return parsed, fake


class TestParseStaffs(unittest.TestCase):
    def test_a_rectangular_score_is_still_assembled_by_position(self) -> None:
        systems = [make_system(0, 1), make_system(2, 3), make_system(4, 5)]

        parsed, fake = parse(systems)

        self.assertEqual(len(parsed.parts), 2)
        self.assertIsNone(parsed.system_sizes)
        # part 0 is staff 0 of every system, part 1 is staff 1 of every system
        self.assertEqual(
            fake.calls,
            [
                systems[0].staffs[0],
                systems[1].staffs[0],
                systems[2].staffs[0],
                systems[0].staffs[1],
                systems[1].staffs[1],
                systems[2].staffs[1],
            ],
        )

    def test_systems_of_different_sizes_are_reported_as_they_are(self) -> None:
        systems = [make_system(0, 1), make_system(2, 3, 4), make_system(5, 6)]

        parsed, fake = parse(systems)

        self.assertEqual(parsed.system_sizes, [2, 3, 2])
        self.assertEqual(len(parsed.parts), 7)
        # every staff is read once, in reading order, and none is dropped
        self.assertEqual(fake.calls, [staff for system in systems for staff in system.staffs])

    def test_a_system_of_a_different_size_is_not_deleted(self) -> None:
        """The old code dropped a first or last system whose staff count differed."""
        systems = [make_system(0), make_system(1, 2), make_system(3, 4)]

        parsed, _ = parse(systems)

        self.assertEqual(parsed.system_sizes, [1, 2, 2])

    def test_groups_are_not_broken_into_one_part_per_staff_of_the_page(self) -> None:
        """B5's collapse: nine singletons became a single monophonic part."""
        systems = [make_system(0, 1, 2, 3), make_system(4, 5), make_system(6, 7, 8)]

        parsed, _ = parse(systems)

        self.assertEqual(parsed.system_sizes, [4, 2, 3])
        self.assertEqual(len(parsed.parts), 9)

    def test_a_staff_the_transformer_read_nothing_from_is_left_out_of_its_system(self) -> None:
        systems = [make_system(0, 1), make_system(2, 3, 4)]
        fake = FakeParse()
        empty_staff = systems[1].staffs[1]

        def parse_one(
            debug: Debug, index: int, staff: Staff, image: object, regions: object, config: object
        ) -> list[EncodedSymbol]:
            if staff is empty_staff:
                return []
            return fake(debug, index, staff, image, regions, config)

        with mock.patch("homr.staff_parsing.parse_staff_image", parse_one):
            parsed = parse_staffs(
                mock.MagicMock(spec=Debug),
                systems,
                np.zeros((1000, 200, 3), dtype=np.uint8),
                config=Config(),
                selected_staff=-1,
            )

        self.assertEqual(parsed.system_sizes, [2, 2])
        self.assertEqual(len(parsed.parts), 4)

    def test_selected_staff_picks_one_system_of_a_ragged_page(self) -> None:
        systems = [make_system(0, 1), make_system(2, 3, 4), make_system(5, 6)]

        parsed, fake = parse(systems, selected_staff=1)

        self.assertEqual(parsed.system_sizes, [3])
        self.assertEqual(fake.calls, systems[1].staffs)
