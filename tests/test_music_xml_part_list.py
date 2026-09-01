import unittest

import musicxml.xmlelement.xmlelement as mxl

from homr.music_xml_generator import build_part_list


def shape(part_list: mxl.XMLPartList) -> list[str]:
    """The part list as a flat sequence, saying what each entry is."""
    result = []
    for child in part_list.get_children():
        if isinstance(child, mxl.XMLPartGroup):
            result.append("group-" + str(child.type) + "-" + str(child.number))
        else:
            result.append("part-" + str(child.id))
    return result


class TestPartList(unittest.TestCase):
    def test_a_rectangular_score_has_no_system_groups(self) -> None:
        self.assertEqual(shape(build_part_list([False, False])), ["part-P1", "part-P2"])
        self.assertEqual(shape(build_part_list([False, False], None)), ["part-P1", "part-P2"])

    def test_each_printed_system_is_bracketed(self) -> None:
        """B4: five systems of 2, 3, 2, 3 and 3 staffs, in the order they are printed."""
        part_list = build_part_list([False] * 13, [2, 3, 2, 3, 3])

        self.assertEqual(
            shape(part_list),
            [
                "group-start-1",
                "part-P1",
                "part-P2",
                "group-stop-1",
                "group-start-2",
                "part-P3",
                "part-P4",
                "part-P5",
                "group-stop-2",
                "group-start-3",
                "part-P6",
                "part-P7",
                "group-stop-3",
                "group-start-4",
                "part-P8",
                "part-P9",
                "part-P10",
                "group-stop-4",
                "group-start-5",
                "part-P11",
                "part-P12",
                "part-P13",
                "group-stop-5",
            ],
        )

    def test_a_group_says_which_system_it_is(self) -> None:
        part_list = build_part_list([False] * 3, [1, 2])
        groups = [
            child
            for child in part_list.get_children()
            if isinstance(child, mxl.XMLPartGroup) and child.type == "start"
        ]

        names = [group.xml_group_name.value_ for group in groups]

        self.assertEqual(names, ["System 1", "System 2"])
