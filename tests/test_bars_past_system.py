"""A staff that decodes a bar past its system's end loses that bar, and only then.

eerovil/musescore-choir-plugins#274: on Finlandia (arr. Morgan) one staff of a
system decoded a whole note after its last barline, so it ran a bar longer than
the other staves and every later bar of the part sat beside the wrong music.
"""

import unittest

import numpy as np

from homr.model import MultiStaff, Staff, StaffPoint
from homr.staff_parsing import _drop_bars_past_the_system
from homr.transformer.vocabulary import EncodedSymbol


def staff(top: float, barlines: list[float]) -> Staff:
    lines = [top + 10 * i for i in range(5)]
    result = Staff([StaffPoint(float(x), lines, 0) for x in np.arange(300.0, 1701.0, 50.0)])
    result.bar_line_xs = [300.0, *barlines, 1700.0]
    return result


def tokens(*bars: str) -> list[EncodedSymbol]:
    out = [EncodedSymbol("clef_F4")]
    for bar in bars:
        out.append(EncodedSymbol("note_1", bar))
        out.append(EncodedSymbol("barline"))
    out.append(EncodedSymbol("newline"))
    return out


def pitches(row: list[EncodedSymbol]) -> list[str]:
    return [s.pitch for s in row if s.rhythm.startswith("note")]


class TestDropBarsPastTheSystem(unittest.TestCase):
    def test_an_extra_trailing_bar_is_dropped(self) -> None:
        staves = [staff(100, [700, 1100]), staff(300, [700, 1100])]
        rows = [[tokens("C3", "D3", "E3")], [tokens("C3", "D3", "E3", "E3")]]
        _drop_bars_past_the_system(rows, [MultiStaff(staves, [])])
        self.assertEqual(pitches(rows[1][0]), ["C3", "D3", "E3"])
        self.assertEqual(rows[1][0][-1].rhythm, "newline")
        self.assertEqual(pitches(rows[0][0]), ["C3", "D3", "E3"])

    def test_a_bar_the_page_prints_is_kept(self) -> None:
        # The other staff lost a bar; the barlines say this one is right.
        staves = [staff(100, [600, 900, 1200]), staff(300, [600, 900, 1200])]
        rows = [[tokens("C3", "D3", "E3")], [tokens("C3", "D3", "E3", "F3")]]
        _drop_bars_past_the_system(rows, [MultiStaff(staves, [])])
        self.assertEqual(pitches(rows[1][0]), ["C3", "D3", "E3", "F3"])

    def test_a_staff_with_as_many_bars_as_another_is_kept(self) -> None:
        staves = [staff(100, [700]), staff(300, [700]), staff(500, [700])]
        rows = [
            [tokens("C3", "D3", "E3")],
            [tokens("C3", "D3", "E3")],
            [tokens("C3", "D3")],
        ]
        _drop_bars_past_the_system(rows, [MultiStaff(staves, [])])
        self.assertEqual([len(pitches(r[0])) for r in rows], [3, 3, 2])

    def test_a_lone_staff_is_left_alone(self) -> None:
        rows = [[tokens("C3", "D3", "E3")]]
        _drop_bars_past_the_system(rows, [MultiStaff([staff(100, [700])], [])])
        self.assertEqual(len(pitches(rows[0][0])), 3)


if __name__ == "__main__":
    unittest.main()
