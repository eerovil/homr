"""Reading a time signature's digits off the page (eerovil/musescore-choir-plugins#267).

The model's vocabulary has no numerator, and a meter change after a barline gets
no token at all, so `time_signature_reader` looks for the digits itself. These
tests draw staffs from the committed digit templates, so they need neither the
models nor a scanned page: the pages the reader was measured on (Legenda, 69 of
70 printed signatures found, none invented) are in copyright and stay off this
repository.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from homr import time_signature_reader
from homr.model import Staff, StaffPoint
from homr.time_signature_reader import (
    PrintedMeter,
    attach_printed_meters,
    classify_digit,
    find_time_signatures,
)
from homr.transformer.vocabulary import EncodedSymbol

UNIT = 20
TOP = 100
WIDTH = 1000


def lines() -> list[int]:
    return [TOP + k * UNIT for k in range(5)]


def blank_staff() -> np.ndarray:
    image = np.full((TOP + 7 * UNIT, WIDTH), 255, dtype=np.uint8)
    for y in lines():
        image[y - 1 : y + 2, 20 : WIDTH - 20] = 0
    return image


def staff() -> Staff:
    return Staff(
        [StaffPoint(float(x), [float(y) for y in lines()], 0.0) for x in range(20, 980, 10)]
    )


def template(digit: int, font: str = "Leipzig") -> np.ndarray:
    rows = time_signature_reader._templates()[digit][
        ["Bravura", "Leipzig", "Gootville"].index(font)
    ]
    return rows > 0


def draw_digit(image: np.ndarray, digit: int, left: int, top: int, font: str = "Leipzig") -> int:
    """Draw `digit` two staff spaces tall with its top at `top`; return its right edge."""
    shape = template(digit, font)
    height = 2 * UNIT + 2
    width = round(shape.shape[1] * height / shape.shape[0])
    scaled = (
        cv2.resize(shape.astype(np.uint8) * 255, (width, height), interpolation=cv2.INTER_AREA)
        > 127
    )
    image[top - 1 : top - 1 + height, left : left + width][scaled] = 0
    return left + width


def draw_signature(
    image: np.ndarray, numerator: int, denominator: int, left: int, font: str = "Leipzig"
) -> None:
    """Each number centred on the other, its digits a little apart, as engraved."""
    widths = {}
    for number in (numerator, denominator):
        scratch = blank_staff()
        x = 0
        for digit in str(number):
            x = draw_digit(scratch, int(digit), x, lines()[0], font) + UNIT // 4
        widths[number] = x - UNIT // 4
    widest = max(widths.values())
    for number, top in ((numerator, lines()[0]), (denominator, lines()[2])):
        x = left + (widest - widths[number]) // 2
        for digit in str(number):
            x = draw_digit(image, int(digit), x, top, font) + UNIT // 4


def draw_barline(image: np.ndarray, x: int) -> None:
    image[lines()[0] : lines()[4] + 1, x : x + 3] = 0


def draw_note_with_stem(image: np.ndarray, x: int, y: int) -> None:
    cv2.ellipse(image, (x, y), (UNIT * 6 // 10, UNIT // 2 - 1), -20, 0, 360, 0, -1)
    image[y - 4 * UNIT : y, x + UNIT // 2 : x + UNIT // 2 + 2] = 0


@pytest.mark.parametrize("font", ["Bravura", "Leipzig", "Gootville"])
def test_every_digit_is_told_from_the_others(font: str) -> None:
    for digit in range(10):
        shape = template(digit, font)
        found, score, runner_up = classify_digit(shape)
        assert found == digit
        assert score - runner_up > 0.1


def test_the_rows_a_staff_line_runs_through_are_left_out_of_the_match() -> None:
    """A line drawn through a digit is not part of its shape."""
    shape = template(3).copy()
    on_line = np.zeros(shape.shape[0], dtype=bool)
    on_line[[0, 15, 16, 31]] = True
    shape[on_line] = True
    assert classify_digit(shape, on_line)[0] == 3


def test_a_signature_at_the_head_and_one_after_a_barline_are_both_read() -> None:
    image = blank_staff()
    draw_signature(image, 3, 4, 120)
    draw_note_with_stem(image, 220, lines()[3])
    draw_barline(image, 400)
    draw_signature(image, 4, 4, 420)
    draw_note_with_stem(image, 520, lines()[2])

    found = find_time_signatures(image, staff())

    assert [(m.numerator, m.denominator) for m in found] == [(3, 4), (4, 4)]
    assert 120 < found[0].x < 160
    assert 420 < found[1].x < 460


def test_a_two_digit_numerator_is_read_as_one_number() -> None:
    image = blank_staff()
    draw_signature(image, 12, 8, 120)

    found = find_time_signatures(image, staff())

    assert [(m.numerator, m.denominator) for m in found] == [(12, 8)]


def test_notes_and_barlines_alone_are_not_read_as_a_signature() -> None:
    image = blank_staff()
    for x in range(120, 900, 60):
        draw_note_with_stem(image, x, lines()[1 + (x // 60) % 3])
    draw_barline(image, 500)

    assert find_time_signatures(image, staff()) == []


def test_a_denominator_no_meter_has_is_refused() -> None:
    image = blank_staff()
    draw_signature(image, 3, 7, 120)

    assert find_time_signatures(image, staff()) == []


def test_one_over_one_is_still_read_when_printed() -> None:
    image = blank_staff()
    draw_signature(image, 1, 1, 120)

    found = find_time_signatures(image, staff())

    assert [(m.numerator, m.denominator) for m in found] == [(1, 1)]


# --- placing what was read into the decoded stream ---


def barline(x: float) -> EncodedSymbol:
    symbol = EncodedSymbol("barline")
    symbol.image_coordinates = (x, 0.0)
    return symbol


def stream() -> list[EncodedSymbol]:
    """Clef, key, the decoder's one signature, then three bars of quarters."""
    quarter = [EncodedSymbol("note_4", "C4", position="upper")]
    return [
        EncodedSymbol("clef_G2", position="upper"),
        EncodedSymbol("keySignature_0"),
        EncodedSymbol("timeSignature/4"),
        *quarter * 3,
        barline(400),
        *quarter * 4,
        barline(700),
        *quarter * 3,
        barline(950),
    ]


def attach(monkeypatch: pytest.MonkeyPatch, meters: list[PrintedMeter]) -> list[EncodedSymbol]:
    monkeypatch.setattr(time_signature_reader, "find_time_signatures", lambda image, s: meters)
    return attach_printed_meters(stream(), np.zeros((1, 1)), staff(), lambda point: point)


def test_digits_at_the_decoders_own_signature_are_attached_to_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = attach(monkeypatch, [PrintedMeter(130, 3, 4, 0.8)])

    signatures = [s for s in out if s.rhythm.startswith("timeSignature")]
    assert len(signatures) == 1
    assert signatures[0].printed_meters == ((3, 4),)


def test_a_change_the_decoder_read_no_token_for_gets_one_after_its_barline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = attach(monkeypatch, [PrintedMeter(130, 3, 4, 0.8), PrintedMeter(430, 4, 4, 0.8)])

    rhythms = [s.rhythm for s in out]
    inserted = rhythms.index("barline") + 1
    assert rhythms[inserted] == "timeSignature/4"
    assert out[inserted].printed_meters == ((4, 4),)


def test_a_courtesy_signature_after_the_last_bar_is_left_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = attach(monkeypatch, [PrintedMeter(970, 4, 4, 0.8)])

    assert [s.rhythm for s in out] == [s.rhythm for s in stream()]
    assert all(not s.printed_meters for s in out)


def placed(x: float) -> EncodedSymbol:
    symbol = EncodedSymbol("note_4", "C4", position="upper")
    symbol.image_coordinates = (x, 0.0)
    return symbol


def misplaced_barline_stream() -> list[EncodedSymbol]:
    """Three quarters, then a barline the decoder put 250px right of where it is
    printed (at 400), then four quarters."""
    return [
        EncodedSymbol("clef_G2", position="upper"),
        EncodedSymbol("keySignature_0"),
        EncodedSymbol("timeSignature/4"),
        placed(150),
        placed(250),
        placed(350),
        barline(650),
        placed(480),
        placed(560),
        placed(640),
        placed(720),
        barline(800),
    ]


def test_a_signature_after_a_misplaced_barline_opens_the_bar_after_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """eerovil/musescore-choir-plugins#274, Kantajani s12: the bass's first barline
    was decoded far right of the print, and the 6/4 after it was read as the
    system's opening signature."""
    monkeypatch.setattr(
        time_signature_reader,
        "find_time_signatures",
        lambda image, s: [PrintedMeter(420, 4, 4, 0.8)],
    )
    out = attach_printed_meters(
        misplaced_barline_stream(), np.zeros((1, 1)), staff(), lambda point: point
    )

    assert not out[2].printed_meters
    rhythms = [s.rhythm for s in out]
    inserted = rhythms.index("barline") + 1
    assert rhythms[inserted] == "timeSignature/4"
    assert out[inserted].printed_meters == ((4, 4),)


def test_a_signature_at_the_head_of_placed_notes_stays_the_opening_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        time_signature_reader,
        "find_time_signatures",
        lambda image, s: [PrintedMeter(110, 3, 4, 0.8)],
    )
    out = attach_printed_meters(
        misplaced_barline_stream(), np.zeros((1, 1)), staff(), lambda point: point
    )

    assert out[2].printed_meters == ((3, 4),)
    assert [s.rhythm for s in out] == [s.rhythm for s in misplaced_barline_stream()]


#: A printed 6, as the reader cut it out of Lempilintu's 6/4 (rows 0-1 and 28 are
#: staff lines). eerovil/musescore-choir-plugins#274: it correlates with the 6 at
#: 0.76 and with the 0 at 0.72, too close to read, so the bar kept the decoder's 4/4.
_PRINTED_SIX = [
    "####################",
    "####################",
    ".....###########....",
    "....#####...######..",
    "...#####....#######.",
    "..#####.....#######.",
    ".######....########.",
    ".######.....#######.",
    ".#####......#######.",
    "######.......#####..",
    "######..............",
    "######..............",
    "######..#########...",
    "##################..",
    "####################",
    "########..#.########",
    "#######......#######",
    "#######......#######",
    "#######.......######",
    "#######.......######",
    ".######.......######",
    ".######......#######",
    "..#####......######.",
    "..#####......######.",
    "...#####....######..",
    "....#############...",
    "......#########.....",
    "....................",
    "####################",
    "......##########....",
]


def test_a_leading_digit_is_never_read_as_zero() -> None:
    ink = np.array([[ch == "#" for ch in row] for row in _PRINTED_SIX])
    on_line = np.zeros(len(_PRINTED_SIX), dtype=bool)
    on_line[[0, 1, 28]] = True

    digit, score, runner_up = time_signature_reader.classify_digit(ink, on_line)
    assert digit == 6 and score - runner_up < 0.1  # 0 is almost as good

    digit, score, runner_up = time_signature_reader.classify_digit(
        ink, on_line, frozenset(range(1, 10))
    )
    assert digit == 6
    assert score - runner_up >= 0.1


#: The numerator a barline gave when read as 1/1 on Vieläkö huvittaisi
#: (eerovil/musescore-choir-plugins#274): the reader's own `raw` and `clean`
#: columns and the staff-line rows. A 3-4px stroke, no flag.
_BARLINE_RAW = [
    "...........##...........",
    "########################",
    "########################",
    "########################",
    "....................###.",
    "...................#####",
    "...................#####",
    "....................###.",
    "....................###.",
    "....................####",
    "....................####",
    "##########........######",
    "########################",
    "########################",
    "###########.###.########",
    "....................####",
    "....................####",
    "....................###.",
    "....................###.",
    "...................####.",
    "....................###.",
    "....................###.",
    "########################",
    "########################",
    "########################",
    "########################",
]
_BARLINE_CLEAN = [
    "...........##...........",
    "########################",
    "########################",
    "########################",
    "....................###.",
    "....................###.",
    "....................###.",
    "....................###.",
    "....................###.",
    "....................####",
    "....................####",
    "....................####",
    "....................####",
    "....................####",
    "....................####",
    "....................####",
    "....................####",
    "....................###.",
    "....................###.",
    "...................####.",
    "....................###.",
    "....................###.",
    "....................###.",
    "....................###.",
    "....................###.",
    "....................###.",
]
_BARLINE_ON_LINE = "11110000000111100000001111"
_BARLINE_UNIT = 10.848


def _bits(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows])


def test_a_barline_read_as_a_one_is_no_number() -> None:
    on_line = np.array([ch == "1" for ch in _BARLINE_ON_LINE])
    raw, clean = _bits(_BARLINE_RAW), _bits(_BARLINE_CLEAN)

    assert time_signature_reader._read_number(raw, clean, on_line, _BARLINE_UNIT) is None


@pytest.mark.parametrize("kind", ["doublebarline", "bolddoublebarline", "repeatStart"])
def test_a_change_after_a_double_barline_or_repeat_opens_the_bar_after_it(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """eerovil/musescore-choir-plugins#274: Kantajani prints 3/4 after a start-repeat
    and 6/4 after a double barline; both landed in the bar before."""
    symbols = stream()
    symbols[6] = EncodedSymbol(kind)
    symbols[6].image_coordinates = (400.0, 0.0)
    monkeypatch.setattr(
        time_signature_reader,
        "find_time_signatures",
        lambda image, s: [PrintedMeter(130, 3, 4, 0.8), PrintedMeter(430, 4, 4, 0.8)],
    )

    out = attach_printed_meters(symbols, np.zeros((1, 1)), staff(), lambda point: point)

    rhythms = [s.rhythm for s in out]
    inserted = rhythms.index(kind) + 1
    assert rhythms[inserted] == "timeSignature/4"
    assert out[inserted].printed_meters == ((4, 4),)
