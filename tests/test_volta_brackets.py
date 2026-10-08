"""Volta brackets over an end repeat, found in the pixels when the decoder skipped them.

eerovil/musescore-choir-plugins#319: a "1." / "2." bracket stands about seven staff
spaces above the top staff, past the four the staff image keeps, so the decoder
never writes a volta token. These draw a staff with barlines and brackets above it
and check that the brackets are found by their hooks, that a lyric extender, a beam
and the staff above are not taken for one, and where the tokens land.
"""

import numpy as np

from homr.model import Staff, StaffPoint
from homr.transformer.vocabulary import EncodedSymbol
from homr.volta_brackets import add_missing_voltas, brackets_above

UNIT = 20
TOP = 300
LINES = [TOP + UNIT * n for n in range(5)]
LEFT, RIGHT = 40, 1240
#: Four bars of 300 px: interior barlines at 340, 640, 940.
BARS = [340, 640, 940]
HIGH = TOP - 7 * UNIT


def staff() -> Staff:
    points = [StaffPoint(float(x), [float(y) for y in LINES], 0.0) for x in range(LEFT, RIGHT, 5)]
    found = Staff(points)
    found.bar_line_xs = [float(x) for x in [LEFT, *BARS, RIGHT]]
    return found


def bracket(image: np.ndarray, x0: int, x1: int, y: int = HIGH, right_hook: bool = False) -> None:
    image[y : y + 2, x0:x1] = 0
    image[y : y + 30, x0 : x0 + 2] = 0
    if right_hook:
        image[y : y + 30, x1 - 2 : x1] = 0


def page() -> np.ndarray:
    image = np.full((420, 1300), 255, dtype=np.uint8)
    for y in LINES:
        image[y - 1 : y + 1, LEFT:RIGHT] = 0
    for x in [LEFT, *BARS, RIGHT]:
        image[LINES[0] : LINES[4], x : x + 2] = 0
    return image


def tokens(*rhythms: str) -> list[EncodedSymbol]:
    return [EncodedSymbol(rhythm) for rhythm in rhythms]


def four_bars(*ends: str) -> list[EncodedSymbol]:
    """``clef note | note | note | note``, the given bar ends in between, each at its barline."""
    out = tokens("clef_G2", "keySignature_0", "note_4")
    for end, x in zip(ends, BARS, strict=False):
        out += tokens(end, "note_4")
        out[-2].coordinates = (float(x), 350.0)
    return out


def same(point: tuple[float, float]) -> tuple[float, float]:
    return point


def rhythms(symbols: list[EncodedSymbol]) -> list[str]:
    return [s.rhythm for s in symbols]


def test_a_hooked_line_high_above_the_staff_is_a_bracket() -> None:
    image = page()
    bracket(image, 350, 630, right_hook=True)
    bracket(image, 645, 900)
    found = brackets_above(image, staff(), 0)
    assert [(round(b.left), round(b.right), b.right_hook) for b in found] == [
        (350, 629, True),
        (645, 899, False),
    ]


def test_a_line_with_no_hook_is_a_lyric_extender_not_a_bracket() -> None:
    image = page()
    image[HIGH : HIGH + 2, 350:630] = 0
    assert brackets_above(image, staff(), 0) == []


def test_a_beam_with_its_stems_is_too_thick_to_be_a_bracket() -> None:
    image = page()
    image[HIGH : HIGH + 10, 350:630] = 0
    image[HIGH : HIGH + 60, 350:353] = 0
    assert brackets_above(image, staff(), 0) == []


def test_the_staff_above_is_not_looked_into() -> None:
    image = page()
    bracket(image, 350, 630, y=HIGH)
    # The staff above ends just under the bracket: the bracket is its business.
    assert brackets_above(image, staff(), HIGH + 10) == []


def test_brackets_over_an_end_repeat_become_volta_tokens() -> None:
    image = page()
    bracket(image, 345, 638, right_hook=True)  # 1. over bar 2
    bracket(image, 645, 900)  # 2. over bar 3, open
    symbols = four_bars("barline", "repeatEnd", "barline")
    assert add_missing_voltas(symbols, staff(), image, 0, same) == 2  # noqa: PLR2004
    assert rhythms(symbols) == [
        "clef_G2",
        "keySignature_0",
        "note_4",
        "barline",
        "voltaStart",
        "note_4",
        "voltaStop",
        "repeatEnd",
        "voltaStart",
        "note_4",
        "voltaDiscontinue",
        "barline",
        "note_4",
    ]


def test_a_1_bracket_from_the_head_of_the_system_and_a_closed_2() -> None:
    image = page()
    bracket(image, 120, 638, right_hook=True)  # 1. over bars 1-2
    bracket(image, 645, 935, right_hook=True)  # 2. over bar 3, closed
    symbols = four_bars("barline", "repeatEnd", "barline")
    assert add_missing_voltas(symbols, staff(), image, 0, same) == 2  # noqa: PLR2004
    assert rhythms(symbols) == [
        "clef_G2",
        "keySignature_0",
        "voltaStart",
        "note_4",
        "barline",
        "note_4",
        "voltaStop",
        "repeatEnd",
        "voltaStart",
        "note_4",
        "voltaStop",
        "barline",
        "note_4",
    ]


def test_a_bracket_that_does_not_end_at_a_repeat_is_left_alone() -> None:
    image = page()
    bracket(image, 345, 638, right_hook=True)
    symbols = four_bars("barline", "barline", "barline")
    assert add_missing_voltas(symbols, staff(), image, 0, same) == 0
    symbols = four_bars("repeatEnd", "barline", "barline")  # the repeat is elsewhere
    assert add_missing_voltas(symbols, staff(), image, 0, same) == 0
    assert "voltaStart" not in rhythms(symbols)


def test_a_bracket_longer_than_the_bars_before_the_repeat_is_left_alone() -> None:
    image = page()
    bracket(image, 60, 638, right_hook=True)  # three bars on the page
    symbols = tokens("clef_G2", "note_4", "repeatEnd", "note_4")  # one bar read before it
    symbols[2].coordinates = (640.0, 350.0)
    assert add_missing_voltas(symbols, staff(), image, 0, same) == 0


def test_a_barline_the_page_detection_missed_outside_the_bracket_does_not_matter() -> None:
    image = page()
    bracket(image, 345, 638, right_hook=True)
    bracket(image, 645, 900)
    found = staff()
    found.bar_line_xs = [float(LEFT), 120.0, 340.0, float(RIGHT)]  # 640 and 940 missed, 120 a |:
    symbols = four_bars("barline", "repeatEnd", "barline")
    assert add_missing_voltas(symbols, found, image, 0, same) == 2  # noqa: PLR2004
    assert rhythms(symbols)[4:11] == [
        "voltaStart",
        "note_4",
        "voltaStop",
        "repeatEnd",
        "voltaStart",
        "note_4",
        "voltaDiscontinue",
    ]


def test_a_volta_the_decoder_wrote_is_never_doubled() -> None:
    image = page()
    bracket(image, 345, 638, right_hook=True)
    symbols = four_bars("barline", "repeatEnd", "barline")
    symbols.insert(4, EncodedSymbol("voltaStart"))
    before = rhythms(symbols)
    assert add_missing_voltas(symbols, staff(), image, 0, same) == 0
    assert rhythms(symbols) == before


def test_the_generator_numbers_and_places_the_endings() -> None:
    from homr.music_xml_generator import XmlGeneratorArguments, generate_xml

    image = page()
    bracket(image, 345, 638, right_hook=True)
    bracket(image, 645, 900)
    symbols = four_bars("barline", "repeatEnd", "barline")
    add_missing_voltas(symbols, staff(), image, 0, same)
    xml = generate_xml(XmlGeneratorArguments(), [symbols], "")
    endings = [
        (measure.get("number"), barline.get("location"), ending.get("number"), ending.get("type"))
        for measure in xml.iter("measure")
        for barline in measure.findall("barline")
        for ending in barline.findall("ending")
    ]
    assert endings == [
        ("2", "left", "1", "start"),
        ("2", "right", "1", "stop"),
        ("3", "left", "2", "start"),
        ("3", "right", "2", "discontinue"),
    ]


def test_a_start_repeat_after_the_2_bracket_closes_its_bar() -> None:
    image = page()
    bracket(image, 345, 638, right_hook=True)
    bracket(image, 645, 900)
    symbols = four_bars("barline", "repeatEnd", "repeatStart")
    assert add_missing_voltas(symbols, staff(), image, 0, same) == 2  # noqa: PLR2004
    assert rhythms(symbols)[8:13] == [
        "voltaStart",
        "note_4",
        "voltaDiscontinue",
        "repeatStart",
        "note_4",
    ]
