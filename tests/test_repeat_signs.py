"""A start repeat opening a staff, found in the pixels when the decoder skipped it.

eerovil/musescore-choir-plugins#312: the decoder reads ``clef, key, note`` where the
page prints ``clef, key, |:, note``. These draw that staff head and check that the
sign is found by its dots, that a plain barline or a note is not taken for it, and
that a token is only ever added where the decoder wrote nothing.
"""

import numpy as np

from homr.model import Staff, StaffPoint
from homr.repeat_signs import add_missing_start_repeat, start_repeat_at
from homr.transformer.vocabulary import EncodedSymbol

UNIT = 20
TOP = 100
LINES = [TOP + UNIT * n for n in range(5)]
WIDTH = 600


def staff() -> Staff:
    return Staff([StaffPoint(float(x), [float(y) for y in LINES], 0.0) for x in range(0, WIDTH, 5)])


def page(bar: bool = True, dots: bool = True, note_at: int | None = None) -> np.ndarray:
    """A staff with a clef-ish blob at 40, a key at 100 and the sign at ~160."""
    image = np.full((300, WIDTH), 255, dtype=np.uint8)
    for y in LINES:
        image[y - 1 : y + 1, :] = 0
    image[LINES[0] - 10 : LINES[4] + 10, 30:50] = 0  # the clef
    image[LINES[1] : LINES[2], 95:105] = 0  # a sharp, roughly
    if bar:
        image[LINES[0] : LINES[4] + 1, 150:157] = 0  # thick
        image[LINES[0] : LINES[4] + 1, 161:163] = 0  # thin
    if dots:
        for space in (1, 2):
            middle = (LINES[space] + LINES[space + 1]) // 2
            image[middle - 4 : middle + 4, 168:176] = 0
    if note_at is not None:
        middle = (LINES[1] + LINES[2]) // 2
        image[middle - 8 : middle + 8, note_at : note_at + 24] = 0
    return image


def tokens(*rhythms: str, at: tuple[float, ...] = ()) -> list[EncodedSymbol]:
    out = []
    for n, rhythm in enumerate(rhythms):
        symbol = EncodedSymbol(rhythm)
        if n < len(at):
            symbol.coordinates = (at[n], 150.0)
        out.append(symbol)
    return out


def same(point: tuple[float, float]) -> tuple[float, float]:
    return point


def test_the_two_dots_after_a_barline_are_a_start_repeat() -> None:
    assert start_repeat_at(page(), staff(), 100, 220)


def test_a_plain_barline_is_not() -> None:
    assert not start_repeat_at(page(dots=False), staff(), 100, 220)


def test_dots_with_no_barline_are_not() -> None:
    assert not start_repeat_at(page(bar=False), staff(), 100, 220)


def test_a_notehead_after_a_barline_is_not_a_dot() -> None:
    assert not start_repeat_at(page(dots=False, note_at=168), staff(), 100, 220)


def test_the_missing_token_goes_after_the_signatures() -> None:
    symbols = tokens("clef_G2", "keySignature_1", "note_4", at=(40, 100, 240))
    assert add_missing_start_repeat(symbols, staff(), page(), same)
    assert [s.rhythm for s in symbols] == ["clef_G2", "keySignature_1", "repeatStart", "note_4"]


def test_nothing_is_added_where_the_decoder_wrote_the_sign() -> None:
    symbols = tokens("clef_G2", "keySignature_1", "repeatStart", "note_4", at=(40, 100, 160, 240))
    assert not add_missing_start_repeat(symbols, staff(), page(), same)
    assert [s.rhythm for s in symbols].count("repeatStart") == 1


def test_nothing_is_added_without_the_sign_on_the_page() -> None:
    symbols = tokens("clef_G2", "keySignature_1", "note_4", at=(40, 100, 240))
    assert not add_missing_start_repeat(symbols, staff(), page(dots=False), same)
    assert [s.rhythm for s in symbols] == ["clef_G2", "keySignature_1", "note_4"]


def test_a_staff_not_opening_with_a_clef_is_left_alone() -> None:
    symbols = tokens("note_4", "note_4", at=(240, 300))
    assert not add_missing_start_repeat(symbols, staff(), page(), same)


def test_the_sign_must_stand_before_the_first_note() -> None:
    # The first note is read left of the sign: the head ends there.
    symbols = tokens("clef_G2", "keySignature_1", "note_4", at=(40, 100, 140))
    assert not add_missing_start_repeat(symbols, staff(), page(), same)


def test_a_time_signature_is_not_taken_for_one() -> None:
    """With the staff lines gone, the edge of a ``2/2`` is one small blob per space."""
    image = page(dots=False)
    for space in range(4):
        middle = (LINES[space] + LINES[space + 1]) // 2
        image[middle - 4 : middle + 4, 168:176] = 0
    assert not start_repeat_at(image, staff(), 100, 220)


def test_a_thin_line_alone_is_not_the_sign() -> None:
    image = page()
    image[LINES[0] : LINES[4] + 1, 150:157] = 255  # take the thick stroke away
    for y in LINES:
        image[y - 1 : y + 1, 150:157] = 0
    assert not start_repeat_at(image, staff(), 100, 220)


def test_a_staff_line_a_few_pixels_off_its_estimate_does_not_hide_the_dots() -> None:
    """After dewarping a line can sit off where the staff says it is; what is left of
    it must not read as ink beside the dots."""
    image = page()
    for y in LINES:
        image[y - 1 : y + 1, :] = 255
        image[y + 3 : y + 5, :] = 0
    for space in (1, 2):
        middle = (LINES[space] + LINES[space + 1]) // 2
        image[middle - 4 : middle + 4, 168:176] = 0
    assert start_repeat_at(image, staff(), 100, 220)
