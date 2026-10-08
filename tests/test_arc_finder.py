"""The picture-based arc finder (eerovil/musescore-choir-plugins#328).

Curves are drawn onto a blank staff the way an engraver draws them, so each rule
of `find_curves` is tested on the shape it is about; the comparison with what
homr wrote is tested on small MusicXML documents."""

import xml.etree.ElementTree as ET
from fractions import Fraction

import cv2
import numpy as np

from homr import arc_finder as A
from homr.doubt import TIE_PICTURE, mark_words, picture_arc_doubts

UNIT = 10.0
TOP = 100
WIDTH = 600


def _staff() -> A.StaffGeometry:
    lines = [TOP + i * UNIT for i in range(5)]
    return A.StaffGeometry([0.0, float(WIDTH)], [lines, lines], UNIT, 20.0, WIDTH - 20.0)


def _page() -> tuple[np.ndarray, np.ndarray]:
    gray = np.full((260, WIDTH), 255, np.uint8)
    for i in range(5):
        y = TOP + i * int(UNIT)
        cv2.line(gray, (20, y), (WIDTH - 20, y), 0, 1)
    return gray, np.zeros_like(gray)


def _head(gray: np.ndarray, heads: np.ndarray, x: int, y: int) -> None:
    cv2.ellipse(gray, (x, y), (6, 4), -20, 0, 360, 0, -1)
    cv2.ellipse(heads, (x, y), (6, 4), -20, 0, 360, 1, -1)


def _arc(gray: np.ndarray, x0: int, x1: int, y: int, height: int, over: bool) -> None:
    centre = ((x0 + x1) // 2, y)
    cv2.ellipse(
        gray, centre, ((x1 - x0) // 2, height), 0, 180 if over else 0, 360 if over else 180, 0, 2
    )


def _find(gray: np.ndarray, heads: np.ndarray) -> list[A.Curve]:
    empty = np.zeros_like(heads)
    return A.find_curves(gray, heads, empty, empty, empty, [_staff()])


def test_a_slur_above_two_notes_is_one_curve_bending_up() -> None:
    gray, heads = _page()
    _head(gray, heads, 150, TOP + 15)
    _head(gray, heads, 300, TOP + 25)
    _arc(gray, 155, 300, TOP - 8, 14, over=True)
    curves = _find(gray, heads)
    assert len(curves) == 1
    assert curves[0].over
    assert curves[0].left[0] < 165 and curves[0].right[0] > 290


def test_a_tie_below_is_a_curve_bending_down() -> None:
    gray, heads = _page()
    _head(gray, heads, 150, TOP + 40)
    _head(gray, heads, 230, TOP + 40)
    _arc(gray, 156, 226, TOP + 48, 7, over=False)
    curves = _find(gray, heads)
    assert len(curves) == 1
    assert not curves[0].over


def test_a_dashed_phrase_arc_is_chained_back_into_one_curve() -> None:
    gray, heads = _page()
    solid = np.full_like(gray, 255)
    _arc(solid, 60, 520, TOP - 10, 25, over=True)
    mask = np.zeros_like(gray, dtype=bool)
    for x in range(60, 520, 8):
        mask[:, x : x + 4] = True
    gray[(solid == 0) & mask] = 0
    curves = _find(gray, heads)
    dashed = [c for c in curves if c.dashed]
    assert len(dashed) == 1
    assert dashed[0].width > 400


def test_a_broken_staff_line_is_not_a_dashed_arc() -> None:
    gray, heads = _page()
    gray[TOP + 20, ::7] = 255  # a scan's dropout
    gray[TOP + 20, 3::7] = 255
    assert _find(gray, heads) == []
    # a scanned line also waves a little; a chain lying along it is still the line
    xs = np.arange(30.0, 560.0)
    wavy = A.Curve(xs, TOP + 20 + 2 * np.sin(xs / 80), over=True, dashed=True)
    assert A._along_lines(wavy, [_staff()])
    arc = A.Curve(xs, TOP - 5 - 20 * np.sin(np.linspace(0, np.pi, len(xs))), over=True)
    assert not A._along_lines(arc, [_staff()])


def test_ties_touching_at_their_shared_notes_are_cut_apart() -> None:
    gray, heads = _page()
    for x0 in (100, 200, 300):
        _arc(gray, x0, x0 + 100, TOP + 48, 8, over=False)
    curves = _find(gray, heads)
    assert len(curves) == 3  # noqa: PLR2004


def test_a_fermata_is_not_an_arc() -> None:
    gray, heads = _page()
    _arc(gray, 140, 160, TOP - 8, 9, over=True)
    gray[TOP - 10 : TOP - 7, 149:152] = 0
    assert _find(gray, heads) == []


# --- what homr wrote against what the page shows ---------------------------------


def _score(notes: str) -> ET.Element:
    return ET.fromstring(
        "<score-partwise><part id='P1'><measure number='1'><attributes><divisions>1"
        "</divisions></attributes>" + notes + "</measure></part></score-partwise>"
    )


def _note(step: str, *marks: str, voice: str = "1", chord: bool = False) -> str:
    return (
        f"<note>{'<chord/>' if chord else ''}<pitch><step>{step}</step><octave>4</octave>"
        f"</pitch><duration>1</duration><voice>{voice}</voice><staff>1</staff>"
        f"<notations>{''.join(marks)}</notations></note>"
    )


def _arcs(xml: ET.Element) -> list[tuple[str, int | None, int | None]]:
    notes = A.read_notes(xml)
    return sorted((w.kind, w.start, w.stop) for w in A.read_written(notes))


def _apply(xml: ET.Element, picture: list[A.PictureArc], curves: list[A.Curve]) -> A.Outcome:
    notes = A.read_notes(xml)
    where = {i: (100.0 + 100 * i, 120.0) for i in range(len(notes))}
    return A.apply(notes, picture, curves, where, [_staff()], cautious=False, marks_only=False)


def _curve(x0: float, x1: float) -> A.Curve:
    xs = np.linspace(x0, x1, 20)
    return A.Curve(xs, 100 - 10 * np.sin(np.linspace(0, np.pi, 20)), over=True)


def test_a_tie_on_the_page_that_homr_did_not_write_is_added() -> None:
    xml = _score(_note("C") + _note("C"))
    outcome = _apply(xml, [A.PictureArc("tie", 0, 1, 0, over=False)], [_curve(110, 190)])
    assert outcome.added == 1
    assert _arcs(xml) == [("tie", 0, 1)]
    assert [t.get("type") for t in xml.iter("tie")] == ["start", "stop"]


def test_homr_s_arc_on_the_same_notes_is_kept_as_it_is() -> None:
    xml = _score(
        _note("C", "<slur type='start' number='1'/>") + _note("D", "<slur type='stop' number='1'/>")
    )
    outcome = _apply(xml, [A.PictureArc("slur", 0, 1, 0, over=True)], [_curve(110, 190)])
    assert (outcome.kept, outcome.added) == (1, 0)
    assert _arcs(xml) == [("slur", 0, 1)]


def test_a_slur_homr_hung_on_the_other_voice_moves_to_the_page_s_voice() -> None:
    xml = _score(
        _note("E", "<slur type='start' number='1'/>")
        + _note("C", voice="2", chord=False)
        + _note("F", "<slur type='stop' number='1'/>")
        + _note("D", voice="2")
    )
    notes = A.read_notes(xml)
    for i, onset in enumerate([0, 0, 1, 1]):
        notes[i].onset = Fraction(onset)
    where = {i: (100.0 + 100 * (i // 2), 120.0) for i in range(len(notes))}
    outcome = A.apply(
        notes,
        [A.PictureArc("slur", 1, 3, 0, over=False)],
        [_curve(110, 190)],
        where,
        [_staff()],
        cautious=False,
        marks_only=False,
    )
    assert (outcome.moved, outcome.added) == (1, 1)
    assert _arcs(xml) == [("slur", 1, 3)]


def test_an_arc_with_no_curve_anywhere_near_it_is_taken_out() -> None:
    xml = _score(
        _note("C", "<slur type='start' number='1'/>") + _note("D", "<slur type='stop' number='1'/>")
    )
    outcome = _apply(xml, [], [])
    assert outcome.removed == 1
    assert _arcs(xml) == []
    assert xml.find(".//notations") is None


def test_an_arc_the_picture_could_not_place_is_never_taken_out() -> None:
    xml = _score(
        _note("C", "<slur type='start' number='1'/>") + _note("D", "<slur type='stop' number='1'/>")
    )
    outcome = _apply(xml, [], [_curve(90, 210)])
    assert outcome.kept == 1
    assert _arcs(xml) == [("slur", 0, 1)]


def test_nested_slurs_get_numbers_of_their_own() -> None:
    xml = _score(_note("C") + _note("D") + _note("E") + _note("F"))
    _apply(
        xml,
        [A.PictureArc("slur", 0, 3, 0, over=True), A.PictureArc("slur", 1, 2, 0, over=True)],
        [_curve(90, 410), _curve(190, 310)],
    )
    numbers = {s.get("number") for s in xml.iter("slur")}
    assert len(numbers) == 2  # noqa: PLR2004
    assert _arcs(xml) == [("slur", 0, 3), ("slur", 1, 2)]


def test_a_tie_the_picture_was_unsure_of_is_marked_tie() -> None:
    xml = _score(_note("C") + _note("C"))
    _apply(xml, [A.PictureArc("tie", 0, 1, 0, over=False, unsure=True)], [_curve(110, 190)])
    doubts = picture_arc_doubts(xml)
    assert doubts == {(0, 1, 1): {TIE_PICTURE}}
    assert mark_words(doubts[(0, 1, 1)]) == ["tie?"]
    A.forget(xml)
    assert not any(A.UNSURE in e.attrib for e in xml.iter())


def test_a_curve_nothing_could_be_hung_on_marks_the_note_beside_it() -> None:
    xml = _score(_note("C") + _note("D"))
    notes = A.read_notes(xml)
    A.apply(notes, [], [], {0: (100.0, 120.0), 1: (200.0, 120.0)}, [_staff()], unplaced=[1])
    assert mark_words(picture_arc_doubts(xml)[(0, 1, 1)]) == ["slur?"]


def test_a_slur_drawn_over_tied_notes_ends_on_the_last_of_them() -> None:
    arcs = [A.PictureArc("slur", 0, 2, 0, over=True)]
    A.extend_over_ties(arcs, [(2, 3), (3, 4)])
    assert arcs[0].stop == 4  # noqa: PLR2004


def test_by_default_no_arc_is_changed_and_the_page_s_arc_is_marked() -> None:
    xml = _score(
        _note("C", "<slur type='start' number='1'/>")
        + _note("D", "<slur type='stop' number='1'/>")
        + _note("E")
        + _note("E")
    )
    before = ET.tostring(xml)
    notes = A.read_notes(xml)
    where = {i: (100.0 + 100 * i, 120.0) for i in range(len(notes))}
    outcome = A.apply(notes, [A.PictureArc("tie", 2, 3, 0, over=False)], [], where, [_staff()])
    assert outcome.added == 0 and outcome.removed == 0 and outcome.moved == 0
    assert mark_words(picture_arc_doubts(xml)[(0, 1, 1)]) == ["slur?"]
    A.forget(xml)
    assert ET.tostring(xml) == before


def test_by_default_homr_s_arc_on_another_voice_stays_and_is_marked() -> None:
    xml = _score(
        _note("E", "<slur type='start' number='1'/>")
        + _note("C", voice="2")
        + _note("F", "<slur type='stop' number='1'/>")
        + _note("D", voice="2")
    )
    notes = A.read_notes(xml)
    for i, onset in enumerate([0, 0, 1, 1]):
        notes[i].onset = Fraction(onset)
    where = {i: (100.0 + 100 * (i // 2), 120.0) for i in range(len(notes))}
    outcome = A.apply(notes, [A.PictureArc("slur", 1, 3, 0, over=False)], [], where, [_staff()])
    assert outcome.moved == 0
    assert _arcs(xml) == [("slur", 0, 2)]
    assert picture_arc_doubts(xml)
