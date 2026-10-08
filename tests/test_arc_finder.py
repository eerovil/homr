"""The picture-based arc finder (eerovil/musescore-choir-plugins#328).

Curves are drawn onto a blank staff the way an engraver draws them, so each rule
of `find_curves` is tested on the shape it is about; the comparison with what
homr wrote is tested on small MusicXML documents."""

import xml.etree.ElementTree as ET

import cv2
import numpy as np

from homr import arc_finder as A
from homr.doubt import mark_words, picture_arc_doubts

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


def _note(step: str, *marks: str, voice: str = "1", backup: bool = False) -> str:
    back = "<backup><duration>2</duration></backup>" if backup else ""
    return (
        f"{back}<note><pitch><step>{step}</step><octave>4</octave>"
        f"</pitch><duration>1</duration><voice>{voice}</voice><staff>1</staff>"
        f"<notations>{''.join(marks)}</notations></note>"
    )


SLUR_START = "<slur type='start' number='1'/>"
SLUR_STOP = "<slur type='stop' number='1'/>"


def _mark(
    xml: ET.Element, picture: list[A.PictureArc], unplaced: list[int] | None = None
) -> A.Outcome:
    return A.mark(A.read_notes(xml), picture, unplaced or [])


def _words(xml: ET.Element) -> list[str]:
    doubts = picture_arc_doubts(xml)
    return sorted(word for reasons in doubts.values() for word in mark_words(reasons))


def test_a_tie_homr_missed_is_marked_tie_and_no_arc_is_written() -> None:
    xml = _score(_note("C") + _note("C"))
    before = ET.tostring(xml)
    outcome = _mark(xml, [A.PictureArc("tie", 0, 1, 0, over=False)])
    assert (outcome.agreed, outcome.marked) == (0, 1)
    assert _words(xml) == ["tie?"]
    assert xml.find(".//tied") is None
    A.forget(xml)
    assert ET.tostring(xml) == before


def test_an_arc_homr_wrote_on_the_same_notes_is_not_marked() -> None:
    xml = _score(_note("C", SLUR_START) + _note("D", SLUR_STOP))
    outcome = _mark(xml, [A.PictureArc("slur", 0, 1, 0, over=True)])
    assert (outcome.agreed, outcome.marked) == (1, 0)
    assert _words(xml) == []


def test_a_slur_homr_hung_on_the_other_voice_is_marked_and_left_as_it_is() -> None:
    xml = _score(
        _note("E", SLUR_START)
        + _note("F", SLUR_STOP)
        + _note("C", voice="2", backup=True)
        + _note("D", voice="2")
    )
    before = ET.tostring(xml)
    outcome = _mark(xml, [A.PictureArc("slur", 2, 3, 0, over=False)])
    assert outcome.marked == 1
    assert _words(xml) == ["slur?"]
    A.forget(xml)
    assert ET.tostring(xml) == before


def test_homr_s_tie_to_its_own_chord_note_beats_the_picture_s_choice_of_note() -> None:
    xml = _score(
        _note("C", "<tied type='start'/>")
        + "<note><chord/><pitch><step>A</step><octave>3</octave></pitch><duration>1</duration>"
        "<voice>1</voice><staff>1</staff></note>" + _note("C", "<tied type='stop'/>")
    )
    outcome = _mark(xml, [A.PictureArc("slur", 1, 2, 0, over=True)])
    assert (outcome.agreed, outcome.marked) == (1, 0)


def test_a_curve_nothing_could_be_hung_on_marks_the_note_beside_it_slur() -> None:
    xml = _score(_note("C") + _note("D"))
    _mark(xml, [], unplaced=[1])
    assert _words(xml) == ["slur?"]


def test_a_note_flagged_for_both_kinds_says_both() -> None:
    xml = _score(_note("C") + _note("C") + _note("D"))
    _mark(xml, [A.PictureArc("tie", 0, 1, 0, over=False), A.PictureArc("slur", 0, 2, 0, over=True)])
    assert _words(xml) == ["slur?", "tie?"]


def test_two_voices_slurs_sharing_a_number_are_left_exactly_as_written() -> None:
    xml = _score(
        _note("E", SLUR_START)
        + _note("F", SLUR_STOP)
        + _note("C", SLUR_START, voice="2", backup=True)
        + _note("D", SLUR_STOP, voice="2")
    )
    before = ET.tostring(xml)
    _mark(
        xml, [A.PictureArc("slur", 0, 1, 0, over=True), A.PictureArc("slur", 2, 3, 0, over=False)]
    )
    A.forget(xml)
    assert ET.tostring(xml) == before


def test_a_failure_while_marking_never_costs_the_page(monkeypatch: object) -> None:
    from homr import main

    def broken(*_: object) -> None:
        raise ValueError("boom")

    monkeypatch.setattr(main, "mark_arcs", broken)  # type: ignore[attr-defined]
    main._mark_arcs(_score(_note("C")), None, [], lambda p: p)  # type: ignore[arg-type]


def test_a_slur_drawn_over_tied_notes_ends_on_the_last_of_them() -> None:
    arcs = [A.PictureArc("slur", 0, 2, 0, over=True)]
    A.extend_over_ties(arcs, [(2, 3), (3, 4)])
    assert arcs[0].stop == 4  # noqa: PLR2004
