"""Regression tests for the slur-stream repair migrated from the choir app.

These cases are the measured contract from musescore-choir-plugins #144/#211,
not a new engraving heuristic. The transformer writes one slur token per note;
these helpers make those streams directly so the pairing failure is explicit.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from homr import music_xml_generator
from homr.slur_resolution import resolve_slurs


def slurred_part(stream: str, bars: int = 8, per_bar: int = 4, number: str = "1") -> ET.Element:
    """Return a part whose stream is spelled ``1( 2) ...`` by measure."""
    wanted: dict[int, list[str]] = {}
    for token in stream.split():
        wanted.setdefault(int(token[:-1]), []).append("start" if token[-1] == "(" else "stop")

    part = ET.Element("part", id="P1")
    for bar in range(1, bars + 1):
        measure = ET.SubElement(part, "measure", number=str(bar))
        for note_no in range(per_bar):
            note = ET.SubElement(measure, "note")
            pitch = ET.SubElement(note, "pitch")
            ET.SubElement(pitch, "step").text = "C"
            ET.SubElement(pitch, "octave").text = "4"
            ET.SubElement(note, "duration").text = "1"
            here = wanted.get(bar, [])
            if note_no < len(here):
                notations = ET.SubElement(note, "notations")
                ET.SubElement(notations, "slur", type=here[note_no], number=number)
    return part


def slur_pairs(part: ET.Element) -> list[tuple[int, int, str]]:
    """Return the remaining (start measure, stop measure, number) pairs."""
    pairs: list[tuple[int, int, str]] = []
    opened: dict[str, list[int]] = {}
    for bar, measure in enumerate(part.findall("measure"), 1):
        for slur in measure.findall("note/notations/slur"):
            number = slur.get("number", "1")
            if slur.get("type") == "start":
                opened.setdefault(number, []).append(bar)
            elif slur.get("type") == "stop":
                pairs.append((opened[number].pop(), bar, number))
    assert all(not starts for starts in opened.values()), "a start was left open"
    return pairs


def test_slur_inside_one_measure_is_kept() -> None:
    part = slurred_part("1( 1)")
    assert resolve_slurs(part) == 0
    assert slur_pairs(part) == [(1, 1, "1")]


def test_melisma_across_one_barline_is_kept() -> None:
    part = slurred_part("1( 2)")
    assert resolve_slurs(part) == 0
    assert slur_pairs(part) == [(1, 2, "1")]


def test_slur_across_two_barlines_is_dropped() -> None:
    part = slurred_part("1( 3)")
    assert resolve_slurs(part) == 1
    assert slur_pairs(part) == []


def test_runaway_does_not_take_surrounding_slurs_with_it() -> None:
    part = slurred_part("1( 1) 2( 6) 7( 7)")
    assert resolve_slurs(part) == 1
    assert slur_pairs(part) == [(1, 1, "1"), (7, 7, "1")]


def test_start_made_while_same_number_is_open_is_removed() -> None:
    part = slurred_part("1( 1( 2)")
    assert resolve_slurs(part) == 0
    assert slur_pairs(part) == [(1, 2, "1")]


def test_stop_that_closes_nothing_is_removed() -> None:
    part = slurred_part("2) 3( 3)")
    assert resolve_slurs(part) == 0
    assert slur_pairs(part) == [(3, 3, "1")]


def test_start_that_never_stops_is_removed() -> None:
    part = slurred_part("1( 1) 3(")
    assert resolve_slurs(part) == 0
    assert slur_pairs(part) == [(1, 1, "1")]


def loose_ends(part: ET.Element) -> list[tuple[int, str]]:
    """The slur ends left without a partner, as (measure, start|stop)."""
    ends: list[tuple[int, str]] = []
    opened: dict[str, list[int]] = {}
    for bar, measure in enumerate(part.findall("measure"), 1):
        for slur in measure.findall("note/notations/slur"):
            number = slur.get("number", "1")
            if slur.get("type") == "start":
                opened.setdefault(number, []).append(bar)
            elif opened.get(number):
                opened[number].pop()
            else:
                ends.append((bar, "stop"))
    ends.extend((bar, "start") for starts in opened.values() for bar in starts)
    return ends


def test_a_slur_from_the_system_before_keeps_its_stop() -> None:
    part = slurred_part("1) 2( 2)", bars=4)
    assert resolve_slurs(part) == 0
    assert loose_ends(part) == [(1, "stop")]


def test_a_slur_into_the_next_system_keeps_its_start() -> None:
    part = slurred_part("1( 1) 4(", bars=4)
    assert resolve_slurs(part) == 0
    assert loose_ends(part) == [(4, "start")]


def test_a_loose_end_away_from_the_edge_is_still_removed() -> None:
    part = slurred_part("2) 2(", bars=4)
    resolve_slurs(part)
    assert loose_ends(part) == []


def test_edges_are_only_kept_when_asked() -> None:
    part = slurred_part("1) 4(", bars=4)
    resolve_slurs(part, keep_edges=False)
    assert loose_ends(part) == []


def test_a_slur_into_the_next_system_may_start_a_bar_before_the_last() -> None:
    """Finlandia s01 Bass 2: a slur from bar 7 of 8 into the next system."""
    part = slurred_part("3(", bars=4)
    resolve_slurs(part)
    assert loose_ends(part) == [(3, "start")]
    part = slurred_part("3(", bars=4)
    resolve_slurs(part, edge_bars=1)
    assert loose_ends(part) == []


def test_a_slur_and_a_tie_both_into_the_next_system_keep_both_starts() -> None:
    """Vielako s01: a slur from the eighth and a tie from the dotted half both
    run off the edge; the second start shares the first's open number."""
    part = slurred_part("4( 4(", bars=4)
    resolve_slurs(part)
    assert loose_ends(part) == [(4, "start"), (4, "start")]


def test_a_redundant_start_a_stop_came_after_is_still_removed() -> None:
    part = slurred_part("3( 3( 4)", bars=4)
    resolve_slurs(part)
    assert slur_pairs(part) == [(3, 4, "1")]


def test_kept_edges_settle_in_one_pass() -> None:
    part = slurred_part("1) 1( 2) 4( 4( 4) 4(", bars=4)
    resolve_slurs(part)
    once = ET.tostring(part)
    resolve_slurs(part)
    assert ET.tostring(part) == once
    assert loose_ends(part) == [(1, "stop"), (4, "start")]


def test_b5_shape_settles_in_one_pass() -> None:
    part = slurred_part("1( 1( 2( 3( 3) 4( 4) 5( 6) 7( 7)")
    assert resolve_slurs(part) == 1
    assert slur_pairs(part) == [(4, 4, "1"), (5, 6, "1"), (7, 7, "1")]
    assert resolve_slurs(part) == 0
    assert slur_pairs(part) == [(4, 4, "1"), (5, 6, "1"), (7, 7, "1")]


def test_empty_notations_does_not_survive_removed_slur() -> None:
    part = slurred_part("1( 3)")
    resolve_slurs(part)
    assert part.findall(".//notations") == []


def test_other_notations_survive_removed_slur() -> None:
    part = slurred_part("1( 3)")
    notations = part.find("measure/note/notations")
    assert notations is not None
    ET.SubElement(notations, "fermata")

    resolve_slurs(part)

    assert part.find("measure/note/notations/fermata") is not None
    assert part.findall(".//slur") == []


def test_slur_numbers_are_paired_independently() -> None:
    part = slurred_part("1( 2)", number="1")
    first = part.find("measure/note/notations/slur")
    assert first is not None
    measure_two = part.findall("measure")[1]
    note = measure_two.findall("note")[1]
    notations = ET.SubElement(note, "notations")
    ET.SubElement(notations, "slur", type="start", number="2")
    measure_three = part.findall("measure")[2]
    note = measure_three.findall("note")[0]
    notations = ET.SubElement(note, "notations")
    ET.SubElement(notations, "slur", type="stop", number="2")

    assert resolve_slurs(part) == 0
    assert slur_pairs(part) == [(1, 2, "1"), (2, 3, "2")]


def test_build_part_resolves_slurs_after_tie_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    part_source = slurred_part("1( 3)", bars=3)
    measures = [ET.fromstring(ET.tostring(measure)) for measure in part_source.findall("measure")]
    calls: list[str] = []
    real_resolve = resolve_slurs

    monkeypatch.setattr(music_xml_generator, "build_measures", lambda *args, **kwargs: measures)
    monkeypatch.setattr(music_xml_generator, "convert_ties", lambda part: calls.append("ties"))

    def resolve(part: ET.Element) -> int:
        calls.append("slurs")
        return real_resolve(part)

    monkeypatch.setattr(music_xml_generator, "resolve_slurs", resolve)

    part = music_xml_generator.build_part(
        music_xml_generator.XmlGeneratorArguments(), [], index=0, has_two_staves=False
    )

    assert calls == ["ties", "slurs"]
    assert part.findall(".//slur") == []


def voice_bar(notes: list[tuple[str, str, str, list[tuple[str, str]]]]) -> ET.Element:
    """A part of one bar (plus an empty one after it), its notes in document
    order as (voice, step, onset marker, marks) -- marks are (slur|tie, type).
    Each note is a quarter; voice 2 is written after a backup to beat one."""
    part = ET.Element("part", id="P1")
    measure = ET.SubElement(part, "measure", number="1")
    current = None
    for voice, step, _, marks in notes:
        if current is not None and voice != current:
            backup = ET.SubElement(measure, "backup")
            ET.SubElement(backup, "duration").text = "4"
        current = voice
        note = ET.SubElement(measure, "note")
        pitch = ET.SubElement(note, "pitch")
        ET.SubElement(pitch, "step").text = step
        ET.SubElement(pitch, "octave").text = "4"
        ET.SubElement(note, "duration").text = "1"
        for kind, what in marks:
            if kind == "tie":
                ET.SubElement(note, "tie", type=what)
        ET.SubElement(note, "voice").text = voice
        ET.SubElement(note, "staff").text = "1"
        slurs = [what for kind, what in marks if kind == "slur"]
        if slurs:
            notations = ET.SubElement(note, "notations")
            for what in slurs:
                ET.SubElement(notations, "slur", type=what, number="1")
    ET.SubElement(part, "measure", number="2")
    return part


def slur_marks(part: ET.Element) -> list[tuple[str | None, str | None, str | None, bool]]:
    from homr.slur_resolution import INFERRED

    return [
        (
            note.findtext("voice"),
            note.findtext("pitch/step"),
            slur.get("type"),
            bool(slur.get(INFERRED)),
        )
        for note in part.iter("note")
        for slur in note.iter("slur")
    ]


def test_a_slur_over_a_tie_ends_where_the_tie_ends() -> None:
    """Vielako s01 staff 4 bar 8: a slur from D over an E tied to the next E.
    The tie took the one stop the model wrote, so the slur's stop is inferred
    on the note the tie ends on, and marked as inferred."""
    part = voice_bar(
        [
            ("1", "D", "", [("slur", "start")]),
            ("1", "E", "", [("tie", "start")]),
            ("1", "E", "", [("tie", "stop")]),
            ("1", "F", "", []),
        ]
    )
    resolve_slurs(part, keep_edges=False)
    assert slur_marks(part) == [("1", "D", "start", False), ("1", "E", "stop", True)]
    assert [
        n.findtext("pitch/step") for n in part.iter("note") if n.find("notations") is not None
    ] == ["D", "E"]
    third = list(part.iter("note"))[2]
    assert third.find("notations") is not None


def test_a_start_with_no_tie_after_it_is_still_removed() -> None:
    part = voice_bar([("1", "D", "", [("slur", "start")]), ("1", "E", "", []), ("1", "F", "", [])])
    resolve_slurs(part, keep_edges=False)
    assert slur_marks(part) == []


def test_a_slur_from_a_shared_notehead_moves_onto_the_voice_it_ends_in() -> None:
    """Illan s05 bar 1: both voices hold G on beat one; the model hung the slur
    on voice 2's copy and ended it on voice 1's F. It moves to voice 1's G."""

    def note(voice: str, step: str, slur: str = "") -> str:
        mark = f"<notations><slur type='{slur}' number='1'/></notations>" if slur else ""
        return (
            f"<note><pitch><step>{step}</step><octave>4</octave></pitch><duration>1</duration>"
            f"<voice>{voice}</voice><staff>1</staff>{mark}</note>"
        )

    part = ET.fromstring(
        "<part id='P1'><measure number='1'>"
        + note("2", "G", "start")
        + note("2", "D")
        + "<backup><duration>2</duration></backup>"
        + note("1", "G")
        + note("1", "F", "stop")
        + "</measure><measure number='2'/></part>"
    )
    resolve_slurs(part, keep_edges=False)
    assert sorted(slur_marks(part)) == sorted(
        [("1", "G", "start", False), ("1", "F", "stop", False)]
    )


def test_a_slur_between_voices_with_no_shared_head_is_left_alone() -> None:
    part = voice_bar(
        [
            ("2", "C", "", [("slur", "start")]),
            ("1", "F", "", [("slur", "stop")]),
        ]
    )
    resolve_slurs(part, keep_edges=False)
    assert sorted(slur_marks(part)) == sorted(
        [("2", "C", "start", False), ("1", "F", "stop", False)]
    )
