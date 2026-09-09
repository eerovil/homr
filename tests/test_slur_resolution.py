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
    part = slurred_part("1) 2( 2)")
    assert resolve_slurs(part) == 0
    assert slur_pairs(part) == [(2, 2, "1")]


def test_start_that_never_stops_is_removed() -> None:
    part = slurred_part("1( 1) 3(")
    assert resolve_slurs(part) == 0
    assert slur_pairs(part) == [(1, 1, "1")]


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
