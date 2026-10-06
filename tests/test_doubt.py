"""Marking the bars a reading is probably wrong about (homr/doubt.py).

eerovil/musescore-choir-plugins#245: the requirement is that no wrong bar goes
unmarked. The acceptance is the bar that started it, Legenda system 11's bar 25,
read for real and saved beside this test: its bass must be marked.
"""

import copy
import json
import math
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from typing import Any

from homr.doubt import (
    ACCIDENTAL_UNSURE,
    COPIES_DISAGREE,
    MARK_PREFIX,
    ODD_INTERVAL,
    ODD_TIME,
    PITCH_UNSURE,
    RHYTHM_CLOSE,
    SECOND_READING,
    SECOND_READING_FAILED,
    SILENT_BESIDE_CHORD,
    confidence_doubts,
    find_doubts,
    find_spots,
    interval_doubts,
    mark_doubts,
    odd_time_doubts,
    reading_doubts,
    reading_gap,
    silent_beside_chord_doubts,
)
from homr.transformer.vocabulary import EncodedSymbol

DATA = Path(__file__).resolve().parent.parent / "fixtures" / "doubt"


def _note(
    rhythm: str,
    pitch: str = "C4",
    position: str = "upper",
    ranked: dict[str, float] | None = None,
    pitch_p: float = 1.0,
    lift_p: float = 1.0,
    pos_p: float = 1.0,
) -> EncodedSymbol:
    ranked = ranked or {rhythm: 0.95}
    return EncodedSymbol(
        rhythm,
        pitch,
        position=position,
        confidence={
            "rhythm": {
                "value": rhythm,
                "probability": ranked.get(rhythm, 0.0),
                "alternatives": [
                    {"value": v, "probability": p}
                    for v, p in sorted(ranked.items(), key=lambda kv: -kv[1])
                ],
            },
            "pitch": {"probability": pitch_p},
            "lift": {"probability": lift_p},
            "position": {"probability": pos_p},
        },
    )


def _sure(rhythm: str, **kw: Any) -> EncodedSymbol:
    return _note(rhythm, ranked={rhythm: 0.95}, **kw)


# ----------------------------------------------------------- the decoder's doubt


def test_two_readings_nearly_as_likely_is_a_close_call() -> None:
    # Two notes in a 2/4 bar: quarter+quarter or (dotted quarter+eighth), about equally liked.
    unsure = [
        [_note("note_4", ranked={"note_4": 0.5, "note_4.": 0.45})],
        [_note("note_4", ranked={"note_4": 0.5, "note_8": 0.45})],
    ]
    gap = reading_gap(unsure, Fraction(1, 2))
    assert gap is not None and gap < 2.0
    sure = [[_sure("note_4")], [_sure("note_4")]]
    assert reading_gap(sure, Fraction(1, 2)) == math.inf


def test_no_reading_that_fills_the_bar_is_not_a_close_call() -> None:
    assert reading_gap([[_sure("note_4")]], Fraction(1, 1)) is None


def test_a_close_call_marks_its_staff_and_names_the_voice() -> None:
    staffs = [
        [
            _note("note_4", position="lower2", ranked={"note_4": 0.5, "note_4.": 0.45}),
            _note("note_4", position="lower2", ranked={"note_4": 0.5, "note_8": 0.45}),
            _sure("note_2", position="upper"),
            EncodedSymbol("barline"),
        ]
    ]
    doubts = confidence_doubts(staffs, [[Fraction(1, 2)]])
    assert doubts == {(0, 2, 1): {f"voice 2: {RHYTHM_CLOSE}"}}


def test_an_unsure_pitch_or_accidental_marks_the_bar() -> None:
    staffs = [
        [
            _sure("note_4", pitch_p=0.6),
            EncodedSymbol("barline"),
            _sure("note_4", lift_p=0.6),
            EncodedSymbol("barline"),
            _sure("note_4"),
        ]
    ]
    doubts = confidence_doubts(staffs, [[Fraction(1, 4)] * 3])
    assert doubts[(0, 1, 1)] == {f"voice 1: {PITCH_UNSURE}"}
    assert doubts[(0, 1, 2)] == {f"voice 1: {ACCIDENTAL_UNSURE}"}
    assert (0, 1, 3) not in doubts


def test_two_voices_giving_one_notehead_different_lengths_is_a_doubt() -> None:
    staffs = [
        [
            _sure("note_4", pitch="E3", position="lower"),
            EncodedSymbol("chord"),
            _sure("note_6", pitch="E3", position="lower2"),
        ]
    ]
    assert COPIES_DISAGREE in confidence_doubts(staffs, [[Fraction(1, 4)]])[(0, 2, 1)]


def test_a_confident_clean_bar_is_not_marked() -> None:
    staffs = [[_sure("note_4"), _sure("note_4"), EncodedSymbol("barline")]]
    assert not confidence_doubts(staffs, [[Fraction(1, 2)]])


# --------------------------------------------------------------- the score itself


def _score(bars: Any, staves: int = 1) -> ET.Element:
    """bars: per bar, list of (staff, voice, step, duration in 16ths) or ('backup', 16ths)."""
    root = ET.Element("score-partwise")
    part = ET.SubElement(root, "part", id="P1")
    for index, notes in enumerate(bars):
        measure = ET.SubElement(part, "measure", number=str(index + 1))
        if index == 0:
            attributes = ET.SubElement(measure, "attributes")
            ET.SubElement(attributes, "divisions").text = "8"
            ET.SubElement(attributes, "staves").text = str(staves)
            time = ET.SubElement(attributes, "time")
            ET.SubElement(time, "beats").text = "2"
            ET.SubElement(time, "beat-type").text = "4"
        for item in notes:
            if item[0] == "backup":
                ET.SubElement(ET.SubElement(measure, "backup"), "duration").text = str(item[1] * 2)
                continue
            staff, voice, step, sixteenths = item
            note = ET.SubElement(measure, "note")
            pitch = ET.SubElement(note, "pitch")
            ET.SubElement(pitch, "step").text = step
            ET.SubElement(pitch, "octave").text = "4"
            ET.SubElement(note, "duration").text = str(sixteenths * 2)
            ET.SubElement(note, "voice").text = str(voice)
            ET.SubElement(note, "staff").text = str(staff)
    return root


def test_a_second_reading_that_differs_marks_only_that_bar_and_staff() -> None:
    first = _score([[(1, 1, "C", 8), ("backup", 8), (2, 1, "E", 8)], [(1, 1, "D", 8)]], staves=2)
    second = _score([[(1, 1, "C", 8), ("backup", 8), (2, 1, "F", 8)], [(1, 1, "D", 8)]], staves=2)
    assert reading_doubts(first, second) == {(0, 2, 1): {SECOND_READING}}
    assert not reading_doubts(first, copy.deepcopy(first))


def test_a_second_reading_with_another_bar_count_doubts_every_bar() -> None:
    first = _score([[(1, 1, "C", 8)], [(1, 1, "D", 8)]])
    second = _score([[(1, 1, "C", 8)]])
    assert set(reading_doubts(first, second)) == {(0, 1, 1), (0, 1, 2)}


def test_no_second_reading_means_nothing_was_checked() -> None:
    first = _score([[(1, 1, "C", 8)], [(1, 1, "D", 8)]])
    assert reading_doubts(first, None) == {
        (0, 1, 1): {SECOND_READING_FAILED},
        (0, 1, 2): {SECOND_READING_FAILED},
    }


def test_a_note_at_an_odd_time_is_a_doubt() -> None:
    # A 32nd-note offset: the third note starts at 3/32, on no sixteenth or triplet sixteenth.
    root = _score([[(1, 1, "C", 1), (1, 1, "D", 7)]])
    note = root.findall(".//note")[0]
    duration = note.find("duration")
    assert duration is not None
    duration.text = "1"  # a 32nd, so D starts at 1/32
    assert odd_time_doubts(root) == {(0, 1, 1): {ODD_TIME}}
    assert not odd_time_doubts(_score([[(1, 1, "C", 4), (1, 1, "D", 4)]]))


def test_marks_are_red_words_on_their_staff_at_the_head_of_the_bar() -> None:
    root = _score([[(1, 1, "C", 8), ("backup", 8), (2, 1, "E", 8)]], staves=2)
    assert mark_doubts(root, {(0, 2, 1): {SECOND_READING, f"voice 2: {RHYTHM_CLOSE}"}}) == 1
    measure = root.find(".//measure")
    assert measure is not None
    direction = measure.find("direction")
    assert direction is not None
    assert [c.tag for c in measure][:2] == ["attributes", "direction"]
    words = direction.find("direction-type/words")
    assert words is not None
    assert words.get("color") == "#FF0000"
    assert words.text == MARK_PREFIX + "rhythm? notes?"
    assert direction.findtext("staff") == "2"


# --------------------------------------------------- the bar that started it


def _symbols_from_sidecar(path: Path) -> list[list[EncodedSymbol]]:
    """Rebuild the decoded symbols, chord links included, from a --output-confidence sidecar."""
    records = json.loads(path.read_text())["symbols"]
    staffs: dict[int, list[EncodedSymbol]] = {}
    last: dict[int, int] = {}
    for record in records:
        staff = record["staff"]
        out = staffs.setdefault(staff, [])
        if staff in last and record["symbol"] == last[staff] + 2:
            out.append(EncodedSymbol("chord"))
        token = record["token"]
        out.append(
            EncodedSymbol(
                token["rhythm"],
                token["pitch"],
                token["lift"],
                token["articulation"],
                token["slur"],
                token["position"],
                confidence=record["confidence"],
            )
        )
        last[staff] = record["symbol"]
    return [staffs[k] for k in sorted(staffs)]


def test_legenda_system_11_bar_25_bass_is_marked() -> None:
    staffs = _symbols_from_sidecar(DATA / "legenda-s11.confidence.json")
    xml = ET.parse(DATA / "legenda-s11.musicxml").getroot()
    second = ET.parse(DATA / "legenda-s11.second.musicxml").getroot()
    doubts = find_doubts(staffs, xml, second)
    assert (0, 2, 3) in doubts
    assert mark_doubts(xml, doubts) == len(doubts)


def _fifth(low_alter: int, high_alter: int) -> ET.Element:
    """One bar: a C and the G above it struck together in two voices of a staff."""
    root = _score([[(1, 1, "C", 4), ("backup", 4), (1, 2, "G", 4)]])
    for note, alter in zip(root.findall(".//note"), (low_alter, high_alter), strict=True):
        pitch = note.find("pitch")
        assert pitch is not None
        if alter:
            ET.SubElement(pitch, "alter").text = str(alter)
    return root


def test_a_doubly_diminished_fifth_is_a_doubt() -> None:
    """Finlandia system 8 (eerovil/musescore-choir-plugins#274), in made-up notes:
    a double flat read as a natural leaves a fifth two semitones short."""
    assert interval_doubts(_fifth(1, -1)) == {(0, 1, 1): {ODD_INTERVAL}}


def test_a_diminished_or_augmented_fifth_is_ordinary() -> None:
    for low, high in ((0, 0), (1, 0), (0, 1), (0, -1), (-1, 0)):
        assert not interval_doubts(_fifth(low, high)), (low, high)


def _two_voices(upper: str, lower: str) -> ET.Element:
    """One bar of one staff, eighth notes at 2 divisions, the two voices as written."""
    return ET.fromstring(
        '<score-partwise><part id="P1"><measure number="1"><attributes>'
        "<divisions>2</divisions></attributes>"
        f"{upper}<backup><duration>8</duration></backup>{lower}</measure></part></score-partwise>"
    )


def _n(step: str, voice: int, length: int = 2, chord: bool = False) -> str:
    return (
        "<note>"
        + ("<chord/>" if chord else "")
        + f"<pitch><step>{step}</step><octave>4</octave></pitch>"
        f"<duration>{length}</duration><voice>{voice}</voice><staff>1</staff></note>"
    )


def test_a_voice_silent_beside_the_other_voices_chord_is_a_doubt() -> None:
    """Finlandia system 10 (eerovil/musescore-choir-plugins#274), in made-up notes:
    the upper voice's third note went into the lower voice as a chord, leaving the
    upper voice nothing on that beat."""
    upper = _n("E", 1, 4) + "<forward><duration>2</duration></forward>" + _n("E", 1)
    lower = _n("C", 2, 4) + _n("C", 2) + _n("A", 2, chord=True) + _n("C", 2)
    assert silent_beside_chord_doubts(_two_voices(upper, lower)) == {
        (0, 1, 1): {SILENT_BESIDE_CHORD}
    }


def test_a_chord_beside_a_rest_or_a_note_is_ordinary() -> None:
    resting = _n("E", 1, 4) + "<note><rest/><duration>2</duration><voice>1</voice></note>"
    lower = _n("C", 2, 4) + _n("C", 2) + _n("A", 2, chord=True) + _n("C", 2)
    assert not silent_beside_chord_doubts(_two_voices(resting + _n("E", 1), lower))
    assert not silent_beside_chord_doubts(_two_voices(_n("E", 1, 6) + _n("E", 1), lower))


def test_the_notes_a_doubt_is_about_are_red_and_the_word_stands_above_them() -> None:
    """The owner's request on eerovil/musescore-choir-plugins#274: a short word,
    and the place in the bar it is about."""
    root = _fifth(1, -1)
    measure = root.find(".//measure")
    assert measure is not None
    doubts = interval_doubts(root)
    spots, words = find_spots([], root, None, doubts)
    assert mark_doubts(root, doubts, spots, words) == 1
    notes = measure.findall("note")
    assert [note.get("color") for note in notes] == ["#FF0000", "#FF0000"]
    assert all(note.find("notehead").get("color") == "#FF0000" for note in notes)  # type: ignore[union-attr]
    children = list(measure)
    direction = measure.find("direction")
    assert direction is not None
    assert children.index(direction) == children.index(notes[0]) - 1
    assert direction.findtext("direction-type/words") == MARK_PREFIX + "accidental?"


def test_a_second_reading_differing_only_in_pitch_says_pitch() -> None:
    first = _score([[(1, 1, "C", 8)]])
    second = _score([[(1, 1, "D", 8)]])
    doubts = reading_doubts(first, second)
    spots, words = find_spots([], first, second, doubts)
    assert words == {(0, 1, 1): {"pitch?"}}
    assert spots[(0, 1, 1)] == first.findall(".//note")


def test_a_doubt_with_no_notes_to_name_stands_at_the_head_of_the_bar() -> None:
    root = _score([[(1, 1, "C", 8)]])
    assert mark_doubts(root, {(0, 1, 1): {SECOND_READING_FAILED}}) == 1
    measure = root.find(".//measure")
    assert measure is not None
    assert [c.tag for c in measure][:2] == ["attributes", "direction"]
    assert measure.find("note").get("color") is None  # type: ignore[union-attr]
