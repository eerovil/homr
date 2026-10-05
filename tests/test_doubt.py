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

from homr.doubt import (
    ACCIDENTAL_UNSURE,
    COPIES_DISAGREE,
    MARK_PREFIX,
    ODD_TIME,
    PITCH_UNSURE,
    RHYTHM_CLOSE,
    SECOND_READING,
    SECOND_READING_FAILED,
    confidence_doubts,
    find_doubts,
    mark_doubts,
    odd_time_doubts,
    reading_doubts,
    reading_gap,
)
from homr.transformer.vocabulary import EncodedSymbol

DATA = Path(__file__).resolve().parent.parent / "fixtures" / "doubt"


def _note(rhythm, pitch="C4", position="upper", ranked=None, pitch_p=1.0, lift_p=1.0, pos_p=1.0):
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


def _sure(rhythm, **kw):
    return _note(rhythm, ranked={rhythm: 0.95}, **kw)


# ----------------------------------------------------------- the decoder's doubt


def test_two_readings_nearly_as_likely_is_a_close_call():
    # Two notes in a 2/4 bar: quarter+quarter or (dotted quarter+eighth), about equally liked.
    unsure = [
        [_note("note_4", ranked={"note_4": 0.5, "note_4.": 0.45})],
        [_note("note_4", ranked={"note_4": 0.5, "note_8": 0.45})],
    ]
    assert reading_gap(unsure, Fraction(1, 2)) < 2.0
    sure = [[_sure("note_4")], [_sure("note_4")]]
    assert reading_gap(sure, Fraction(1, 2)) == math.inf


def test_no_reading_that_fills_the_bar_is_not_a_close_call():
    assert reading_gap([[_sure("note_4")]], Fraction(1, 1)) is None


def test_a_close_call_marks_its_staff_and_names_the_voice():
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


def test_an_unsure_pitch_or_accidental_marks_the_bar():
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


def test_two_voices_giving_one_notehead_different_lengths_is_a_doubt():
    staffs = [
        [
            _sure("note_4", pitch="E3", position="lower"),
            EncodedSymbol("chord"),
            _sure("note_6", pitch="E3", position="lower2"),
        ]
    ]
    assert COPIES_DISAGREE in confidence_doubts(staffs, [[Fraction(1, 4)]])[(0, 2, 1)]


def test_a_confident_clean_bar_is_not_marked():
    staffs = [[_sure("note_4"), _sure("note_4"), EncodedSymbol("barline")]]
    assert not confidence_doubts(staffs, [[Fraction(1, 2)]])


# --------------------------------------------------------------- the score itself


def _score(bars, staves=1):
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


def test_a_second_reading_that_differs_marks_only_that_bar_and_staff():
    first = _score([[(1, 1, "C", 8), ("backup", 8), (2, 1, "E", 8)], [(1, 1, "D", 8)]], staves=2)
    second = _score([[(1, 1, "C", 8), ("backup", 8), (2, 1, "F", 8)], [(1, 1, "D", 8)]], staves=2)
    assert reading_doubts(first, second) == {(0, 2, 1): {SECOND_READING}}
    assert not reading_doubts(first, copy.deepcopy(first))


def test_a_second_reading_with_another_bar_count_doubts_every_bar():
    first = _score([[(1, 1, "C", 8)], [(1, 1, "D", 8)]])
    second = _score([[(1, 1, "C", 8)]])
    assert set(reading_doubts(first, second)) == {(0, 1, 1), (0, 1, 2)}


def test_no_second_reading_means_nothing_was_checked():
    first = _score([[(1, 1, "C", 8)], [(1, 1, "D", 8)]])
    assert reading_doubts(first, None) == {
        (0, 1, 1): {SECOND_READING_FAILED},
        (0, 1, 2): {SECOND_READING_FAILED},
    }


def test_a_note_at_an_odd_time_is_a_doubt():
    # A 32nd-note offset: the third note starts at 3/32, on no sixteenth or triplet sixteenth.
    root = _score([[(1, 1, "C", 1), (1, 1, "D", 7)]])
    note = root.findall(".//note")[0]
    note.find("duration").text = "1"  # a 32nd, so D starts at 1/32
    assert odd_time_doubts(root) == {(0, 1, 1): {ODD_TIME}}
    assert not odd_time_doubts(_score([[(1, 1, "C", 4), (1, 1, "D", 4)]]))


def test_marks_are_red_words_on_their_staff_at_the_head_of_the_bar():
    root = _score([[(1, 1, "C", 8), ("backup", 8), (2, 1, "E", 8)]], staves=2)
    assert mark_doubts(root, {(0, 2, 1): {SECOND_READING, f"voice 2: {RHYTHM_CLOSE}"}}) == 1
    measure = root.find(".//measure")
    direction = measure.find("direction")
    assert [c.tag for c in measure][:2] == ["attributes", "direction"]
    words = direction.find("direction-type/words")
    assert words.get("color") == "#FF0000"
    assert words.text == (
        MARK_PREFIX + f"check against the page: voice 2: {RHYTHM_CLOSE}; {SECOND_READING}"
    )
    assert direction.findtext("staff") == "2"


# --------------------------------------------------- the bar that started it


def _symbols_from_sidecar(path):
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


def test_legenda_system_11_bar_25_bass_is_marked():
    staffs = _symbols_from_sidecar(DATA / "legenda-s11.confidence.json")
    xml = ET.parse(DATA / "legenda-s11.musicxml").getroot()
    second = ET.parse(DATA / "legenda-s11.second.musicxml").getroot()
    doubts = find_doubts(staffs, xml, second)
    assert (0, 2, 3) in doubts
    assert mark_doubts(xml, doubts) == len(doubts)
