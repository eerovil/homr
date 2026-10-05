"""Notes the decoder put into the wrong voice: Legenda (Rautavaara), system 9, bar 18.

eerovil/musescore-choir-plugins#265. The two basses cross on beat 3: the second
voice's G is printed above the first voice's E, each with its own stem. The
decoder read the two heads as one chord of the first voice, so the second voice
had nothing there, ended a beat short, and both voices were written early from
there on. Two smaller misreadings of the same system -- the second tenor's last
quarter of bar 18 read as a triplet quarter, and a skipped unison eighth in bar 19
taken for an open triplet -- came out once the voices were right.

The staff here is the decoder's own reading of system 9, bars 18 and 19, token by
token: values, the alternatives it ranked under them, and the stems the
segmentation found. What is pinned is that every voice comes out as the page
prints it, plus the refusals that keep the stem rule from guessing.
"""

from __future__ import annotations

from fractions import Fraction

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.score_reconstruction import (
    ReconstructionChange,
    SymbolChord,
    voices_from_opposite_stems,
)
from homr.transformer.vocabulary import EncodedSymbol

CHORD = EncodedSymbol("chord")
BARLINE = EncodedSymbol("barline")
_STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def head(
    rhythm: str,
    position: str,
    pitch: str,
    lift: str,
    stem: str | None,
    alternatives: tuple[str, ...],
    slur: str = "_",
) -> EncodedSymbol:
    confidence = {
        "rhythm": {
            "value": rhythm,
            "alternatives": [{"value": value} for value in (rhythm, *alternatives)],
        }
    }
    articulation = "_" if rhythm.startswith(("note", "rest")) else "."
    return EncodedSymbol(
        rhythm,
        pitch=pitch,
        lift=lift,
        articulation=articulation,
        slur=slur,
        position=position,
        confidence=confidence,
        stem_direction=stem,
    )


def system_9() -> list[EncodedSymbol]:
    """Bars 18 and 19 as the decoder read them, both staves."""
    return [
        EncodedSymbol("clef_G2", position="upper"),
        CHORD,
        EncodedSymbol("clef_F4", position="lower"),
        EncodedSymbol("keySignature_0"),
        head(
            "note_6",
            "upper",
            "B4",
            "b",
            "up",
            (
                "note_4",
                "voltaStart",
                "timeSignature/4",
                "note_3",
                "rest_6",
                "note_5",
                "keySignature_-1",
            ),
        ),
        CHORD,
        head(
            "note_4",
            "upper2",
            "G4",
            "_",
            "down",
            ("note_6", "note_6.", "note_3", "rest_6", "note_3.", "note_36", "note_4.."),
        ),
        CHORD,
        head(
            "note_6",
            "lower",
            "F3",
            "_",
            "up",
            ("note_4", "note_3", "note_5", "rest_6", "note_6.", "rest_12", "note_2"),
        ),
        CHORD,
        head(
            "note_4",
            "lower2",
            "G2",
            "_",
            "down",
            ("note_6", "note_6.", "note_3", "note_5", "rest_12", "rest_6", "note_4.."),
        ),
        head(
            "note_12",
            "upper",
            "B4",
            "b",
            "up",
            ("note_5", "note_10.", "note_10", "note_3", "note_6", "note_16.", "note_11"),
            slur="slurStart",
        ),
        CHORD,
        head(
            "note_8",
            "upper2",
            "G4",
            "_",
            "down",
            ("note_12", "note_16", "note_5", "rest_12", "note_2", "note_6", "note_8."),
        ),
        CHORD,
        head(
            "note_12",
            "lower",
            "F3",
            "_",
            "up",
            ("note_8", "note_5", "note_2", "note_16", "rest_16", "note_16.", "rest_2"),
            slur="slurStart",
        ),
        CHORD,
        head(
            "note_8",
            "lower2",
            "C3",
            "_",
            "down",
            ("note_12", "note_5", "rest_12", "note_8G", "note_4", "note_16", "note_8."),
        ),
        head(
            "note_12",
            "upper",
            "B4",
            "b",
            "up",
            ("note_10.", "note_8", "note_5", "note_2", "note_12.", "note_10", "note_3."),
            slur="slurStop",
        ),
        CHORD,
        head(
            "note_8",
            "upper2",
            "G4",
            "_",
            "down",
            ("note_12", "note_64", "rest_12", "note_16.", "note_6", "note_2", "keySignature_2"),
        ),
        CHORD,
        head(
            "note_12",
            "lower",
            "F3",
            "_",
            "up",
            ("note_8", "note_16.", "rest_12", "note_5", "note_12.", "note_2", "rest_6"),
            slur="slurStop",
        ),
        CHORD,
        head(
            "note_8",
            "lower2",
            "D3",
            "_",
            "down",
            ("note_12", "rest_12", "note_5", "note_8G", "note_16.", "note_64", "note_22"),
        ),
        head(
            "note_12",
            "upper",
            "B4",
            "b",
            "up",
            ("note_8", "note_5", "note_10.", "note_10", "note_12.", "note_11", "note_3."),
        ),
        CHORD,
        head(
            "note_8",
            "upper2",
            "G4",
            "_",
            "down",
            ("note_12", "note_16", "rest_16", "note_16.", "rest_12", "rest_1.", "keySignature_5"),
        ),
        CHORD,
        head(
            "note_12",
            "lower",
            "F3",
            "_",
            "up",
            ("note_8", "note_16", "note_5", "note_6", "note_4", "note_8.", "note_16."),
        ),
        CHORD,
        head(
            "note_8",
            "lower2",
            "E3",
            "b",
            "down",
            ("note_12", "note_6", "rest_12", "note_5", "note_16", "rest_6", "rest_4"),
        ),
        head(
            "note_12",
            "upper2",
            "A4",
            "b",
            "both",
            ("note_2", "note_10.", "note_10", "note_12.", "note_8", "rest_1", "note_2.."),
        ),
        CHORD,
        head(
            "note_12",
            "upper",
            "A4",
            "b",
            "both",
            ("note_8", "rest_12", "note_8G", "rest_4", "note_4", "note_2", "note_10."),
        ),
        CHORD,
        head(
            "note_12",
            "lower",
            "F3",
            "_",
            "both",
            ("note_8", "note_2", "note_4", "rest_4", "rest_2", "note_2.", "note_1"),
        ),
        head(
            "note_4",
            "upper2",
            "G4",
            "_",
            "both",
            ("note_2", "note_8", "note_6", "rest_12", "note_4G", "note_16.", "note_1"),
        ),
        CHORD,
        head(
            "note_4",
            "upper",
            "G4",
            "_",
            "both",
            ("note_6", "note_2", "rest_12", "rest_4", "timeSignature/2", "note_6.", "rest_6"),
        ),
        CHORD,
        head(
            "note_6",
            "lower",
            "G3",
            "_",
            "down",
            ("note_4", "note_3", "note_6.", "note_2", "rest_6", "note_20", "note_4.."),
        ),
        CHORD,
        head(
            "note_6",
            "lower",
            "E3",
            "_",
            "up",
            ("note_4", "note_6.", "note_3", "rest_12", "rest_6", "note_20", "note_5"),
        ),
        head(
            "note_12",
            "lower2",
            "C3",
            "_",
            "down",
            ("note_6", "rest_12", "note_3", "note_11", "note_5", "note_12.", "note_16."),
        ),
        head(
            "note_4",
            "upper",
            "C5",
            "_",
            "up",
            ("note_12.", "note_1", "note_8", "rest_1", "note_16", "note_2.", "rest_4"),
        ),
        CHORD,
        head(
            "note_4",
            "upper2",
            "G4",
            "_",
            "down",
            ("note_6", "note_8", "note_6.", "note_16", "keySignature_-1", "note_3.", "rest_12"),
        ),
        CHORD,
        head(
            "note_4",
            "lower",
            "E3",
            "_",
            "up",
            ("note_6", "note_6.", "note_3", "note_4..", "note_8", "note_20", "rest_6"),
        ),
        CHORD,
        head(
            "note_6",
            "lower2",
            "B2",
            "b",
            "down",
            ("note_12", "note_4", "note_6.", "note_3", "rest_6", "note_40", "rest_12"),
        ),
        head(
            "note_12",
            "lower2",
            "A2",
            "_",
            "down",
            ("note_3", "note_6", "note_12.", "note_11", "note_16.", "note_5", "clef_C1"),
        ),
        BARLINE,
        head(
            "rest_2",
            "upper",
            "_",
            "_",
            None,
            ("rest_12", "note_16.", "note_6", "rest_6", "keySignature_0", "note_8G", "note_3."),
        ),
        CHORD,
        head(
            "note_12",
            "lower2",
            "G2",
            "_",
            "both",
            ("note_8", "note_16", "note_2", "note_4", "note_5", "note_32", "rest_6"),
        ),
        CHORD,
        head(
            "note_12",
            "lower",
            "G2",
            "_",
            "both",
            ("rest_4", "note_2", "note_8", "rest_12", "rest_2", "note_8G", "note_2."),
        ),
        head(
            "note_12",
            "lower2",
            "C3",
            "_",
            "both",
            ("note_8", "note_4", "note_8.", "note_2", "rest_4", "note_12.", "note_8.."),
        ),
        CHORD,
        head(
            "note_12",
            "lower",
            "C3",
            "_",
            "both",
            ("rest_4", "repeatStart", "rest_64", "rest_7m", "note_4", "clef_C2", "clef_C1"),
        ),
        head(
            "note_12",
            "lower2",
            "D3",
            "_",
            "up",
            ("note_12.", "rest_2.", "rest_4", "note_2", "note_8", "rest_7m", "rest_64"),
        ),
        CHORD,
        head(
            "note_12",
            "lower",
            "D3",
            "_",
            "down",
            (
                "rest_4",
                "clef_C1",
                "repeatStart",
                "rest_7m",
                "rest_12",
                "keySignature_2",
                "timeSignature/16",
            ),
        ),
        head(
            "note_4",
            "lower2",
            "E3",
            "b",
            "both",
            ("note_6", "note_12", "note_6.", "rest_12", "rest_1", "keySignature_-1", "note_12."),
        ),
        CHORD,
        head(
            "note_6",
            "lower",
            "E3",
            "b",
            "both",
            ("note_4", "note_12", "note_6.", "rest_12", "note_3", "rest_6", "note_24"),
        ),
        head(
            "note_12",
            "lower2",
            "C3",
            "_",
            None,
            ("note_8", "note_6", "note_2", "clef_G2", "note_5", "note_11", "note_4"),
        ),
        head(
            "rest_4",
            "upper",
            "_",
            "_",
            None,
            ("rest_12", "note_12", "note_12.", "rest_24", "rest_6", "rest_8", "note_8"),
        ),
        CHORD,
        head(
            "note_2",
            "lower",
            "F3",
            "#",
            "both",
            ("note_6", "note_12", "note_6.", "note_3", "note_3.", "note_4", "note_20"),
            slur="slurStart",
        ),
        CHORD,
        head(
            "note_2",
            "lower2",
            "F3",
            "#",
            "both",
            ("rest_4", "rest_2", "note_12", "clef_F4", "rest_16", "note_6.", "rest_32"),
        ),
        head(
            "note_4",
            "upper",
            "D5",
            "#",
            "up",
            ("rest_4", "rest_2", "note_16.", "note_16", "note_6", "note_8G", "note_96"),
        ),
        CHORD,
        head(
            "note_4",
            "upper2",
            "B4",
            "_",
            "down",
            ("note_2", "note_96", "rest_1.", "rest_7m", "rest_0", "timeSignature/16", "note_6"),
        ),
        BARLINE,
    ]


Timeline = list[tuple[Fraction, tuple[int, ...], Fraction]]


def voices(staff: list[EncodedSymbol]) -> dict[tuple[int, int, int], Timeline]:
    """(bar, staff, voice of the staff) -> (onset, MIDI pitches, duration), in quarters."""
    xml = generate_xml(XmlGeneratorArguments(), [staff], "")
    division = int(xml.findtext(".//divisions") or 0)
    out: dict[tuple[int, int, int], Timeline] = {}
    for bar, measure in enumerate(xml.findall(".//measure"), start=1):
        clock = start = Fraction(0)
        for item in measure:
            if item.tag in ("backup", "forward"):
                step = Fraction(int(item.findtext("duration") or 0), division)
                clock += step if item.tag == "forward" else -step
                continue
            if item.tag != "note":
                continue
            duration = Fraction(int(item.findtext("duration") or 0), division)
            in_chord = item.find("chord") is not None
            if not in_chord:
                start = clock
                clock += duration
            pitch = item.find("pitch")
            if pitch is None:
                continue
            midi = (
                12 * (int(pitch.findtext("octave") or 0) + 1)
                + _STEPS[pitch.findtext("step") or "C"]
                + int(float(pitch.findtext("alter") or 0))
            )
            key = (
                bar,
                int(item.findtext("staff") or 1),
                (int(item.findtext("voice") or 1) - 1) % 4,
            )
            line = out.setdefault(key, [])
            if in_chord and line and line[-1][0] == start:
                line[-1] = (start, tuple(sorted((*line[-1][1], midi))), line[-1][2])
            else:
                line.append((start, (midi,), duration))
    return out


def line(*notes: tuple[str, str | int, str]) -> Timeline:
    """(onset, MIDI pitch, duration), onsets and durations written as fractions of a quarter."""
    return [(Fraction(at), (int(pitch),), Fraction(length)) for at, pitch, length in notes]


# What the page prints. Triplet quarters last 2/3, triplet eighths 1/3.
PAGE = {
    # Bar 18, tenors: a quarter-and-eighth triplet, a triplet of eighths, two quarters.
    (1, 1, 0): line(
        ("0", 70, "2/3"),
        ("2/3", 70, "1/3"),
        ("1", 70, "1/3"),
        ("4/3", 70, "1/3"),
        ("5/3", 68, "1/3"),
        ("2", 67, "1"),
        ("3", 72, "1"),
    ),
    (1, 1, 1): line(
        ("0", 67, "2/3"),
        ("2/3", 67, "1/3"),
        ("1", 67, "1/3"),
        ("4/3", 67, "1/3"),
        ("5/3", 68, "1/3"),
        ("2", 67, "1"),
        ("3", 67, "1"),
    ),
    # Bar 18, first bass: F F under the bracket, F F F under the upper beam, E, E.
    (1, 2, 0): line(
        ("0", 53, "2/3"),
        ("2/3", 53, "1/3"),
        ("1", 53, "1/3"),
        ("4/3", 53, "1/3"),
        ("5/3", 53, "1/3"),
        ("2", 52, "1"),
        ("3", 52, "1"),
    ),
    # Bar 18, second bass: G C, D E-flat F under the lower beam, then G C and B-flat A.
    (1, 2, 1): line(
        ("0", 43, "2/3"),
        ("2/3", 48, "1/3"),
        ("1", 50, "1/3"),
        ("4/3", 51, "1/3"),
        ("5/3", 53, "1/3"),
        ("2", 55, "2/3"),
        ("8/3", 48, "1/3"),
        ("3", 46, "2/3"),
        ("11/3", 45, "1/3"),
    ),
    # Bar 19, both basses in unison: G C D, E-flat C, then F-sharp held for a half.
    (2, 2, 0): line(
        ("0", 43, "1/3"),
        ("1/3", 48, "1/3"),
        ("2/3", 50, "1/3"),
        ("1", 51, "2/3"),
        ("5/3", 48, "1/3"),
        ("2", 54, "2"),
    ),
    (2, 2, 1): line(
        ("0", 43, "1/3"),
        ("1/3", 48, "1/3"),
        ("2/3", 50, "1/3"),
        ("1", 51, "2/3"),
        ("5/3", 48, "1/3"),
        ("2", 54, "2"),
    ),
    # Bar 19, tenors: rest, then the first tenor's D-sharp over the second's B.
    (2, 1, 0): line(("3", 75, "1")),
    (2, 1, 1): line(("3", 71, "1")),
}


def test_every_voice_of_bars_18_and_19_comes_out_as_the_page_prints_it() -> None:
    assert voices(system_9()) == PAGE


def test_the_second_bass_g_is_moved_out_of_the_first_basss_chord() -> None:
    changes: list[ReconstructionChange] = []
    staff = [
        SymbolChord(
            [
                head("note_6", "lower", "G3", "_", "down", ()),
                head("note_6", "lower", "E3", "_", "up", ()),
            ]
        )
    ]
    moved = voices_from_opposite_stems(staff, changes)
    assert [(s.pitch, s.position) for s in moved[0].symbols] == [("G3", "lower2"), ("E3", "lower")]
    assert all(s.split_from_chord for s in moved[0].symbols)
    assert [(c.kind, c.pitch, c.before, c.after) for c in changes] == [
        ("voice_from_stem", "G3", "lower", "lower2")
    ]


def test_an_up_stem_note_of_the_second_voice_goes_to_the_first() -> None:
    staff = [
        SymbolChord(
            [
                head("note_4", "upper2", "C5", "_", "up", ()),
                head("note_4", "upper2", "G4", "_", "down", ()),
            ]
        )
    ]
    moved = voices_from_opposite_stems(staff)
    assert [(s.pitch, s.position) for s in moved[0].symbols] == [("C5", "upper"), ("G4", "upper2")]


def test_a_chord_under_one_stem_is_left_a_chord() -> None:
    chord = [
        head("note_4", "lower", "G3", "_", "up", ()),
        head("note_4", "lower", "E3", "_", "up", ()),
    ]
    moved = voices_from_opposite_stems([SymbolChord(chord)])
    assert [s.position for s in moved[0].symbols] == ["lower", "lower"]
    assert not any(s.split_from_chord for s in moved[0].symbols)


def test_a_head_without_a_stem_of_its_own_is_no_evidence() -> None:
    for stem in (None, "both"):
        chord = [
            head("note_4", "lower", "G3", "_", "down", ()),
            head("note_4", "lower", "E3", "_", "up", ()),
            head("note_4", "lower", "C3", "_", stem, ()),
        ]
        moved = voices_from_opposite_stems([SymbolChord(chord)])
        assert [s.position for s in moved[0].symbols] == ["lower"] * 3


def test_nothing_is_moved_into_a_voice_that_already_sounds_there() -> None:
    chord = [
        head("note_4", "lower", "G3", "_", "down", ()),
        head("note_4", "lower", "E3", "_", "up", ()),
        head("note_4", "lower2", "C3", "_", "down", ()),
    ]
    moved = voices_from_opposite_stems([SymbolChord(chord)])
    assert [s.position for s in moved[0].symbols] == ["lower", "lower", "lower2"]
