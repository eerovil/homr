"""Triplets the decoder half-read: Legenda (Rautavaara), system 1, bar 2.

eerovil/musescore-choir-plugins#245. The two basses sing in unison under two
stems, so every head is written into both voices of the staff. The page prints a
quarter-and-eighth triplet twice and then two triplets of eighths. The decoder
read the first quarter as a triplet quarter and nothing else of that half as a
triplet; one voice read the beamed eighths as triplets and the other as plain.
The bar came out about seven beats long and the brackets never closed.

What is pinned: a triplet the decoder opened is closed, the one run that puts
the bar back on the beat is read as a triplet, the second voice follows the
head it shares, and each voice is bracketed where its notes add up -- plus the
refusals that keep each rule from guessing.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from fractions import Fraction

from homr.music_xml_generator import (
    XmlGeneratorArguments,
    bracket_tuplets,
    generate_xml,
    hide_shared_rests,
)
from homr.score_reconstruction import (
    ReconstructionChange,
    SymbolChord,
    complete_open_triplets,
    fill_unison_copies,
    repair_tuplet_overlaps_until_settled,
    triplets_onto_the_beat,
)
from homr.transformer.vocabulary import EncodedSymbol


def note(
    rhythm: str, position: str, pitch: str = "C3", alternatives: tuple[str, ...] = ()
) -> EncodedSymbol:
    confidence = None
    if alternatives:
        confidence = {
            "rhythm": {
                "value": rhythm,
                "alternatives": [
                    {"value": value, "probability": 0.01} for value in (rhythm, *alternatives)
                ],
            }
        }
    return EncodedSymbol(rhythm=rhythm, pitch=pitch, position=position, confidence=confidence)


def moment(*symbols: EncodedSymbol) -> SymbolChord:
    return SymbolChord(list(symbols))


def barline() -> SymbolChord:
    return SymbolChord([EncodedSymbol("barline")])


def signature() -> SymbolChord:
    return SymbolChord([EncodedSymbol("timeSignature/4")])


def rhythms(voice: list[SymbolChord], position: str) -> list[str]:
    return [s.rhythm for chord in voice for s in chord.symbols if s.position == position]


def opening_bar() -> list[SymbolChord]:
    """Bar 1: a rest and a half note, the span's opening bar."""
    return [
        signature(),
        moment(note("rest_4", "lower")),
        moment(note("note_2", "lower2", "A2"), note("note_2", "lower", "A2")),
        barline(),
    ]


def legenda_bar_2() -> list[SymbolChord]:
    """Bar 2 of the lower staff as the decoder read it, with its own alternatives."""
    unison = [
        ("C3", "note_8", ("note_12",), "note_12"),
        ("E3", "note_8", ("note_4", "note_12"), "note_12"),
        ("C3", "note_8", ("note_12",), "note_12"),
        ("D3", "note_8", ("note_12",), "note_12"),
        ("D3", "note_8", ("rest_32",), "note_12"),  # this head never offered note_12
        ("C3", "note_8", ("note_12",), "note_12"),
    ]
    return [
        moment(note("note_6", "lower", "G2", ("note_4",))),
        moment(
            note("note_8", "lower2", "C3", ("note_12",)),
            note("note_8", "lower", "C3", ("note_12",)),
        ),
        moment(
            note("note_4", "lower2", "D3", ("note_6",)),
            note("note_4", "lower", "D3", ("note_6",)),
        ),
        moment(
            note("note_8", "lower2", "C3", ("note_12",)),
            note("note_8", "lower", "C3", ("note_12",)),
        ),
        *[
            moment(
                note(second, "lower2", pitch, offered),
                note(first, "lower", pitch, ("note_8",)),
            )
            for pitch, second, offered, first in unison
        ],
        barline(),
    ]


def reconstruct(voice: list[SymbolChord]) -> list[SymbolChord]:
    voice = complete_open_triplets(voice)
    voice = triplets_onto_the_beat(voice)
    return repair_tuplet_overlaps_until_settled(voice)


def test_a_triplet_the_decoder_opened_is_closed_by_the_note_after_it() -> None:
    changes: list[ReconstructionChange] = []
    voice = complete_open_triplets(opening_bar() + legenda_bar_2(), changes)
    assert rhythms(voice, "lower")[2:4] == ["note_6", "note_12"]
    assert [(c.bar, c.pitch, c.before, c.after) for c in changes] == [
        (2, "C3", "note_8", "note_12")
    ]


def test_a_triplet_is_not_closed_with_a_value_the_decoder_never_offered() -> None:
    bar = [
        moment(note("note_6", "lower", "G2")),
        moment(note("note_8", "lower", "C3", ("note_16",))),
        moment(note("note_2", "lower", "D3")),
        barline(),
    ]
    assert rhythms(complete_open_triplets(bar), "lower") == ["note_6", "note_8", "note_2"]


def test_a_triplet_still_open_at_the_barline_is_left_as_read() -> None:
    bar = [
        moment(note("note_6", "lower", "G2")),
        moment(note("note_4", "lower", "C3", ("note_6",))),
        barline(),
    ]
    assert rhythms(complete_open_triplets(bar), "lower") == ["note_6", "note_4"]


def test_the_one_run_that_puts_the_bar_on_the_beat_is_read_as_a_triplet() -> None:
    changes: list[ReconstructionChange] = []
    voice = complete_open_triplets(opening_bar() + legenda_bar_2())
    voice = triplets_onto_the_beat(voice, changes)
    assert rhythms(voice, "lower")[2:6] == ["note_6", "note_12", "note_6", "note_12"]
    assert [(c.bar, c.pitch, c.before, c.after) for c in changes] == [
        (2, "D3", "note_4", "note_6"),
        (2, "C3", "note_8", "note_12"),
    ]


def test_two_runs_that_would_each_fit_are_two_readings_and_left_alone() -> None:
    bar = [
        moment(note("note_4", "lower", "D3", ("note_6",))),
        moment(note("note_8", "lower", "C3", ("note_12",))),
        moment(note("note_4", "lower", "D3", ("note_6",))),
        moment(note("note_8", "lower", "C3", ("note_12",))),
        moment(note("note_4", "lower", "D3", ("note_6",))),
        moment(note("note_8", "lower", "C3", ("note_12",))),
        barline(),
    ]
    voice = triplets_onto_the_beat(opening_bar() + bar)
    assert all(not EncodedSymbol(r).is_tuplet() for r in rhythms(voice[4:], "lower"))


def test_the_bar_that_opens_a_span_is_never_put_on_the_beat() -> None:
    bar = [
        signature(),
        moment(note("note_4", "lower", "D3", ("note_6",))),
        moment(note("note_8", "lower", "C3", ("note_12",))),
        moment(note("note_2", "lower", "D3")),
        barline(),
    ]
    assert rhythms(triplets_onto_the_beat(bar), "lower") == ["note_4", "note_8", "note_2"]


def test_without_a_time_signature_nothing_is_put_on_the_beat() -> None:
    voice = opening_bar()[1:] + legenda_bar_2()
    assert rhythms(triplets_onto_the_beat(voice), "lower") == rhythms(voice, "lower")


def test_the_second_voice_follows_the_head_it_shares_to_the_end_of_the_bar() -> None:
    changes: list[ReconstructionChange] = []
    voice = complete_open_triplets(opening_bar() + legenda_bar_2())
    voice = triplets_onto_the_beat(voice)
    voice = repair_tuplet_overlaps_until_settled(voice, changes)
    assert rhythms(voice, "lower2")[1:] == ["note_12", "note_6", "note_12"] + ["note_12"] * 6
    assert rhythms(voice, "lower2")[1:] == rhythms(voice, "lower")[3:]
    # The fifth beamed head offered no note_12 itself: the head beside it said so.
    assert any(c.pitch == "D3" and c.before == "note_8" for c in changes)


def test_a_different_pitch_beside_it_is_not_the_same_head() -> None:
    bar = [
        moment(note("note_12", "lower", "C3"), note("note_8", "lower2", "E3")),
        moment(note("note_12", "lower", "D3"), note("note_12", "lower2", "D3")),
        moment(note("note_12", "lower", "E3"), note("note_12", "lower2", "E3")),
        barline(),
    ]
    assert rhythms(repair_tuplet_overlaps_until_settled(bar), "lower2")[0] == "note_8"


def test_the_whole_bar_comes_out_one_whole_note_long_in_both_voices() -> None:
    staff = [
        EncodedSymbol("clef_F4", position="upper"),
        EncodedSymbol("timeSignature/4"),
    ]
    for chord in opening_bar()[1:] + legenda_bar_2():
        for index, symbol in enumerate(chord.symbols):
            if index:
                staff.append(EncodedSymbol("chord"))
            staff.append(symbol)
    xml = generate_xml(XmlGeneratorArguments(), [staff], "")
    measure = xml.findall(".//measure")[1]
    division = int(xml.findtext(".//divisions") or 0) * 4
    lengths: dict[str, Fraction] = {}
    for item in measure.findall("note"):
        if item.find("chord") is None:
            voice = item.findtext("voice") or ""
            lengths[voice] = lengths.get(voice, Fraction(0)) + Fraction(
                int(item.findtext("duration") or 0), division
            )
    # The second voice doubles the first note for note, so it is given the G it
    # skipped (`fill_unison_copies`) and both fill the bar.
    assert list(lengths.values()) == [Fraction(1), Fraction(1)]
    for voice in lengths:
        marks = [
            t.get("type")
            for item in measure.findall("note")
            if item.findtext("voice") == voice
            for t in item.iter("tuplet")
        ]
        assert marks == ["start", "stop"] * 4


def measure_of(*notes: tuple[str, int, bool]) -> ET.Element:
    """A measure of one voice: (pitch, duration in 12ths of a quarter, triplet?)."""
    measure = ET.Element("measure")
    for pitch, duration, triplet in notes:
        item = ET.SubElement(measure, "note")
        ET.SubElement(ET.SubElement(item, "pitch"), "step").text = pitch
        ET.SubElement(item, "duration").text = str(duration)
        ET.SubElement(item, "voice").text = "1"
        if triplet:
            modification = ET.SubElement(item, "time-modification")
            ET.SubElement(modification, "actual-notes").text = "3"
            ET.SubElement(modification, "normal-notes").text = "2"
    return measure


def marks(measure: ET.Element) -> list[str]:
    return [",".join(t.get("type") or "" for t in item.iter("tuplet")) for item in measure]


def test_a_quarter_and_an_eighth_under_one_triplet_are_one_bracket() -> None:
    measure = measure_of(("G", 8, True), ("C", 4, True), ("D", 4, True), ("D", 4, True))
    measure.append(measure_of(("C", 4, True))[0])
    bracket_tuplets(measure, 48)
    assert marks(measure) == ["start", "stop", "start", "", "stop"]


def test_a_run_that_starts_off_the_beat_or_never_adds_up_gets_no_bracket() -> None:
    off_beat = measure_of(("D", 4, True), ("E", 4, True), ("F", 4, True))
    forward = ET.Element("forward")
    ET.SubElement(forward, "duration").text = "4"
    off_beat.insert(0, forward)
    bracket_tuplets(off_beat, 48)
    assert marks(off_beat) == ["", "", "", ""]  # the forward, then three notes
    short = measure_of(("G", 8, True), ("C", 12, False))
    bracket_tuplets(short, 48)
    assert marks(short) == ["", ""]


def test_brackets_the_tokens_wrote_are_replaced_rather_than_added_to() -> None:
    measure = measure_of(("C", 4, True), ("D", 4, True), ("E", 4, True))
    for item in measure:
        ET.SubElement(ET.SubElement(item, "notations"), "tuplet", type="start")
    bracket_tuplets(measure, 48)
    assert marks(measure) == ["start", "", "stop"]


def test_a_triplet_eighth_leaving_a_gap_is_read_as_the_triplet_quarter_that_fills_it() -> None:
    """System 2, bar 2: the lower bass's G was read note_12 where the page prints note_6."""
    bar = [
        moment(note("rest_12", "lower"), note("note_12", "lower2", "G2", ("note_4", "note_6"))),
        moment(note("rest_12", "lower")),
        moment(note("note_12", "lower", "E3"), note("note_12", "lower2", "C3")),
        moment(note("note_6", "lower", "E3"), note("note_6", "lower2", "D3")),
        moment(note("note_12", "lower", "E3"), note("note_12", "lower2", "C3")),
        barline(),
    ]
    changes: list[ReconstructionChange] = []
    voice = repair_tuplet_overlaps_until_settled(bar, changes)
    assert rhythms(voice, "lower2")[0] == "note_6"
    assert [(c.pitch, c.before, c.after) for c in changes] == [("G2", "note_12", "note_6")]


def test_a_gap_is_not_filled_with_a_value_the_decoder_never_offered() -> None:
    bar = [
        moment(note("rest_12", "lower"), note("note_12", "lower2", "G2", ("note_4",))),
        moment(note("rest_12", "lower")),
        moment(note("note_12", "lower", "E3"), note("note_12", "lower2", "C3")),
        barline(),
    ]
    assert rhythms(repair_tuplet_overlaps_until_settled(bar), "lower2")[0] == "note_12"


def test_a_plain_quarter_among_triplet_eighths_is_read_as_a_triplet_quarter() -> None:
    """System 2, bar 2: the lower bass's last D was read note_4 among triplet eighths."""
    bar = [
        moment(note("note_6", "lower", "F3"), note("note_6", "lower2", "E3")),
        moment(note("note_12", "lower", "F3"), note("note_12", "lower2", "C3")),
        moment(
            note("note_12", "lower", "F3"), note("note_4", "lower2", "D3", ("note_6", "note_12"))
        ),
        moment(note("note_12", "lower", "F3")),
        moment(note("note_12", "lower", "F3"), note("note_12", "lower2", "C3")),
        barline(),
    ]
    voice = repair_tuplet_overlaps_until_settled(bar)
    assert rhythms(voice, "lower2") == ["note_6", "note_12", "note_6", "note_12"]


def test_a_value_is_not_read_off_a_clock_an_earlier_note_has_put_out() -> None:
    """System 1 once lost a quarter to a 16th: an earlier misread moved the clock under it.

    Read one at a time from the left, the earlier note is mended first and the
    later one then fits the value the page prints.
    """
    voice = complete_open_triplets(opening_bar() + legenda_bar_2())
    voice = triplets_onto_the_beat(voice)
    with_sixteenth = []
    for chord in voice:
        symbols = [
            (
                s.change_rhythm(s.rhythm)
                if s.confidence is None
                else EncodedSymbol(
                    s.rhythm,
                    s.pitch,
                    position=s.position,
                    confidence={
                        "rhythm": {
                            "alternatives": [
                                *s.confidence["rhythm"]["alternatives"],
                                {"value": "note_24", "probability": 0.01},
                            ]
                        }
                    },
                )
            )
            for s in chord.symbols
        ]
        with_sixteenth.append(SymbolChord(symbols))
    repaired = repair_tuplet_overlaps_until_settled(with_sixteenth)
    assert "note_24" not in rhythms(repaired, "lower2")
    assert rhythms(repaired, "lower2")[1:4] == ["note_12", "note_6", "note_12"]


def test_a_unison_copy_is_given_the_note_it_skipped() -> None:
    """System 2, bar 1: the second voice's copy of the middle beamed D was never read."""
    bar = [
        moment(note("note_12", "lower", "D3"), note("note_12", "lower2", "D3")),
        moment(note("note_12", "lower", "D3")),
        moment(note("note_12", "lower", "C3"), note("note_12", "lower2", "C3")),
        barline(),
    ]
    changes: list[ReconstructionChange] = []
    filled = fill_unison_copies(bar, changes)
    assert rhythms(filled, "lower2") == ["note_12"] * 3
    assert [(c.kind, c.pitch, c.staff) for c in changes] == [("unison_fill", "D3", "lower2")]


def test_a_voice_that_sings_its_own_line_is_not_filled() -> None:
    bar = [
        moment(note("note_12", "lower", "D3"), note("note_12", "lower2", "B2")),
        moment(note("note_12", "lower", "D3")),
        moment(note("note_12", "lower", "C3"), note("note_12", "lower2", "C3")),
        barline(),
    ]
    assert rhythms(fill_unison_copies(bar), "lower2") == ["note_12", "note_12"]


def test_a_voice_holding_a_rest_is_not_filled() -> None:
    bar = [
        moment(note("note_12", "lower", "D3"), note("rest_12", "lower2", "_")),
        moment(note("note_12", "lower", "D3")),
        moment(note("note_12", "lower", "C3"), note("note_12", "lower2", "C3")),
        moment(note("note_12", "lower", "C3"), note("note_12", "lower2", "C3")),
        barline(),
    ]
    assert rhythms(fill_unison_copies(bar), "lower2") == ["rest_12", "note_12", "note_12"]


def two_voices(*events: tuple[str, int, int, bool]) -> ET.Element:
    """A one-staff measure written voice by voice: (voice, onset, length, rest?)."""
    measure = ET.Element("measure")
    cursor = 0
    for voice, onset, length, rest in events:
        if onset != cursor:
            step = ET.SubElement(measure, "backup" if onset < cursor else "forward")
            ET.SubElement(step, "duration").text = str(abs(cursor - onset))
        item = ET.SubElement(measure, "note")
        if rest:
            ET.SubElement(item, "rest")
        else:
            ET.SubElement(ET.SubElement(item, "pitch"), "step").text = "G"
        ET.SubElement(item, "duration").text = str(length)
        ET.SubElement(item, "voice").text = voice
        ET.SubElement(item, "staff").text = "1"
        cursor = onset + length
    return measure


def test_a_rest_both_voices_share_is_written_hidden_into_the_one_without_it() -> None:
    """System 2, bar 2: the second tenor comes in after two rests printed once."""
    measure = two_voices(
        ("1", 0, 4, True), ("1", 4, 4, True), ("1", 8, 4, False), ("2", 8, 4, False)
    )
    assert hide_shared_rests(measure) == 2
    second = [item for item in measure if item.tag == "note" and item.findtext("voice") == "2"]
    assert [item.get("print-object") for item in second] == ["no", "no", None]
    # In time order: the hidden rests stand in front of the note the gap ends at.
    tags = [item.tag for item in measure]
    assert tags[tags.index("note", 3) - 1] == "backup"


def test_a_gap_the_other_voice_sings_through_is_left_alone() -> None:
    measure = two_voices(("1", 0, 8, False), ("1", 8, 4, False), ("2", 8, 4, False))
    assert hide_shared_rests(measure) == 0


def sangerhilsen_beat_four() -> list[SymbolChord]:
    """Sangerhilsen system 6, bar 1, beat 4: a unison triplet, one copy of E3 lost."""
    return [
        moment(
            note("note_12", "upper", "C4"),
            note("note_8", "lower2", "C3", ("note_12",)),
            note("note_12", "lower", "C3", ("note_12.", "note_6", "note_8")),
        ),
        moment(note("note_12", "upper", "E4"), note("note_12", "lower2", "E3")),
        moment(
            note("note_12", "upper", "G4"),
            note("note_12", "lower2", "G3"),
            note("note_12", "lower", "G3"),
        ),
        barline(),
    ]


def test_a_gap_the_other_voice_sings_in_is_a_lost_copy_not_a_long_note() -> None:
    voice = repair_tuplet_overlaps_until_settled(sangerhilsen_beat_four())
    assert rhythms(voice, "lower") == ["note_12", "note_12"]


def test_the_copy_may_be_either_voice_of_the_staff() -> None:
    changes: list[ReconstructionChange] = []
    filled = fill_unison_copies(sangerhilsen_beat_four(), changes)
    assert [(c.pitch, c.staff) for c in changes] == [("E3", "lower")]
    voice = repair_tuplet_overlaps_until_settled(filled)
    assert rhythms(voice, "lower") == ["note_12"] * 3
    assert rhythms(voice, "lower2") == ["note_12"] * 3
