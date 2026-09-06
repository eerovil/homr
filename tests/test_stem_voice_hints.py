import xml.etree.ElementTree as ET
from types import SimpleNamespace

from homr.model import StemDirection
from homr.music_xml_generator import (
    XmlGeneratorArguments,
    generate_xml,
    rebalance_measure_voices,
)
from homr.stem_voice_hints import SHARED, add_stem_voice_hints
from homr.transformer.vocabulary import EncodedSymbol


def _note(x: float, y: float, directions: list[StemDirection]) -> SimpleNamespace:
    return SimpleNamespace(center=(x, y), stem_directions=directions)


def _symbol(x: float, y: float) -> EncodedSymbol:
    return EncodedSymbol("note_4", pitch="C4", position="upper", coordinates=(x, y))


def test_a_decoded_note_takes_the_stem_of_the_notehead_it_is_at() -> None:
    """Both sides are in the staff image's coordinates, so this is a lookup.

    It used to be a fit: the coordinates handed to this module stopped short of
    the transforms the image went through, so an affine was recovered to undo
    the difference.  With the staff transformed the whole way there is nothing
    to undo, and all that is left of the decoder is a few pixels of jitter.
    """
    notes = [
        _note(20 + index * 30, 50 if index % 2 == 0 else 80, [StemDirection.UP])
        for index in range(8)
    ]
    jitter = [3, -2, 1, 4, -3, 0, 2, -1]
    symbols = [
        _symbol(note.center[0] + shift, note.center[1] - shift)
        for note, shift in zip(notes, jitter, strict=True)
    ]

    assert add_stem_voice_hints(symbols, notes) == 8
    assert [symbol.stem_direction for symbol in symbols] == ["up"] * 8


def test_a_decoded_note_nowhere_near_a_notehead_gets_no_stem() -> None:
    notes = [_note(20, 50, [StemDirection.UP])]
    symbols = [_symbol(200, 50)]

    assert add_stem_voice_hints(symbols, notes) == 0
    assert symbols[0].stem_direction is None


def test_geometry_hints_refuse_shared_or_ambiguous_noteheads() -> None:
    notes = [
        _note(20 + index * 30, 50 if index % 2 == 0 else 80, [StemDirection.UP])
        for index in range(8)
    ]
    shared = _note(260, 50, [StemDirection.UP, StemDirection.DOWN])
    ambiguous = _note(261, 50, [StemDirection.DOWN])
    notes.extend([shared, ambiguous])
    symbols = [_symbol(note.center[0] + 2, note.center[1]) for note in notes[:8]]
    symbols.extend([_symbol(262, 50), _symbol(261.5, 50)])

    assert add_stem_voice_hints(symbols, notes) == 8
    assert [symbol.stem_direction for symbol in symbols[-2:]] == [None, None]


def test_mixed_stem_chord_becomes_two_simultaneous_voices() -> None:
    up = EncodedSymbol("note_4", pitch="C4", position="upper", stem_direction="up")
    down = EncodedSymbol("note_4", pitch="C3", position="upper", stem_direction="down")
    xml = generate_xml(
        XmlGeneratorArguments(), [[up, EncodedSymbol("chord"), down, EncodedSymbol("barline")]], ""
    )
    measure = xml.find(".//measure")
    assert measure is not None
    notes = measure.findall("note")

    assert [note.findtext("voice") for note in notes] == ["1", "2"]
    assert len(measure.findall("backup")) == 1


def test_one_voice_keeps_its_voice_however_its_stems_point() -> None:
    """A staff with one voice stems by height, not by voice."""
    measure = ET.fromstring(
        """<measure number="1">
             <note><pitch><step>D</step><octave>5</octave></pitch>
               <duration>4</duration><voice>1</voice><stem>down</stem><staff>1</staff></note>
             <note><pitch><step>E</step><octave>4</octave></pitch>
               <duration>4</duration><voice>1</voice><stem>up</stem><staff>1</staff></note>
           </measure>"""
    )

    rebalance_measure_voices(measure)

    assert [note.findtext("voice") for note in measure.findall("note")] == ["1", "1"]


def test_two_voices_sounding_together_are_told_apart_by_their_stems() -> None:
    measure = ET.fromstring(
        """<measure number="1">
             <note><pitch><step>D</step><octave>5</octave></pitch>
               <duration>4</duration><voice>1</voice><stem>up</stem><staff>1</staff></note>
             <backup><duration>4</duration></backup>
             <note><pitch><step>F</step><octave>4</octave></pitch>
               <duration>4</duration><voice>1</voice><stem>down</stem><staff>1</staff></note>
           </measure>"""
    )

    rebalance_measure_voices(measure)

    assert [note.findtext("voice") for note in measure.findall("note")] == ["1", "2"]


def test_a_head_drawn_with_both_stems_is_written_into_both_voices() -> None:
    """The unison the page prints as one head is two singing parts."""
    shared = EncodedSymbol("note_4", pitch="C4", position="upper", stem_direction=SHARED)
    xml = generate_xml(XmlGeneratorArguments(), [[shared, EncodedSymbol("barline")]], "")
    measure = xml.find(".//measure")
    assert measure is not None
    notes = measure.findall("note")

    assert [note.findtext("voice") for note in notes] == ["1", "2"]
    assert [note.findtext("stem") for note in notes] == ["up", "down"]
    assert len(measure.findall("backup")) == 1
    pitches = {(note.findtext("pitch/step"), note.findtext("pitch/octave")) for note in notes}
    assert pitches == {("C", "4")}
    assert {note.findtext("duration") for note in notes} == {notes[0].findtext("duration")}


def test_the_mark_never_reaches_the_file() -> None:
    shared = EncodedSymbol("note_4", pitch="C4", position="upper", stem_direction=SHARED)
    xml = generate_xml(XmlGeneratorArguments(), [[shared, EncodedSymbol("barline")]], "")

    assert not [note for note in xml.iter("note") if note.get("stem-shared")]


def test_a_shared_head_still_says_the_staff_has_two_voices() -> None:
    """The gate `_staff_carries_two_voices` feeds reads the same mark.

    The head below the middle line stems up, which on a one-voice staff is how
    a low note is drawn and on a two-voice staff is what the upper voice does.
    The shared head is what settles it, so the stem is honoured -- and taking
    the mark further must not cost that.
    """
    measure = ET.fromstring(
        """<measure number="1">
             <note stem-shared="yes"><pitch><step>C</step><octave>4</octave></pitch>
               <duration>4</duration><voice>1</voice><staff>1</staff></note>
             <note><pitch><step>E</step><octave>4</octave></pitch>
               <duration>4</duration><voice>1</voice><stem>down</stem><staff>1</staff></note>
           </measure>"""
    )

    rebalance_measure_voices(measure, {1: ("G", 2, 0)})

    assert [note.findtext("voice") for note in measure.findall("note")] == ["1", "2", "2"]


def test_a_head_is_not_doubled_into_a_voice_already_sounding() -> None:
    """Then the second voice is in the bar under its own stem."""
    measure = ET.fromstring(
        """<measure number="1">
             <note stem-shared="yes"><pitch><step>D</step><octave>5</octave></pitch>
               <duration>4</duration><voice>1</voice><staff>1</staff></note>
             <backup><duration>4</duration></backup>
             <note><pitch><step>F</step><octave>4</octave></pitch>
               <duration>4</duration><voice>1</voice><stem>down</stem><staff>1</staff></note>
           </measure>"""
    )

    rebalance_measure_voices(measure)

    assert [note.findtext("voice") for note in measure.findall("note")] == ["1", "2"]
    assert len(measure.findall("backup")) == 1


def test_only_the_shared_head_of_a_chord_is_doubled() -> None:
    """A chord tone under one stem is one voice's; the shared head is both."""
    measure = ET.fromstring(
        """<measure number="1">
             <note stem-shared="yes"><pitch><step>C</step><octave>4</octave></pitch>
               <duration>4</duration><voice>1</voice><staff>1</staff></note>
             <note><chord/><pitch><step>E</step><octave>4</octave></pitch>
               <duration>4</duration><voice>1</voice><staff>1</staff></note>
           </measure>"""
    )

    rebalance_measure_voices(measure)
    notes = measure.findall("note")

    assert [note.findtext("pitch/step") for note in notes] == ["C", "E", "C"]
    assert [note.findtext("voice") for note in notes] == ["1", "1", "2"]
    assert notes[2].find("chord") is None


def test_a_grace_note_has_no_duration_to_double_behind() -> None:
    measure = ET.fromstring(
        """<measure number="1">
             <note stem-shared="yes"><grace/><pitch><step>C</step><octave>4</octave></pitch>
               <voice>1</voice><staff>1</staff></note>
           </measure>"""
    )

    rebalance_measure_voices(measure)

    assert len(measure.findall("note")) == 1
    assert not measure.findall("backup")
