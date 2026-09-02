import xml.etree.ElementTree as ET
from types import SimpleNamespace

from homr.model import StemDirection
from homr.music_xml_generator import (
    XmlGeneratorArguments,
    generate_xml,
    rebalance_measure_voices,
)
from homr.stem_voice_hints import add_stem_voice_hints
from homr.transformer.vocabulary import EncodedSymbol


def _note(x: float, y: float, directions: list[StemDirection]) -> SimpleNamespace:
    return SimpleNamespace(center=(x, y), stem_directions=directions)


def _symbol(x: float, y: float) -> EncodedSymbol:
    return EncodedSymbol("note_4", pitch="C4", position="upper", coordinates=(x, y))


def test_geometry_hints_calibrate_without_ground_truth() -> None:
    notes = [
        _note(20 + index * 30, 50 if index % 2 == 0 else 80, [StemDirection.UP])
        for index in range(8)
    ]
    symbols = [_symbol(1.05 * note.center[0] + 2, note.center[1]) for note in notes]

    assert add_stem_voice_hints(symbols, notes) == 8
    assert [symbol.stem_direction for symbol in symbols] == ["up"] * 8


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
