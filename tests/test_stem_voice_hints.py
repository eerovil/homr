import xml.etree.ElementTree as ET
from types import SimpleNamespace

from homr.model import StemDirection
from homr.music_xml_generator import (
    XmlGeneratorArguments,
    generate_xml,
    rebalance_measure_voices,
)
from homr.stem_voice_hints import (
    SHARED,
    add_stem_voice_hints,
    pair_unison_by_attention,
    pair_unison_stems,
)
from homr.transformer.vocabulary import EncodedSymbol, _remove_duplicated_piches


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


def _head(x: float, y: float, position: int, directions: list[StemDirection]) -> SimpleNamespace:
    return SimpleNamespace(center=(x, y), position=position, stem_directions=directions)


def _unison(rhythms: tuple[str, str], xs: tuple[float, float]) -> list[EncodedSymbol]:
    """A treble staff whose one moment holds two decoded notes of one pitch."""
    return [
        EncodedSymbol("clef_G2", position="upper", coordinates=(0.0, 60.0)),
        EncodedSymbol(rhythms[0], pitch="B4", position="upper", coordinates=(xs[0], 55.0)),
        EncodedSymbol(rhythms[1], pitch="B4", position="upper", coordinates=(xs[1], 58.0)),
    ]


def test_a_unison_drawn_as_two_heads_gives_each_note_its_own_stem() -> None:
    """The shape `SHARED` does not carry: two heads side by side, not one.

    Both decoded notes claim the same staff position and their attention points
    are a few pixels apart, so `_at_position` hands both of them the same head
    and the pair then reads as one note written twice.
    """
    heads = [
        _head(164.0, 62.0, 5, [StemDirection.DOWN]),
        _head(172.0, 62.0, 5, [StemDirection.UP]),
    ]
    symbols = _unison(("note_2", "note_8"), (166.0, 170.0))

    assert pair_unison_stems(symbols, heads) == 1
    assert [symbol.stem_direction for symbol in symbols[1:]] == ["down", "up"]


def test_the_note_on_the_left_takes_the_head_on_the_left() -> None:
    """Matched one to one across the staff, which is the trustworthy axis.

    On `hanget-soi`'s bar 3 the left head is the lower voice's open half and the
    right one the upper voice's filled eighth; swapping them puts each duration
    in the other singer's part.
    """
    heads = [
        _head(264.4, 58.0, 5, [StemDirection.DOWN]),
        _head(272.4, 58.0, 5, [StemDirection.UP]),
    ]
    symbols = _unison(("note_8", "note_2"), (275.8, 266.5))

    assert pair_unison_stems(symbols, heads) == 1
    by_rhythm = {symbol.rhythm: symbol.stem_direction for symbol in symbols[1:]}
    assert by_rhythm == {"note_2": "down", "note_8": "up"}


def test_one_head_is_not_a_unison() -> None:
    heads = [_head(164.0, 62.0, 5, [StemDirection.DOWN])]
    symbols = _unison(("note_2", "note_8"), (166.0, 170.0))

    assert pair_unison_stems(symbols, heads) == 0
    assert [symbol.stem_direction for symbol in symbols[1:]] == [None, None]


def test_two_heads_stemmed_the_same_way_are_not_two_voices() -> None:
    """One line cannot hold the same pitch twice at once, so this stays a duplicate."""
    heads = [
        _head(164.0, 62.0, 5, [StemDirection.DOWN]),
        _head(172.0, 62.0, 5, [StemDirection.DOWN]),
    ]
    symbols = _unison(("note_2", "note_8"), (166.0, 170.0))

    assert pair_unison_stems(symbols, heads) == 0


def test_two_heads_at_two_positions_are_left_to_the_rescue() -> None:
    """That is a misread pitch, and `rescue_duplicate_pitches` owns it."""
    heads = [
        _head(164.0, 62.0, 5, [StemDirection.DOWN]),
        _head(172.0, 50.0, 8, [StemDirection.UP]),
    ]
    symbols = _unison(("note_2", "note_8"), (166.0, 170.0))

    assert pair_unison_stems(symbols, heads) == 0


def test_the_rest_of_the_chord_does_not_hide_the_pair() -> None:
    """The other notes of the moment share the column and say nothing about it."""
    heads = [
        _head(164.0, 62.0, 5, [StemDirection.DOWN]),
        _head(172.0, 62.0, 5, [StemDirection.UP]),
        _head(165.0, 78.0, 2, [StemDirection.DOWN]),
    ]
    symbols = _unison(("note_2", "note_8"), (166.0, 170.0))
    symbols.append(
        EncodedSymbol("note_2", pitch="F4", position="upper", coordinates=(167.0, 76.0))
    )

    assert pair_unison_stems(symbols, heads) == 1


def test_a_pitch_written_twice_with_opposite_stems_is_kept() -> None:
    """Two stems means the segmentation found two heads, so it is two voices."""
    chord = [
        EncodedSymbol("note_2", pitch="B4", position="upper", stem_direction="down"),
        EncodedSymbol("note_8", pitch="B4", position="upper", stem_direction="up"),
    ]

    assert len(_remove_duplicated_piches(chord)) == 2


def test_a_pitch_written_twice_the_same_way_is_still_one_note() -> None:
    same = [
        EncodedSymbol("note_2", pitch="B4", position="upper", stem_direction="down"),
        EncodedSymbol("note_8", pitch="B4", position="upper", stem_direction="down"),
    ]
    stemless = [
        EncodedSymbol("note_2", pitch="B4", position="upper"),
        EncodedSymbol("note_8", pitch="B4", position="upper"),
    ]

    assert len(_remove_duplicated_piches(same)) == 1
    assert len(_remove_duplicated_piches(stemless)) == 1


def test_one_stem_beside_none_is_not_two_heads() -> None:
    """Both directions together are the evidence; one of them is not.

    Read symbol by symbol this pair gets two different keys -- `B4 upper down`
    and `B4 upper` -- and neither is dropped, so a second note is written where
    nothing ever saw a second head.
    """
    chord = [
        EncodedSymbol("note_2", pitch="B4", position="upper", stem_direction="down"),
        EncodedSymbol("note_8", pitch="B4", position="upper"),
    ]

    assert len(_remove_duplicated_piches(chord)) == 1


def test_a_stem_beside_a_shared_head_is_not_two_heads() -> None:
    """`SHARED` is doubled in the XML layer; doubling it here as well invents one."""
    chord = [
        EncodedSymbol("note_2", pitch="B4", position="upper", stem_direction="up"),
        EncodedSymbol("note_8", pitch="B4", position="upper", stem_direction=SHARED),
    ]

    assert len(_remove_duplicated_piches(chord)) == 1


def test_a_third_note_of_the_same_pitch_collapses_the_whole_group() -> None:
    """Two heads are two heads; three decoded notes on them are the decoder repeating."""
    chord = [
        EncodedSymbol("note_2", pitch="B4", position="upper", stem_direction="down"),
        EncodedSymbol("note_8", pitch="B4", position="upper", stem_direction="up"),
        EncodedSymbol("note_4", pitch="B4", position="upper"),
    ]

    assert len(_remove_duplicated_piches(chord)) == 1


def test_the_pair_survives_beside_the_rest_of_its_chord() -> None:
    """The other pitches of the moment are their own groups and are untouched."""
    chord = [
        EncodedSymbol("note_2", pitch="B4", position="upper", stem_direction="down"),
        EncodedSymbol("note_8", pitch="B4", position="upper", stem_direction="up"),
        EncodedSymbol("note_2", pitch="G4", position="upper", stem_direction="down"),
    ]

    kept = _remove_duplicated_piches(chord)

    assert [symbol.pitch for symbol in kept] == ["B4", "B4", "G4"]


def test_the_two_notes_of_the_unison_reach_the_file_as_two_voices() -> None:
    """End to end: opposite stems are what the voice rebalancer splits on."""
    voice = [
        EncodedSymbol("clef_G2", position="upper"),
        EncodedSymbol("note_2", pitch="B4", position="upper", stem_direction="down"),
        EncodedSymbol("chord"),
        EncodedSymbol("note_8", pitch="B4", position="upper", stem_direction="up"),
        EncodedSymbol("barline"),
    ]
    xml = generate_xml(XmlGeneratorArguments(), [voice], "")
    notes = xml.findall(".//measure/note")

    assert [note.findtext("pitch/step") for note in notes] == ["B", "B"]
    # The upper voice is written first, so the eighth leads.
    assert [note.findtext("stem") for note in notes] == ["up", "down"]
    assert [note.findtext("type") for note in notes] == ["eighth", "half"]
    assert len({note.findtext("voice") for note in notes}) == 2


def _straddled(
    rhythms: tuple[str, str] = ("note_1", "note_1"),
    ys: tuple[float, float] = (50.0, 74.0),
    stems: tuple[str | None, str | None] = (None, None),
) -> list[EncodedSymbol]:
    """One moment of a treble staff holding a pitch the decoder emitted twice.

    The two attention points sit either side of where the head is drawn, which
    is what an up stem and a down stem look like to the decoder.
    """
    return [
        EncodedSymbol("clef_G2", position="upper", coordinates=(0.0, 60.0)),
        EncodedSymbol(
            rhythms[0], pitch="B4", position="upper",
            coordinates=(166.0, ys[0]), stem_direction=stems[0],
        ),
        EncodedSymbol(
            rhythms[1], pitch="B4", position="upper",
            coordinates=(170.0, ys[1]), stem_direction=stems[1],
        ),
    ]


def test_a_stemless_unison_is_read_off_the_attention() -> None:
    """`sammon-ryosto`'s bar 2: two breve heads side by side, no stem on either.

    `_pair` wants one stem each and there are none to have, so until this the
    pair was read as one note written twice and the lower part lost its note.
    """
    heads = [
        _head(164.0, 62.0, 5, []),
        _head(172.0, 62.0, 5, []),
    ]
    symbols = _straddled()

    assert pair_unison_by_attention(symbols, heads) == 1
    assert [symbol.stem_direction for symbol in symbols[1:]] == ["up", "down"]


def test_a_shared_head_missing_one_of_its_stems_is_still_a_unison() -> None:
    """`system4`'s last sixteenth: one head, and only the up stem hung on it.

    The bar's previous note is the same printed shape and the segmentation gives
    it both stems, so it is written into both voices; this one is not.
    """
    heads = [_head(168.0, 62.0, 5, [StemDirection.UP])]
    symbols = _straddled(rhythms=("note_16", "note_16"))

    assert pair_unison_by_attention(symbols, heads) == 1
    assert [symbol.stem_direction for symbol in symbols[1:]] == ["up", "down"]


def test_a_lone_stemless_head_is_the_decoder_repeating_itself() -> None:
    """`kolme-kakea`'s first bar, and the reason the shapes are named separately.

    A whole note has no stems for the segmentation to have missed, so one
    stemless head cannot be a head drawn with two of them -- and there is no
    second head to say the unison was printed as a pair.
    """
    heads = [_head(168.0, 62.0, 5, [])]
    symbols = _straddled()

    assert pair_unison_by_attention(symbols, heads) == 0
    assert [symbol.stem_direction for symbol in symbols[1:]] == [None, None]


def test_two_attention_points_on_one_side_are_not_two_stems() -> None:
    """A head read twice is read twice from the same place; a pair straddles it."""
    heads = [
        _head(164.0, 62.0, 5, []),
        _head(172.0, 62.0, 5, []),
    ]

    assert pair_unison_by_attention(_straddled(ys=(48.0, 52.0)), heads) == 0
    assert pair_unison_by_attention(_straddled(ys=(72.0, 76.0)), heads) == 0


def test_the_two_notes_of_a_unison_agree_on_their_duration() -> None:
    """One head serves both parts, so the engraver drew one duration for both."""
    heads = [
        _head(164.0, 62.0, 5, []),
        _head(172.0, 62.0, 5, []),
    ]

    assert pair_unison_by_attention(_straddled(rhythms=("note_1", "note_2")), heads) == 0


def test_a_head_carrying_both_stems_is_left_to_the_shared_mark() -> None:
    heads = [_head(168.0, 62.0, 5, [StemDirection.UP, StemDirection.DOWN])]

    assert pair_unison_by_attention(_straddled(), heads) == 0


def test_a_pair_already_told_apart_by_its_stems_is_left_alone() -> None:
    """`pair_unison_stems` ran first and read the picture; this adds nothing."""
    heads = [
        _head(164.0, 62.0, 5, [StemDirection.DOWN]),
        _head(172.0, 62.0, 5, [StemDirection.UP]),
    ]
    symbols = _straddled(stems=("down", "up"))

    assert pair_unison_by_attention(symbols, heads) == 0
    assert [symbol.stem_direction for symbol in symbols[1:]] == ["down", "up"]


def test_the_attention_read_pair_survives_the_duplicate_remover() -> None:
    """Which is the whole point: opposite stems are what keeps both notes."""
    heads = [
        _head(164.0, 62.0, 5, []),
        _head(172.0, 62.0, 5, []),
    ]
    symbols = _straddled()
    pair_unison_by_attention(symbols, heads)

    assert len(_remove_duplicated_piches(symbols[1:])) == 2
