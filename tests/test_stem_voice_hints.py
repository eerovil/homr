import xml.etree.ElementTree as ET
from types import SimpleNamespace
from typing import cast

from homr.model import Note, StemDirection
from homr.music_xml_generator import (
    XmlGeneratorArguments,
    generate_xml,
    rebalance_measure_voices,
)
from homr.stem_voice_hints import (
    SHARED,
    add_stem_voice_hints,
    drop_unprinted_chord_notes,
    move_notes_to_their_heads,
    pair_unison_by_attention,
    pair_unison_stems,
    rescue_duplicate_pitches,
)
from homr.transformer.vocabulary import EncodedSymbol, _remove_duplicated_piches


def _note(x: float, y: float, directions: list[StemDirection]) -> Note:
    return cast(Note, SimpleNamespace(center=(x, y), stem_directions=directions))


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
    measure = ET.fromstring("""<measure number="1">
             <note><pitch><step>D</step><octave>5</octave></pitch>
               <duration>4</duration><voice>1</voice><stem>down</stem><staff>1</staff></note>
             <note><pitch><step>E</step><octave>4</octave></pitch>
               <duration>4</duration><voice>1</voice><stem>up</stem><staff>1</staff></note>
           </measure>""")

    rebalance_measure_voices(measure)

    assert [note.findtext("voice") for note in measure.findall("note")] == ["1", "1"]


def test_two_voices_sounding_together_are_told_apart_by_their_stems() -> None:
    measure = ET.fromstring("""<measure number="1">
             <note><pitch><step>D</step><octave>5</octave></pitch>
               <duration>4</duration><voice>1</voice><stem>up</stem><staff>1</staff></note>
             <backup><duration>4</duration></backup>
             <note><pitch><step>F</step><octave>4</octave></pitch>
               <duration>4</duration><voice>1</voice><stem>down</stem><staff>1</staff></note>
           </measure>""")

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
    measure = ET.fromstring("""<measure number="1">
             <note stem-shared="yes"><pitch><step>C</step><octave>4</octave></pitch>
               <duration>4</duration><voice>1</voice><staff>1</staff></note>
             <note><pitch><step>E</step><octave>4</octave></pitch>
               <duration>4</duration><voice>1</voice><stem>down</stem><staff>1</staff></note>
           </measure>""")

    rebalance_measure_voices(measure, {1: ("G", 2, 0)})

    assert [note.findtext("voice") for note in measure.findall("note")] == ["1", "2", "2"]


def test_a_head_is_not_doubled_into_a_voice_already_sounding() -> None:
    """Then the second voice is in the bar under its own stem."""
    measure = ET.fromstring("""<measure number="1">
             <note stem-shared="yes"><pitch><step>D</step><octave>5</octave></pitch>
               <duration>4</duration><voice>1</voice><staff>1</staff></note>
             <backup><duration>4</duration></backup>
             <note><pitch><step>F</step><octave>4</octave></pitch>
               <duration>4</duration><voice>1</voice><stem>down</stem><staff>1</staff></note>
           </measure>""")

    rebalance_measure_voices(measure)

    assert [note.findtext("voice") for note in measure.findall("note")] == ["1", "2"]
    assert len(measure.findall("backup")) == 1


def test_only_the_shared_head_of_a_chord_is_doubled() -> None:
    """A chord tone under one stem is one voice's; the shared head is both."""
    measure = ET.fromstring("""<measure number="1">
             <note stem-shared="yes"><pitch><step>C</step><octave>4</octave></pitch>
               <duration>4</duration><voice>1</voice><staff>1</staff></note>
             <note><chord/><pitch><step>E</step><octave>4</octave></pitch>
               <duration>4</duration><voice>1</voice><staff>1</staff></note>
           </measure>""")

    rebalance_measure_voices(measure)
    notes = measure.findall("note")

    assert [note.findtext("pitch/step") for note in notes] == ["C", "E", "C"]
    assert [note.findtext("voice") for note in notes] == ["1", "1", "2"]
    assert notes[2].find("chord") is None


def test_a_grace_note_has_no_duration_to_double_behind() -> None:
    measure = ET.fromstring("""<measure number="1">
             <note stem-shared="yes"><grace/><pitch><step>C</step><octave>4</octave></pitch>
               <voice>1</voice><staff>1</staff></note>
           </measure>""")

    rebalance_measure_voices(measure)

    assert len(measure.findall("note")) == 1
    assert not measure.findall("backup")


def _head(x: float, y: float, position: int, directions: list[StemDirection]) -> Note:
    return cast(Note, SimpleNamespace(center=(x, y), position=position, stem_directions=directions))


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
    symbols.append(EncodedSymbol("note_2", pitch="F4", position="upper", coordinates=(167.0, 76.0)))

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
            rhythms[0],
            pitch="B4",
            position="upper",
            coordinates=(166.0, ys[0]),
            stem_direction=stems[0],
        ),
        EncodedSymbol(
            rhythms[1],
            pitch="B4",
            position="upper",
            coordinates=(170.0, ys[1]),
            stem_direction=stems[1],
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


def test_unison_pairing_requires_coordinates_for_both_notes() -> None:
    first = _symbol(20, 40)
    missing = EncodedSymbol("note_4", pitch="C4", position="upper")
    from homr.stem_voice_hints import _pair, _pair_by_attention

    for left, right in ((first, missing), (missing, first)):
        assert _pair(left, right, [], 1) == 0
        assert _pair_by_attention(left, right, [], 1) == 0
    assert first.stem_direction is None
    assert missing.stem_direction is None


def _second_voice_unison(xs: tuple[float, float]) -> list[EncodedSymbol]:
    """Talviuni's bar 3: model 465 marks both notes of the unison as the second voice.

    The clef is the staff's (`upper`); the notes are its second voice (`upper2`).
    """
    return [
        EncodedSymbol("clef_G2", position="upper", coordinates=(0.0, 60.0)),
        EncodedSymbol("note_2.", pitch="A4", position="upper2", coordinates=(xs[0], 56.0)),
        EncodedSymbol("note_4", pitch="A4", position="upper2", coordinates=(xs[1], 80.0)),
    ]


_TALVIUNI_HEADS = [
    _head(253.0, 71.0, 4, [StemDirection.DOWN]),
    _head(263.0, 71.0, 4, [StemDirection.UP]),
]


def test_a_second_voice_note_reads_against_its_staffs_clef() -> None:
    """Keyed by the raw position, an `upper2` note had no clef and was never paired."""
    symbols = _second_voice_unison((255.0, 249.0))

    assert pair_unison_stems(symbols, _TALVIUNI_HEADS) == 1


def test_the_heads_are_looked_for_across_both_notes_not_around_one() -> None:
    """The right-hand head sits 14px from the left-hand note: out of reach of it alone."""
    symbols = _second_voice_unison((255.0, 249.0))
    pair_unison_stems(symbols, _TALVIUNI_HEADS)

    by_rhythm = {symbol.rhythm: symbol.stem_direction for symbol in symbols[1:]}
    assert by_rhythm == {"note_4": "down", "note_2.": "up"}


def test_a_pair_marked_as_one_voice_is_split_by_its_stems() -> None:
    """One voice cannot sound one pitch twice at once, so the marks follow the stems."""
    symbols = _second_voice_unison((255.0, 249.0))
    pair_unison_stems(symbols, _TALVIUNI_HEADS)

    by_rhythm = {symbol.rhythm: symbol.position for symbol in symbols[1:]}
    assert by_rhythm == {"note_4": "upper2", "note_2.": "upper"}


def test_a_pair_already_marked_as_two_voices_keeps_its_marks() -> None:
    symbols = _second_voice_unison((255.0, 249.0))
    symbols[1].position = "upper"
    pair_unison_stems(symbols, _TALVIUNI_HEADS)

    assert [symbol.position for symbol in symbols[1:]] == ["upper", "upper2"]


def test_the_split_pair_survives_and_lands_in_two_voices() -> None:
    """End to end: the quarter is no longer deleted as a repeat of the dotted half."""
    symbols = _second_voice_unison((255.0, 249.0))
    pair_unison_stems(symbols, _TALVIUNI_HEADS)
    kept = _remove_duplicated_piches(symbols[1:])
    assert len(kept) == 2
    for symbol in kept:
        symbol.coordinates = None
    voice = [
        EncodedSymbol("clef_G2", position="upper"),
        kept[0],
        EncodedSymbol("chord"),
        kept[1],
        EncodedSymbol("note_2", pitch="G4", position="upper2", stem_direction="down"),
        EncodedSymbol("note_4", pitch="A4", position="upper", stem_direction="up"),
        EncodedSymbol("chord"),
        EncodedSymbol("note_4", pitch="F4", position="upper2", stem_direction="down"),
        EncodedSymbol("barline"),
    ]
    notes = generate_xml(XmlGeneratorArguments(), [voice], "").findall(".//measure/note")
    by_voice: dict[str, list[str]] = {}
    for note in notes:
        by_voice.setdefault(note.findtext("voice") or "", []).append(
            f"{note.findtext('pitch/step')}{note.findtext('duration')}"
        )
    assert sorted(by_voice.values()) == sorted([["A3", "A1"], ["A1", "G2", "F1"]])


def test_a_pitch_read_twice_beside_a_free_head_is_rescued_onto_it() -> None:
    """Two heads at two positions and only one read: the other note is the free head."""
    heads = [
        _head(168.0, 56.0, 5, [StemDirection.UP]),
        _head(168.0, 48.0, 7, [StemDirection.DOWN]),
    ]
    symbols = _unison(("note_4", "note_4"), (166.0, 170.0))

    assert rescue_duplicate_pitches(symbols, heads) == 1
    assert sorted(symbol.pitch for symbol in symbols[1:]) == ["B4", "D5"]


def test_a_unison_is_not_moved_onto_the_neighbouring_notes_head() -> None:
    """Sangerhilsen bar 1: a triplet's last E, one head both voices read.

    The triplet's middle note sits close enough to fall in the same column of
    heads, and its head is not free -- the middle note has already read it. Moving
    the unison's second reading onto it turned an E into a C natural.
    """
    heads = [
        _head(156.0, 64.0, 3, [StemDirection.DOWN]),
        _head(168.0, 56.0, 5, [StemDirection.DOWN]),
    ]
    symbols = _unison(("note_12", "note_12"), (166.0, 170.0))
    symbols.insert(
        1, EncodedSymbol("note_12", pitch="G4", position="upper", coordinates=(150.0, 62.0))
    )

    assert rescue_duplicate_pitches(symbols, heads) == 0
    assert [symbol.pitch for symbol in symbols[1:]] == ["G4", "B4", "B4"]


def _chord() -> EncodedSymbol:
    return EncodedSymbol("chord")


def _read(
    pitch: str, position: str, at: tuple[float, float], rhythm: str = "note_12", sure: float = 0.9
) -> EncodedSymbol:
    confidence = {"rhythm": {"probability": sure}, "pitch": {"probability": 1.0}}
    return EncodedSymbol(
        rhythm, pitch=pitch, position=position, coordinates=at, confidence=confidence
    )


def _layout(symbols: list[EncodedSymbol]) -> list[list[tuple[str, str]]]:
    """Moments as (pitch, position) lists, `chord` joining a note to the one before."""
    moments: list[list[tuple[str, str]]] = []
    joined = False
    for symbol in symbols:
        if symbol.rhythm == "chord":
            joined = True
            continue
        if symbol.rhythm.startswith("note"):
            if joined and moments:
                moments[-1].append((symbol.pitch, symbol.position))
            else:
                moments.append([(symbol.pitch, symbol.position)])
        joined = False
    return moments


def _early_triplet_note(upper_already_there: bool = False) -> list[EncodedSymbol]:
    """Sangerhilsen bar 33: the upper voice's middle note read into the first moment."""
    second = [_read("C5", "upper2", (168.0, 56.0))]
    if upper_already_there:
        second += [_chord(), _read("C5", "upper", (168.0, 54.0))]
    return [
        EncodedSymbol("clef_G2", position="upper", coordinates=(0.0, 60.0)),
        _read("A4", "upper2", (150.0, 62.0)),
        _chord(),
        _read("C5", "upper", (166.0, 70.0)),
        *second,
    ]


_TRIPLET_HEADS = [
    _head(150.0, 62.0, 4, [StemDirection.UP, StemDirection.DOWN]),
    _head(167.0, 56.0, 6, [StemDirection.UP, StemDirection.DOWN]),
]


def test_a_note_read_a_moment_early_moves_to_the_head_it_names() -> None:
    symbols = _early_triplet_note()

    assert move_notes_to_their_heads(symbols, _TRIPLET_HEADS) == 1
    assert _layout(symbols) == [[("A4", "upper2")], [("C5", "upper2"), ("C5", "upper")]]


def test_a_note_is_not_moved_into_a_moment_its_voice_already_sounds_in() -> None:
    symbols = _early_triplet_note(upper_already_there=True)

    assert move_notes_to_their_heads(symbols, _TRIPLET_HEADS) == 0


def test_a_note_whose_head_is_in_its_own_column_stays() -> None:
    symbols = _early_triplet_note()
    heads = [*_TRIPLET_HEADS, _head(151.0, 56.0, 6, [StemDirection.UP])]

    assert move_notes_to_their_heads(symbols, heads) == 0


def _phantom_chord_note(sure: float) -> list[EncodedSymbol]:
    """Sangerhilsen bar 15: the lower voice's D read as a D and an E together."""
    return [
        EncodedSymbol("clef_G2", position="upper", coordinates=(0.0, 60.0)),
        _read("G5", "upper", (166.0, 40.0), "note_2"),
        _chord(),
        _read("E5", "upper2", (160.0, 72.0), "note_2.", sure=sure),
        _chord(),
        _read("D5", "upper2", (166.0, 78.0), "note_2."),
    ]


_ONE_HEAD_EACH = [
    _head(166.0, 50.0, 10, [StemDirection.UP]),
    _head(167.0, 64.0, 7, [StemDirection.DOWN]),
]


def test_a_chord_note_the_page_has_no_head_for_is_dropped() -> None:
    symbols = _phantom_chord_note(sure=0.79)

    assert drop_unprinted_chord_notes(symbols, _ONE_HEAD_EACH) == 1
    assert _layout(symbols) == [[("G5", "upper"), ("D5", "upper2")]]


def test_a_chord_note_the_decoder_was_sure_of_is_kept() -> None:
    """A lost head is the segmentation's commonest fault; it never outvotes a sure read."""
    symbols = _phantom_chord_note(sure=0.95)

    assert drop_unprinted_chord_notes(symbols, _ONE_HEAD_EACH) == 0


def test_a_chord_with_a_head_for_every_note_is_kept() -> None:
    symbols = _phantom_chord_note(sure=0.79)
    heads = [*_ONE_HEAD_EACH, _head(160.0, 60.0, 8, [StemDirection.DOWN])]

    assert drop_unprinted_chord_notes(symbols, heads) == 0


def test_a_note_does_not_borrow_the_stem_of_a_head_another_note_reads() -> None:
    """Sangerhilsen bar 47: the lower D has no head found; the E beside it is the upper's."""
    heads = [_head(476.0, 64.0, 8, [StemDirection.UP])]
    symbols = [
        EncodedSymbol("clef_G2", position="upper", coordinates=(0.0, 60.0)),
        _read("E5", "upper", (476.0, 62.0), "note_4"),
        _chord(),
        _read("D5", "upper2", (474.0, 70.0), "note_4"),
    ]

    add_stem_voice_hints(symbols, heads)

    assert symbols[1].stem_direction == "up"
    assert symbols[3].stem_direction is None


def test_a_chord_is_stemmed_by_its_note_furthest_from_the_middle() -> None:
    """D5 over A4 on one down stem is one voice, not a second one.

    A4 alone would stem up; in the chord D5 is further from the middle line
    and decides. Read note by note, the down stem looked like a second voice,
    and every chord went to voice 2 with the rests left behind in voice 1.
    """
    measure = ET.fromstring("""<measure number="1">
             <note><pitch><step>D</step><octave>5</octave></pitch>
               <duration>2</duration><voice>1</voice><stem>down</stem><staff>1</staff></note>
             <note><chord/><pitch><step>A</step><octave>4</octave></pitch>
               <duration>2</duration><voice>1</voice><stem>down</stem><staff>1</staff></note>
             <note><rest/><duration>2</duration><voice>1</voice><staff>1</staff></note>
             <note><pitch><step>C</step><octave>5</octave></pitch>
               <duration>2</duration><voice>1</voice><stem>up</stem><staff>1</staff></note>
             <note><chord/><pitch><step>G</step><octave>4</octave></pitch>
               <duration>2</duration><voice>1</voice><stem>up</stem><staff>1</staff></note>
           </measure>""")

    rebalance_measure_voices(measure, {1: ("G", 2, 0)})

    assert {note.findtext("voice") for note in measure.findall("note")} == {"1"}


def test_a_chord_stemmed_against_its_extreme_note_is_still_a_second_voice() -> None:
    """D5 over B4 stemmed up: nothing in a lone voice draws that."""
    measure = ET.fromstring("""<measure number="1">
             <note><pitch><step>D</step><octave>5</octave></pitch>
               <duration>4</duration><voice>1</voice><stem>up</stem><staff>1</staff></note>
             <note><chord/><pitch><step>B</step><octave>4</octave></pitch>
               <duration>4</duration><voice>1</voice><stem>up</stem><staff>1</staff></note>
             <note><pitch><step>G</step><octave>4</octave></pitch>
               <duration>4</duration><voice>1</voice><stem>down</stem><staff>1</staff></note>
           </measure>""")

    rebalance_measure_voices(measure, {1: ("G", 2, 0)})

    assert [note.findtext("voice") for note in measure.findall("note")] == ["1", "1", "2"]


def test_a_beamed_run_is_stemmed_by_its_note_furthest_from_the_middle() -> None:
    """Kantajani: C5 Bb4 A4 Bb4 under one beam, stemmed down, is one voice.

    A4 alone would stem up. Read note by note its down stem looked like a
    second voice, and the eighths were dealt alternately into two voices.
    """
    eighths = "".join(
        f"""<note><pitch><step>{step}</step><octave>{octave}</octave></pitch>
               <duration>2</duration><voice>1</voice><type>eighth</type>{stem}<staff>1</staff></note>"""
        for step, octave, stem in [
            ("C", 5, "<stem>down</stem>"),
            ("B", 4, ""),
            ("A", 4, "<stem>down</stem>"),
            ("B", 4, ""),
        ]
    )
    measure = ET.fromstring(f'<measure number="1">{eighths}</measure>')

    rebalance_measure_voices(measure, {1: ("G", 2, 0)})

    assert {note.findtext("voice") for note in measure.findall("note")} == {"1"}


def test_a_low_quarter_stemmed_down_is_still_a_second_voice() -> None:
    """A run only joins beamable notes; a lone quarter is judged on its own."""
    measure = ET.fromstring("""<measure number="1">
             <note><pitch><step>C</step><octave>5</octave></pitch>
               <duration>2</duration><voice>1</voice><type>eighth</type><stem>down</stem><staff>1</staff></note>
             <note><pitch><step>E</step><octave>4</octave></pitch>
               <duration>4</duration><voice>1</voice><type>quarter</type><stem>down</stem><staff>1</staff></note>
           </measure>""")

    rebalance_measure_voices(measure, {1: ("G", 2, 0)})

    assert [note.findtext("voice") for note in measure.findall("note")] == ["2", "2"]


def test_a_whole_note_is_never_voiced_by_a_stem_it_does_not_have() -> None:
    """eerovil/musescore-choir-plugins#274: a whole note came out with a down stem
    and went to voice 2 on a one-voice staff (Finlandia, Shakkitarina, Vieläkö)."""
    measure = ET.fromstring("""<measure number="1">
             <note><pitch><step>D</step><octave>4</octave></pitch>
               <duration>8</duration><voice>1</voice><type>whole</type><stem>down</stem><staff>1</staff></note>
           </measure>""")

    rebalance_measure_voices(measure, {1: ("G", 2, 0)})

    assert [note.findtext("voice") for note in measure.findall("note")] == ["1"]


def _two_stem_chords(upper_mark: str, lower_mark: str) -> ET.Element:
    """Three two-stem chords: the upper head stemmed up, the lower one down."""
    notes = ""
    for upper, lower in [(("D", 4), ("B", 3)), (("F", 4), ("D", 4)), (("A", 4), ("F", 4))]:
        for (step, octave), stem, token in (
            (upper, "up", upper_mark),
            (lower, "down", lower_mark),
        ):
            chord = "<chord/>" if stem == "down" else ""
            notes += (
                f'<note voice-token="{token}">{chord}<pitch><step>{step}</step>'
                f"<octave>{octave}</octave></pitch><duration>2</duration><voice>1</voice>"
                f"<type>eighth</type><stem>{stem}</stem><staff>1</staff></note>"
            )
    return ET.fromstring(f'<measure number="1">{notes}</measure>')


def _voices_by_stem(measure: ET.Element) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {"up": set(), "down": set()}
    for note in measure.findall("note"):
        found[note.findtext("stem") or ""].add(note.findtext("voice") or "")
    return found


def test_model_voices_swapped_against_every_stem_are_turned_back() -> None:
    """Illan viimeinen tango s4 (eerovil/musescore-choir-plugins#274): the model
    marked every up-stemmed upper head as the second voice."""
    measure = _two_stem_chords(upper_mark="2", lower_mark="1")

    rebalance_measure_voices(measure, {1: ("G", 2, 0)})

    assert _voices_by_stem(measure) == {"up": {"1"}, "down": {"2"}}


def test_model_voices_that_agree_with_the_stems_are_kept() -> None:
    measure = _two_stem_chords(upper_mark="1", lower_mark="2")

    rebalance_measure_voices(measure, {1: ("G", 2, 0)})

    assert _voices_by_stem(measure) == {"up": {"1"}, "down": {"2"}}


def test_a_partial_disagreement_is_still_the_models() -> None:
    measure = _two_stem_chords(upper_mark="2", lower_mark="1")
    first_up, first_down = measure.findall("note")[:2]
    first_up.set("voice-token", "1")
    first_down.set("voice-token", "2")

    rebalance_measure_voices(measure, {1: ("G", 2, 0)})

    notes = measure.findall("note")
    assert [n.findtext("voice") for n in notes[2:]] == ["2", "1", "2", "1"]
