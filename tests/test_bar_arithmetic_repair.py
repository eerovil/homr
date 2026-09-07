"""Taking the decoder's second answer where its first one does not fit the bar.

A misread note value is not silent -- the bar it is in stops adding up, and the
notes after it are written past the end of the bar and lost. On `hanget-soi` bar
2 the page prints a dotted quarter, a 16th rest and a 16th against a half chord,
and the model read the dotted quarter as a half: that staff measures two and a
half quarters of a bar every other piece of evidence says is two, and the 16th
the page prints at beat 1.75 never reaches the file.

The value the page prints is not out of reach -- the decoder scores every rhythm
token at every step, and `note_4.` is sitting fourth at 0.9%. What was missing is
a reason to prefer it, and the arithmetic is one.

Most of what is pinned here is the refusals, because a rule that repairs an
ambiguous bar is guessing at the music: one staff is never enough, a bar that
opens a span is where an anacrusis is, a note sharing a stem with another is a
chord and cannot be shortened on its own, and two alternatives that both make the
bar add up are two readings with nothing to choose between them.
"""

from fractions import Fraction

from homr.music_xml_generator import SymbolChord, repair_bar_arithmetic
from homr.transformer.vocabulary import EncodedSymbol


def note(rhythm, position="upper", stem=None, alternatives=(), pitch="C4"):
    """One decoded note, carrying the readings the decoder ranked under it."""
    confidence = None
    if alternatives:
        confidence = {
            "rhythm": {
                "value": rhythm,
                "alternatives": [{"value": value, "probability": 0.01} for value in alternatives],
            }
        }
    return EncodedSymbol(
        rhythm=rhythm,
        pitch=pitch,
        position=position,
        confidence=confidence,
        stem_direction=stem,
    )


def moment(*symbols):
    return SymbolChord(list(symbols))


def barline():
    return SymbolChord([EncodedSymbol("barline")])


def even_bar(quarters=2, positions=("upper", "lower")):
    """A bar of plain quarter notes on each staff, and its barline."""
    return [
        moment(*[note("note_4", position, stem="up") for position in positions])
        for _ in range(quarters)
    ] + [barline()]


def rhythms(voice, position="upper"):
    return [
        symbol.rhythm
        for chord in voice
        for symbol in chord.symbols
        if symbol.position == position and symbol.rhythm.startswith(("note", "rest"))
    ]


def hanget_soi_bar_2(alternatives=("note_4.", "note_4", "note_2.")):
    """The shape of the bar this exists for, tokens and stems as homr read it.

    Voice 1 is a dotted quarter, a 16th rest and a 16th; voice 2 is a half chord
    beside it, drawn with the other stem. The dotted quarter came back a half.
    """
    return [
        moment(
            note("note_2", "upper", stem="up", alternatives=alternatives, pitch="D5"),
            note("note_2", "upper", stem="down", alternatives=alternatives, pitch="D5"),
            note("note_2", "upper", stem="down", alternatives=alternatives, pitch="B4"),
            note("note_4", "lower", stem="up", pitch="D3"),
            note("note_2", "lower", stem="down", pitch="G2"),
        ),
        moment(
            note("rest_16", "upper", alternatives=("rest_8", "rest_32")),
            note("note_4", "lower", stem="up", pitch="G3"),
        ),
        moment(note("note_16", "upper", stem="up", alternatives=("note_8",), pitch="D5")),
        barline(),
    ]


def system(*bars):
    """A system of bars, the first two of them read cleanly so a bar has a length."""
    voice = []
    for bar in bars:
        voice.extend(bar)
    return voice


# --- the bar this exists for ---


def test_the_half_the_page_prints_as_a_dotted_quarter_is_corrected():
    voice = system(even_bar(), hanget_soi_bar_2(), even_bar())
    repaired = repair_bar_arithmetic(voice)
    assert rhythms(repaired)[2:5] == ["note_4.", "note_2", "note_2"]


def test_the_corrected_bar_then_adds_up():
    voice = system(even_bar(), hanget_soi_bar_2(), even_bar())
    repaired = repair_bar_arithmetic(voice)
    upper = sum(
        min(
            symbol.get_duration().fraction
            for symbol in chord.symbols
            if symbol.position == "upper" and symbol.rhythm.startswith(("note", "rest"))
        )
        for chord in repaired[3:6]
    )
    assert upper == Fraction(1, 2)


def test_a_bar_that_already_adds_up_is_left_alone():
    voice = system(even_bar(), even_bar(), even_bar())
    assert repair_bar_arithmetic(voice) is voice


# --- the refusals ---


def test_a_chord_head_is_never_shortened_on_its_own():
    """The two stems are the same, so the two notes are one chord the page drew."""
    voice = system(even_bar(), hanget_soi_bar_2(), even_bar())
    voice[3].symbols[0].stem_direction = "down"
    assert repair_bar_arithmetic(voice) is voice


def test_a_note_with_no_stem_beside_another_is_never_shortened():
    voice = system(even_bar(), hanget_soi_bar_2(), even_bar())
    voice[3].symbols[0].stem_direction = None
    assert repair_bar_arithmetic(voice) is voice


def test_two_alternatives_that_both_fit_are_refused():
    """Either half could be the quarter: two readings of the bar, and no choice."""
    voice = system(
        even_bar(3),
        [
            moment(
                note("note_2", "upper", stem="up", alternatives=("note_4",)),
                note("note_4", "lower", stem="up"),
            ),
            moment(
                note("note_2", "upper", stem="up", alternatives=("note_4",)),
                note("note_2", "lower", stem="up"),
            ),
            barline(),
        ],
        even_bar(3),
    )
    assert repair_bar_arithmetic(voice) is voice


def test_the_value_the_page_prints_has_to_be_one_the_decoder_offered():
    voice = system(even_bar(), hanget_soi_bar_2(alternatives=("note_4", "note_1")), even_bar())
    assert repair_bar_arithmetic(voice) is voice


def test_a_rest_is_never_read_as_a_note():
    """Only the rest could fix this bar, and only by becoming something it is not."""
    voice = system(even_bar(), hanget_soi_bar_2(alternatives=()), even_bar())
    voice[4].symbols[0].confidence["rhythm"]["alternatives"] = [
        {"value": "note_2", "probability": 0.01}
    ]
    assert repair_bar_arithmetic(voice) is voice


def test_one_staff_is_never_enough():
    """A system printing one staff has nothing in the bar to disagree with it."""
    voice = system(
        even_bar(positions=("upper",)),
        [
            moment(note("note_2", "upper", stem="up", alternatives=("note_4.",))),
            moment(note("rest_16", "upper")),
            moment(note("note_16", "upper", stem="up")),
            barline(),
        ],
        even_bar(positions=("upper",)),
    )
    assert repair_bar_arithmetic(voice) is voice


def test_the_bar_that_opens_a_system_is_never_repaired():
    """That is where an anacrusis is, and a pickup is a bar short of its meter."""
    voice = system(hanget_soi_bar_2(), even_bar(), even_bar())
    assert repair_bar_arithmetic(voice) is voice


def test_one_clean_bar_is_not_a_meter():
    voice = system(even_bar(), hanget_soi_bar_2())
    assert repair_bar_arithmetic(voice) is voice


def test_a_bar_holding_a_tuplet_is_left_alone():
    voice = system(even_bar(), hanget_soi_bar_2(), even_bar())
    voice[5].symbols[0].rhythm = "note_12"
    assert repair_bar_arithmetic(voice) is voice


def test_a_staff_two_notes_adrift_is_left_alone_when_nothing_single_fixes_it():
    voice = system(even_bar(), hanget_soi_bar_2(alternatives=("note_4",)), even_bar())
    assert repair_bar_arithmetic(voice) is voice


def test_the_repair_is_measured_against_the_meter_the_bars_agree_on():
    """Three-quarter bars around it, so the target is three and not two."""
    voice = system(
        even_bar(3),
        [
            moment(
                note("note_2", "upper", stem="up", alternatives=("note_2.", "note_4")),
                note("note_2.", "upper", stem="down"),
                note("note_2.", "lower", stem="up"),
            ),
            barline(),
        ],
        even_bar(3),
    )
    assert rhythms(repair_bar_arithmetic(voice))[3] == "note_2."
