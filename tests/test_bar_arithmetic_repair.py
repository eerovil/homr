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
bar add up are two readings the arithmetic has nothing to choose between.

Where the arithmetic has run out, the **moments** are asked -- and that is the
other half of what is pinned here. Bar 3 of the same fixture is a sixteenth
short and four alternatives close it; only one of them leaves the two staves
dating every moment they share alike, and it is the one the page prints. It is
asked as a discriminator and never as a precondition: bar 2's staves disagree
under the reading the page prints, and asking them first would take back the
repair the file above exists for.
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


def test_two_alternatives_the_moments_cannot_choose_between_are_refused():
    """Either eighth could be the 16th, and both sit past the other staff's last
    moment, so nothing about when they sound tells them apart."""
    voice = system(
        even_bar(1),
        [
            moment(
                note("note_16", "upper", stem="up"),
                note("note_4", "lower", stem="up"),
            ),
            moment(note("note_8", "upper", stem="up", alternatives=("note_16",))),
            moment(note("note_8", "upper", stem="up", alternatives=("note_16",))),
            barline(),
        ],
        even_bar(1),
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


# --- when more than one alternative fits, the moments are asked ---


def hanget_soi_bar_3():
    """Bar 3 of the same fixture, tokens, stems and alternatives as homr read it.

    The bass sings a quarter and two beamed eighths over a dotted eighth, a 16th
    and a quarter, and the first of the two eighths came back a 16th -- so that
    staff measures a 16th short of the two quarters every other bar says a bar
    is. Four of the decoder's own alternatives close it. Two are readings of the
    other voice and one is not, and the two that are a reading of this one are
    the page's (`note_8` on the first eighth) and a dotted eighth on the second.
    """
    return [
        moment(
            note("note_8.", "lower", stem="down", alternatives=("note_1", "note_2"), pitch="C3"),
            note("note_8", "upper", stem="up", pitch="E5"),
            note(
                "note_4",
                "lower",
                stem="up",
                alternatives=("note_8", "note_2", "note_16", "note_8.", "note_4."),
                pitch="G3",
            ),
            note("note_2", "upper", stem="down", pitch="E5"),
            note("note_2", "upper", stem="down", pitch="C5"),
        ),
        moment(note("note_16", "upper", stem="up", pitch="E5")),
        moment(
            note("note_16", "upper", stem="up", pitch="E5"),
            note(
                "note_16", "lower", stem="down",
                alternatives=("note_4", "note_8..", "note_32"), pitch="B2",
            ),
        ),
        moment(
            note("note_8", "upper", stem="up", pitch="A5"),
            note("note_4", "lower", stem="down", alternatives=("note_8",), pitch="A2"),
            note(
                "note_16", "lower", stem="up",
                alternatives=("note_8", "note_16.", "note_4"), pitch="A3",
            ),
        ),
        moment(
            note(
                "note_8", "lower", stem="up",
                alternatives=("note_16", "note_8.", "note_32", "note_4", "note_8.."),
                pitch="G3",
            ),
            note("note_16", "upper", stem="up", pitch="E5"),
        ),
        moment(note("note_16", "upper", stem="up", pitch="E5")),
        barline(),
    ]


def test_the_moments_pick_the_eighth_the_page_prints():
    """Four alternatives close the bar; only one leaves the two staves in step."""
    voice = system(even_bar(), hanget_soi_bar_3(), even_bar())
    assert rhythms(repair_bar_arithmetic(voice), "lower")[2:8] == [
        "note_8.", "note_4", "note_16", "note_4", "note_8", "note_8",
    ]


def test_the_note_the_page_prints_lands_on_the_beat_the_other_staff_dates_it_at():
    """The treble adds up and puts that moment at beat 1.5; so must the bass."""
    repaired = repair_bar_arithmetic(system(even_bar(), hanget_soi_bar_3(), even_bar()))
    assert _lower_beats(repaired[3:9]) == [Fraction(0), Fraction(3, 16), Fraction(4, 16),
                                           Fraction(6, 16)]


def _lower_beats(bar):
    """Where each of the lower staff's moments starts, on its own cursor."""
    beats, cursor = [], Fraction(0)
    for chord in bar:
        durations = [
            symbol.get_duration().fraction
            for symbol in chord.symbols
            if symbol.position == "lower" and symbol.rhythm.startswith(("note", "rest"))
        ]
        if durations:
            beats.append(cursor)
            cursor += min(durations)
    return beats


def test_the_moments_are_only_asked_when_the_arithmetic_has_already_refused():
    """Bar 2's staves disagree under the reading the page prints, and it still repairs.

    A moment is only approximately a column of the page: here a 16th rest
    printed at beat 1.5 shares one with a bass quarter printed at beat 1. Asking
    the moments before the arithmetic has refused would take this repair back.
    """
    voice = system(even_bar(), hanget_soi_bar_2(), even_bar())
    assert rhythms(repair_bar_arithmetic(voice))[2] == "note_4."


def test_a_candidate_that_moves_a_shared_moment_out_of_step_is_not_taken():
    """Shortening the second note would leave the staves apart at the second moment."""
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
    assert rhythms(repair_bar_arithmetic(voice)) == ["note_4"] * 3 + [
        "note_4", "note_2"] + ["note_4"] * 3
