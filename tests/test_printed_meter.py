"""Weighing the printed digits against the bar lengths (eerovil/musescore-choir-plugins#267).

`time_signature_reader` puts the digits each staff prints onto the time
signature token. Neither they nor the bars the notes add up to are trusted alone
where they disagree; these tests pin that rule and the three places it is used:
the signature written into the score, the meter changes inferred from bars, and
the bar length the triplet repairs check each bar against.
"""

from __future__ import annotations

from fractions import Fraction

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml, xml_to_string
from homr.score_reconstruction import (
    SymbolChord,
    find_nominator_per_time_signature,
    infer_meter_changes,
    printed_bar_lengths,
    solve_bar_rhythms,
)
from homr.transformer.vocabulary import EncodedSymbol, remove_duplicated_symbols


def note(rhythm: str, position: str, alternatives: tuple[str, ...] = ()) -> EncodedSymbol:
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
    return EncodedSymbol(rhythm=rhythm, pitch="C4", position=position, confidence=confidence)


def signature(*printed: tuple[int, int], denominator: int = 4) -> SymbolChord:
    symbol = EncodedSymbol(f"timeSignature/{denominator}")
    symbol.printed_meters = printed
    return SymbolChord([symbol])


def barline() -> SymbolChord:
    return SymbolChord([EncodedSymbol("barline")])


def bar(upper: int, lower: int | None = None) -> list[SymbolChord]:
    """A bar of quarter notes, `upper` of them on the upper staff and `lower` below."""
    lower = upper if lower is None else lower
    chords = []
    for index in range(max(upper, lower)):
        symbols = []
        if index < upper:
            symbols.append(note("note_4", "upper"))
        if index < lower:
            symbols.append(note("note_4", "lower"))
        chords.append(SymbolChord(symbols))
    return [*chords, barline()]


def numerators(voice: list[SymbolChord]) -> list[Fraction]:
    return find_nominator_per_time_signature(voice, Fraction(1))


# --- the rule ---


def test_digits_the_notes_agree_with_are_the_meter() -> None:
    voice = [signature((3, 4), (3, 4)), *bar(3), *bar(3)]
    assert printed_bar_lengths(voice) == [Fraction(3, 4)] * 2


def test_digits_every_staff_printed_stand_against_bars_misread_the_same_way() -> None:
    """Legenda system 10's bar 22: every bar of the span misread, both staffs print 3/4."""
    voice = [signature((3, 4), (3, 4)), *bar(4), *bar(4)]
    assert numerators(voice) == [Fraction(3, 4)]


def test_one_staffs_digits_lose_to_bars_every_staff_agrees_on() -> None:
    voice = [signature((3, 4)), *bar(4), *bar(4)]
    assert printed_bar_lengths(voice) == [None, None]
    assert numerators(voice) == [Fraction(1)]


def test_one_staffs_digits_stand_where_the_bars_settle_nothing() -> None:
    voice = [signature((3, 4)), *bar(4, 2)]
    assert numerators(voice) == [Fraction(3, 4)]


def test_staffs_reading_different_digits_say_nothing() -> None:
    voice = [signature((3, 4), (4, 4)), *bar(4), *bar(4)]
    assert printed_bar_lengths(voice) == [None, None]


def test_digits_whose_denominator_the_model_did_not_read_are_not_used() -> None:
    voice = [signature((6, 8), (6, 8), denominator=4), *bar(3), *bar(3)]
    assert printed_bar_lengths(voice) == [None, None]


def test_without_digits_the_bars_decide_as_before() -> None:
    voice = [signature(), *bar(3), *bar(3)]
    assert printed_bar_lengths(voice) == [None, None]
    assert numerators(voice) == [Fraction(3, 4)]


# --- where it is used ---


def test_a_bar_under_printed_digits_is_not_read_as_a_meter_change() -> None:
    voice = [signature((4, 4), (4, 4)), *bar(4), *bar(4), *bar(3)]
    assert len(infer_meter_changes(voice)) == len(voice)


def test_a_change_of_numerator_alone_survives_the_duplicate_cleanup() -> None:
    """3/4 then 4/4 are both `timeSignature/4` to the model."""
    three = EncodedSymbol("timeSignature/4")
    three.printed_meters = ((3, 4),)
    four = EncodedSymbol("timeSignature/4")
    four.printed_meters = ((4, 4),)
    quarter = EncodedSymbol("note_4", "C4", position="upper")
    out = remove_duplicated_symbols([three, quarter, EncodedSymbol("barline"), four, quarter])
    assert [s.rhythm for s in out].count("timeSignature/4") == 2  # noqa: PLR2004


def test_the_score_says_the_meter_the_page_prints() -> None:
    clef = EncodedSymbol("clef_G2", position="upper")
    three = EncodedSymbol("timeSignature/4")
    three.printed_meters = ((3, 4), (3, 4))
    four = EncodedSymbol("timeSignature/4")
    four.printed_meters = ((4, 4), (4, 4))
    quarter = EncodedSymbol("note_4", "C4", position="upper")
    # The first bar misread a beat long, as the decoder does.
    tokens = [clef, three, *[quarter] * 4, EncodedSymbol("barline"), four, *[quarter] * 4]

    xml = xml_to_string(generate_xml(XmlGeneratorArguments(), [tokens], "test"))

    assert xml.count("<time>") == 2  # noqa: PLR2004
    assert "<beats>3</beats><beat-type>4</beat-type>" in xml
    assert "<beats>4</beats><beat-type>4</beat-type>" in xml


def test_a_single_bar_span_is_solved_against_the_printed_length() -> None:
    """Bars alone never settle a span of one bar; printed digits do.

    The upper staff read both its quarters as triplet quarters, which no single
    lengthening repairs. At the printed 3/4 the voice is read back in step with
    the lower staff.
    """
    triplet_quarter = ("note_4",)
    voice = [
        signature((3, 4), (3, 4)),
        SymbolChord([note("note_6", "upper", triplet_quarter), note("note_4", "lower")]),
        SymbolChord([note("note_6", "upper", triplet_quarter), note("note_4", "lower")]),
        SymbolChord([note("note_4", "upper"), note("note_4", "lower")]),
        barline(),
    ]
    unsolved = [signature(), *voice[1:]]

    assert [s.rhythm for c in solve_bar_rhythms(unsolved) for s in c.symbols][1] == "note_6"
    solved = solve_bar_rhythms(voice)
    assert [s.rhythm for c in solved for s in c.symbols if s.position == "upper"] == ["note_4"] * 3
