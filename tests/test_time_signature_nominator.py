"""The numerator of a time signature, which is inferred rather than read.

The model's vocabulary holds only denominators, so the numerator comes from how
long the bars actually are. Taking one median for a whole voice made a *changing*
meter impossible to express: every signature got the same numerator and differed
only in its denominator, so a part going 3/4, 5/4, 5/2 could have at most one of
them right. Measured on a real system of Sammon ryösto it had none.
"""

from fractions import Fraction

from homr.music_xml_generator import (
    ConversionState,
    find_nominator_per_time_signature,
)
from homr.transformer.vocabulary import EncodedSymbol


def chord(rhythm, position="upper"):
    from homr.music_xml_generator import SymbolChord

    return SymbolChord([EncodedSymbol(rhythm=rhythm, position=position)])


def bar(quarters):
    """One bar of that many quarter notes, then its barline."""
    return [chord("note-C4_quarter") for _ in range(quarters)] + [chord("barline")]


def test_each_span_between_signatures_gets_its_own_numerator():
    """Sammon ryösto's shape: 3/4, then 5/4, then 5/2."""
    voice = ([chord("timeSignature/4")] + bar(3)
             + [chord("timeSignature/4")] + bar(5)
             + [chord("timeSignature/2")] + bar(10) + bar(10))
    found = find_nominator_per_time_signature(voice, Fraction(1))
    assert found == [Fraction(3, 4), Fraction(5, 4), Fraction(10, 4)]
    # ...which is what `build_time_signature` turns into beats.
    assert [int(n * d) for n, d in zip(found, (4, 4, 2))] == [3, 5, 5]


def test_one_median_for_the_whole_voice_could_not_do_that():
    """The behaviour this replaces, kept as the reason it was replaced.

    Every bar length in one pot, one median, applied to every signature: the
    three spans above are 3, 5 and 10 quarters, whose median is 5 quarters, so
    all three signatures come out of it and two of them are wrong.
    """
    voice = ([chord("timeSignature/4")] + bar(3)
             + [chord("timeSignature/4")] + bar(5)
             + [chord("timeSignature/2")] + bar(10) + bar(10))
    from homr.music_xml_generator import find_division_and_time_signature_nominator

    _, single = find_division_and_time_signature_nominator(voice)
    assert [int(single * d) for d in (4, 4, 2)] != [3, 5, 5]


def test_an_unchanging_meter_is_unaffected():
    """The ordinary case has to come out exactly as it did before."""
    voice = [chord("timeSignature/4")] + bar(4) + bar(4) + bar(4)
    assert find_nominator_per_time_signature(voice, Fraction(1)) == [Fraction(1)]


def test_a_span_with_no_complete_bar_falls_back():
    """Rather than inventing a numerator out of nothing."""
    voice = [chord("timeSignature/4"), chord("timeSignature/2")] + bar(4)
    found = find_nominator_per_time_signature(voice, Fraction(7, 8))
    assert found == [Fraction(7, 8), Fraction(1)]


def test_a_voice_with_no_signature_at_all_asks_for_nothing():
    assert find_nominator_per_time_signature(bar(4) + bar(4), Fraction(1)) == []


def test_the_state_hands_them_out_in_order_then_falls_back():
    """A signature past the end of the list still gets the voice-wide figure."""
    state = ConversionState(8, Fraction(1), [Fraction(3, 4), Fraction(5, 4)])
    assert [state.next_nominator() for _ in range(3)] == [
        Fraction(3, 4), Fraction(5, 4), Fraction(1)]
    # ...and the rest of the state survived being given a third argument.
    assert state.tremolo_state == "stop"
    assert state.volta_number == 1
    assert state.last_volta_measure == -10
