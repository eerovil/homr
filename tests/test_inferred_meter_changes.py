"""Writing a time signature the model never read, where the page plainly changed meter.

The vocabulary holds only denominators, so a change of numerator alone -- 3/4 to
5/4, which is what a page does when it adds a beat -- cannot be emitted however
well the bar is read. On `sammon-ryosto` both bars of the opening span are
measured exactly right, 3 quarters and 5, and both came out declared the same,
because one number was being fitted to a span holding two.

Two rules together do it, and each is here because the other is not enough. A
span's numerator is now what its bars *consecutively* agree on rather than their
median, so the signature describes the bar it opens instead of an average of
bars that disagree; and a later bar whose own length contradicts that, with
every staff agreeing about it, has the signature written at it.

The guards are most of the point, and an anacrusis is why: a pickup is exactly a
first bar shorter than its own meter, so a rule that reads a short bar as a
meter change reads every pickup wrong. Those cases are pinned here beside the
one this exists to fix.
"""

from fractions import Fraction

from homr.music_xml_generator import (
    XmlGeneratorArguments,
    find_nominator_per_time_signature,
    generate_xml,
    infer_meter_changes,
    prevailing_length,
    xml_to_string,
)
from homr.transformer.vocabulary import EncodedSymbol


def chord(rhythm, *positions):
    from homr.music_xml_generator import SymbolChord

    return SymbolChord(
        [EncodedSymbol(rhythm=rhythm, position=p) for p in (positions or ("upper",))]
    )


def bar(quarters, staves=("upper", "lower")):
    """One bar of that many quarter notes on each staff, then its barline."""
    return [chord("note-C4_quarter", *staves) for _ in range(quarters)] + [chord("barline")]


def signatures_of(voice):
    return [
        int(n * int(c.symbols[0].rhythm.split("/")[1]))
        for n, c in zip(
            find_nominator_per_time_signature(voice, Fraction(1)),
            [c for c in voice if c.symbols[0].rhythm.startswith("timeSignature")],
        )
    ]


# --- what a span's numerator is ---


def test_the_length_the_most_consecutive_bars_agree_on():
    assert prevailing_length([Fraction(1)] * 3 + [Fraction(3, 4)]) == Fraction(1)


def test_an_odd_bar_at_the_front_does_not_become_the_meter():
    """An anacrusis: one short bar, then the meter."""
    lengths = [Fraction(1, 4)] + [Fraction(1)] * 3
    assert prevailing_length(lengths) == Fraction(1)


def test_a_tie_takes_the_earliest_run():
    """`3/4, 5/4` has no majority; the printed signature opens the first of them.

    A median says a whole note, which is neither bar. The other bar is left to
    `infer_meter_changes` rather than averaged away.
    """
    assert prevailing_length([Fraction(3, 4), Fraction(5, 4)]) == Fraction(3, 4)


def test_one_bar_is_its_own_meter():
    assert prevailing_length([Fraction(5, 4)]) == Fraction(5, 4)


# --- where a signature gets written that nobody read ---


def test_a_bar_that_changed_meter_and_read_no_signature_gets_one():
    """sammon-ryosto's opening span: 3/4 printed, then 5/4 printed and not read."""
    voice = [chord("timeSignature/4")] + bar(3) + bar(5)

    out = infer_meter_changes(voice)

    assert len(out) == len(voice) + 1
    assert signatures_of(out) == [3, 5]


def test_an_anacrusis_is_not_a_meter_change():
    """The first bar of a span is where a printed signature stands, short or not.

    Read the pickup as the meter and the meter becomes a change: the score would
    open in 1/4 and change to 4/4 in bar 2, which is not what any page prints.
    """
    voice = [chord("timeSignature/4")] + bar(1) + bar(4) + bar(4)

    assert infer_meter_changes(voice) is voice
    assert signatures_of(voice) == [4]


def test_a_bar_the_staffs_disagree_about_is_left_alone():
    """A bar homr lost a note from is short in one staff and right in the other.

    That is the shape of a misread bar, not of a meter change, and it is the
    shape `sammon-ryosto`'s bar 3 still has.
    """
    ragged = [chord("note-C4_quarter", "upper", "lower")] * 4 + [chord("note-C4_quarter", "upper")]
    voice = [chord("timeSignature/4")] + bar(4) + [*ragged, chord("barline")]

    assert infer_meter_changes(voice) is voice


def test_a_single_staff_system_never_gets_one():
    """Nothing to agree with, so nothing corroborates the length."""
    voice = [chord("timeSignature/4")] + bar(4, ("upper",)) + bar(5, ("upper",))

    assert infer_meter_changes(voice) is voice


def test_the_first_bar_after_a_read_signature_is_never_given_one():
    voice = [chord("timeSignature/4")] + bar(4) + bar(4) + [chord("timeSignature/2")] + bar(3)

    assert infer_meter_changes(voice) is voice


def test_the_denominator_in_force_is_carried():
    """A bar length settles a numerator; a denominator is a spelling it cannot pick."""
    voice = [chord("timeSignature/2")] + bar(10) + bar(6)

    out = infer_meter_changes(voice)

    written = [c.symbols[0].rhythm for c in out if c.symbols[0].rhythm.startswith("timeSignature")]
    assert written == ["timeSignature/2", "timeSignature/2"]
    assert signatures_of(out) == [5, 3]


def test_a_voice_with_nothing_to_change_comes_back_unchanged():
    voice = [chord("timeSignature/4")] + bar(4) + bar(4) + bar(4)

    assert infer_meter_changes(voice) is voice


def test_a_voice_with_no_signature_at_all_is_left_alone():
    """There is no meter in force to contradict, and none to carry a denominator."""
    voice = bar(4) + bar(5)

    assert infer_meter_changes(voice) is voice


# --- and what the score then says ---


def tokens(voice):
    """The chords back as one token stream, `chord` separators and all.

    `generate_xml` regroups the stream itself, and without the separators each
    staff's note becomes a moment of its own -- a bar of three then reads as six.
    """
    out = []
    for group in voice:
        for index, symbol in enumerate(group.symbols):
            if index:
                out.append(EncodedSymbol("chord"))
            out.append(symbol)
    return out


def test_the_score_declares_both_meters_the_page_prints():
    voice = [chord("clef_G2"), chord("timeSignature/4")] + bar(3) + bar(5)

    xml = xml_to_string(generate_xml(XmlGeneratorArguments(), [tokens(voice)], "test"))

    assert "<beats>3</beats><beat-type>4</beat-type>" in xml
    assert "<beats>5</beats><beat-type>4</beat-type>" in xml
