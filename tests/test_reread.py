"""Reading a fused grand staff again, one staff at a time, when the decoder doubts itself.

The defect this exists for is `sammon-ryosto`'s bar 2: the fused pass wrote one
notehead per staff where the page prints six, at rhythm probabilities of
0.31-0.44 against a median of 0.89 on the same page, and reading each staff on
its own recovered every head. So the tests here are about the three things that
turns into -- when a re-read is worth spending, whether two separate readings can
be put back together as one grand staff, and which of the two readings wins.

Nothing here runs the model. What the model does with a picture is measured on
the fixtures; what is pinned here is the arithmetic around it, which is where a
mistake would be silent.
"""

from fractions import Fraction

from homr import reread
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml, xml_to_string
from homr.transformer.vocabulary import EncodedSymbol


def note(rhythm, pitch="C4", probability=0.9, position="upper"):
    return EncodedSymbol(
        rhythm=rhythm,
        pitch=pitch,
        position=position,
        confidence={"rhythm": {"value": rhythm, "probability": probability}},
    )


def plain(rhythm, position=None):
    """A symbol the decoder recorded no confidence for; `position` defaults to none."""
    if position is None:
        return EncodedSymbol(rhythm=rhythm)
    return EncodedSymbol(rhythm=rhythm, position=position)


def tokens(symbols):
    return [f"{s.rhythm}:{s.pitch}:{s.position}" for s in symbols]


# --- when a re-read is worth spending ---


def test_a_note_read_unsurely_is_what_makes_a_pair_worth_reading_again():
    assert reread.doubtful([note("note_4"), note("note_1", probability=0.31)])
    assert not reread.doubtful([note("note_4"), note("note_1", probability=0.9)])


def test_an_unsure_barline_is_not_the_symptom():
    """kolme-kakea's fused pair holds a barline at 0.46 and is note-perfect.

    The thing being chased is a lost notehead, so counting anything but a note
    or a rest would spend two passes on a page that has nothing wrong with it.
    """
    barline = EncodedSymbol(
        rhythm="barline", confidence={"rhythm": {"value": "barline", "probability": 0.46}}
    )
    assert not reread.doubtful([note("note_4"), barline])


def test_a_reading_with_no_confidences_is_never_doubtful():
    assert not reread.doubtful([plain("note_4"), plain("note_1")])


def test_sureness_averages_the_notes_and_ignores_everything_else():
    symbols = [note("note_4", probability=0.4), note("note_2", probability=0.8), plain("barline")]
    assert reread.surety(symbols) == 0.6000000000000001


def test_sureness_of_a_reading_that_carries_no_confidence_is_unknown():
    assert reread.surety([plain("note_4")]) is None


# --- putting two readings back together ---


def test_the_two_staffs_sound_together_where_they_start_together():
    """sammon-ryosto's bar 2, as the two staffs actually read it.

    A quarter and then a whole chord above, a quarter and then a whole below.
    The point of the splice is that the two quarters land in *one* moment and
    the two wholes in the next -- lay the streams end to end instead and the
    bass sings the bar after the tenors have finished it.
    """
    upper = [note("note_4", "B4"), note("note_1", "D5"), plain("chord"), note("note_1", "C5")]
    lower = [note("note_4", "F3"), note("note_1", "F3")]

    spliced = reread.splice(upper, lower)

    assert tokens(spliced) == [
        "note_4:B4:upper",
        "chord:.:.",
        "note_4:F3:lower",
        "note_1:D5:upper",
        "chord:.:.",
        "note_1:C5:upper",
        "chord:.:.",
        "note_1:F3:lower",
    ]


def test_a_staff_moving_faster_than_the_other_keeps_its_own_onsets():
    upper = [note("note_2", "C5"), note("note_2", "D5")]
    lower = [note("note_4", "C3"), note("note_4", "D3"), note("note_4", "E3"), note("note_4", "F3")]

    spliced = reread.splice(upper, lower)

    assert tokens(spliced) == [
        "note_2:C5:upper",
        "chord:.:.",
        "note_4:C3:lower",
        "note_4:D3:lower",
        "note_2:D5:upper",
        "chord:.:.",
        "note_4:E3:lower",
        "note_4:F3:lower",
    ]


def test_both_clefs_survive_and_the_key_is_declared_once():
    upper = [plain("clef_G2", "upper"), plain("keySignature_-5"), note("note_1", "C5")]
    lower = [plain("clef_F4", "upper"), plain("keySignature_-5"), note("note_1", "C3")]

    spliced = reread.splice(upper, lower)

    assert tokens(spliced) == [
        "clef_G2:.:upper",
        "chord:.:.",
        "clef_F4:.:lower",
        "keySignature_-5:.:.",
        "note_1:C5:upper",
        "chord:.:.",
        "note_1:C3:lower",
    ]


def test_every_bar_keeps_its_barline():
    upper = [note("note_1", "C5"), plain("barline"), note("note_1", "D5")]
    lower = [note("note_1", "C3"), plain("barline"), note("note_1", "D3")]

    spliced = reread.splice(upper, lower)

    assert [s.rhythm for s in spliced].count("barline") == 1
    assert tokens(spliced)[3] == "barline:.:."


def test_two_staffs_that_disagree_about_the_bars_are_not_spliced():
    """There is no way to tell which staff invented or lost the barline.

    A stream spliced across that misalignment puts one staff's music into the
    other staff's bar for the rest of the system, which is worse than the fused
    reading it would replace.
    """
    upper = [note("note_1", "C5"), plain("barline"), note("note_1", "D5")]
    lower = [note("note_1", "C3")]

    assert reread.splice(upper, lower) is None


def test_an_empty_reading_is_not_spliced():
    assert reread.splice([], [note("note_1")]) is None
    assert reread.splice([note("note_1")], []) is None


def test_a_spliced_reading_is_still_one_part_on_two_staffs():
    """The output has to keep looking like the page: a brace, not two parts."""
    upper = [plain("clef_G2", "upper"), note("note_1", "C5")]
    lower = [plain("clef_F4", "upper"), note("note_1", "C3")]

    spliced = reread.splice(upper, lower)
    xml = xml_to_string(generate_xml(XmlGeneratorArguments(), [spliced], "test"))

    assert xml.count("<part ") == 1
    assert "<staves>2</staves>" in xml
    assert "<staff>1</staff>" in xml
    assert "<staff>2</staff>" in xml


# --- which reading wins ---


def test_the_surer_reading_replaces_the_fused_one():
    fused = [note("note_1", probability=0.31)]
    spliced = [note("note_4", probability=0.88)]

    kept, replaced = reread.better_of(fused, spliced)

    assert replaced
    assert kept is spliced


def test_a_re_read_that_is_no_surer_changes_nothing():
    """Ties keep the fused reading, so a false trigger costs time and not correctness."""
    fused = [note("note_1", probability=0.8)]
    spliced = [note("note_4", probability=0.8)]

    kept, replaced = reread.better_of(fused, spliced)

    assert not replaced
    assert kept is fused


def test_a_reading_that_could_not_be_spliced_changes_nothing():
    fused = [note("note_1", probability=0.31)]

    kept, replaced = reread.better_of(fused, None)

    assert not replaced
    assert kept is fused


def test_a_reading_nobody_can_judge_does_not_win():
    fused = [note("note_1", probability=0.31)]
    spliced = [plain("note_4")]

    kept, replaced = reread.better_of(fused, spliced)

    assert not replaced
    assert kept is fused


def test_onsets_are_measured_as_the_shortest_note_of_each_moment():
    """The same rule `SymbolChord.get_duration` uses, which is what makes the two line up."""
    bar = [note("note_1", "C5"), plain("chord"), note("note_4", "E5"), note("note_4", "F5")]

    _, timed = reread._onsets(bar)

    assert [onset for onset, _ in timed] == [Fraction(0), Fraction(1, 4)]
