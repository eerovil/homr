"""A note the decoder read twice off one printed head is taken out, and only then.

eerovil/musescore-choir-plugins#274, Kantajani s8: the first sopranos' sixth
eighth was decoded twice, 7px apart, and the second sopranos gained a note
standing to the right of the one decoded after it; each bar came out an eighth
over its printed 5/4.
"""

from fractions import Fraction

from homr.score_reconstruction import SymbolChord, drop_double_reads
from homr.transformer.vocabulary import EncodedSymbol


def note(rhythm: str, x: float, probability: float = 0.9, pitch: str = "A4") -> EncodedSymbol:
    symbol = EncodedSymbol(
        rhythm=rhythm,
        pitch=pitch,
        position="upper",
        confidence={"rhythm": {"value": rhythm, "probability": probability}},
    )
    symbol.image_coordinates = (x, 100.0)
    symbol.stem_direction = "up"
    return symbol


def bar(*notes: EncodedSymbol) -> list[SymbolChord]:
    return [SymbolChord([n]) for n in notes] + [SymbolChord([EncodedSymbol("barline")])]


def pitches(voice: list[SymbolChord]) -> list[str]:
    return [s.pitch for chord in voice for s in chord.symbols if s.rhythm.startswith("note")]


def eighths_then_quarters(extra: EncodedSymbol | None, at: int) -> list[EncodedSymbol]:
    notes = [note("note_8", 320 + 48 * i, pitch=f"A{4}") for i in range(6)]
    notes += [note("note_4", 600), note("note_4", 670)]
    if extra is not None:
        notes.insert(at, extra)
    return notes


def test_a_head_read_twice_in_place_is_taken_out_once() -> None:
    notes = eighths_then_quarters(note("note_8", 320 + 48 * 5 + 7, 0.7, "B4"), 6)
    voice = bar(*notes)
    fixed = drop_double_reads(voice, None, [Fraction(5, 4)])
    assert len(pitches(fixed)) == 8
    assert "B4" not in pitches(fixed)


def test_a_note_out_of_order_on_the_page_is_taken_out() -> None:
    notes = eighths_then_quarters(note("note_8", 320 + 48 * 5 + 30, 0.6, "G4"), 5)
    fixed = drop_double_reads(bar(*notes), None, [Fraction(5, 4)])
    assert "G4" not in pitches(fixed)


def test_a_bar_that_is_not_over_is_left_alone() -> None:
    notes = eighths_then_quarters(note("note_8", 320 + 48 * 5 + 7, 0.7, "B4"), 6)
    notes.pop()  # the bar now measures exactly 5/4 with the double read in it
    fixed = drop_double_reads(bar(*notes), None, [Fraction(5, 4)])
    assert "B4" in pitches(fixed)


def test_without_a_target_nothing_is_taken_out() -> None:
    notes = eighths_then_quarters(note("note_8", 320 + 48 * 5 + 7, 0.7, "B4"), 6)
    fixed = drop_double_reads(bar(*notes), None, None)
    assert "B4" in pitches(fixed)


def test_two_candidates_in_one_bar_are_both_left() -> None:
    notes = eighths_then_quarters(note("note_8", 320 + 48 * 5 + 7, 0.7, "B4"), 6)
    notes.insert(2, note("note_8", 320 + 48 + 5, 0.7, "C5"))
    fixed = drop_double_reads(bar(*notes), None, [Fraction(5, 4)])
    assert "B4" in pitches(fixed) and "C5" in pitches(fixed)


def test_two_notes_far_apart_in_pitch_are_never_one_head() -> None:
    """Finlandia s8: the decoder put a bar's last eighth, A3, 2px from the E3 before
    it; the bar was over for another reason (a quarter that is an eighth)."""
    notes = eighths_then_quarters(note("note_8", 320 + 48 * 5 + 2, 0.7, "D4"), 6)
    fixed = drop_double_reads(bar(*notes), None, [Fraction(5, 4)])
    assert "D4" in pitches(fixed)
