"""Reading a note as a tuplet's when its own voice would otherwise overlap itself.

Sangerhilsen bars 21 and 37 (eerovil/musescore-choir-plugins#240): both voices of
each staff sing a triplet of eighths in unison. The decoder read three heads of
the first moment as triplet eighths and the fourth as a plain eighth, so that
voice was still sounding when its second triplet note began, the writer's clock
started the third note late, and the bar came out 9/8 under 4/4 in every part.

Most of what is pinned here is the refusals: the decoder has to have offered the
reading, the moment has to hold that value already, and only a tuplet's twin is
ever tried -- a voice overlapping itself for any other reason is left alone.
"""

from __future__ import annotations

from fractions import Fraction

from homr.score_reconstruction import (
    ReconstructionChange,
    SymbolChord,
    _onsets,
    repair_tuplet_overlaps,
)
from homr.transformer.vocabulary import EncodedSymbol


def note(
    rhythm: str, position: str, pitch: str = "C4", alternatives: tuple[str, ...] = ()
) -> EncodedSymbol:
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
    return EncodedSymbol(rhythm=rhythm, pitch=pitch, position=position, confidence=confidence)


def moment(*symbols: EncodedSymbol) -> SymbolChord:
    return SymbolChord(list(symbols))


def barline() -> SymbolChord:
    return SymbolChord([EncodedSymbol("barline")])


def triplet_bar(misread: str = "note_8", alternatives: tuple[str, ...] = ("note_12",)) -> list:
    """Beat four of Sangerhilsen bar 37 as homr read it, one head a plain eighth."""
    return [
        moment(
            note("note_4.", "upper", "C5"),
            note("note_4.", "lower", "C4"),
        ),
        moment(note("note_8", "upper", "C5"), note("note_8", "lower", "C4")),
        moment(note("note_4", "upper", "C5"), note("note_4", "lower", "C4")),
        moment(
            note("note_12", "upper", "C4"),
            note(misread, "lower", "C3", alternatives),
        ),
        moment(note("note_12", "upper", "E4"), note("note_12", "lower", "E3")),
        moment(note("note_12", "upper", "G4"), note("note_12", "lower", "G3")),
        barline(),
    ]


def rhythms(voice: list[SymbolChord], position: str) -> list[str]:
    return [s.rhythm for chord in voice for s in chord.symbols if s.position == position]


def test_the_plain_eighth_is_read_as_the_triplet_eighth_beside_it() -> None:
    changes: list[ReconstructionChange] = []
    repaired = repair_tuplet_overlaps(triplet_bar(), changes)
    assert rhythms(repaired, "lower")[3] == "note_12"
    assert [(c.kind, c.before, c.after, c.bar) for c in changes] == [
        ("tuplet_repair", "note_8", "note_12", 1)
    ]


def test_the_third_triplet_note_then_starts_on_time() -> None:
    before = triplet_bar()
    after = repair_tuplet_overlaps(before)
    # The third triplet note starts on time once nothing overlaps it.
    assert _onsets(before, (0, 6))[5] == Fraction(7, 8)
    assert _onsets(after, (0, 6))[5] == Fraction(11, 12)


def test_a_reading_the_decoder_did_not_offer_is_never_taken() -> None:
    voice = triplet_bar(alternatives=("note_4",))
    assert repair_tuplet_overlaps(voice) is voice


def test_only_the_tuplet_twin_is_tried() -> None:
    """A sixteenth would end the overlap too early; a tuplet's twin or nothing."""
    voice = triplet_bar(misread="note_8", alternatives=("note_16",))
    assert repair_tuplet_overlaps(voice) is voice


def test_a_value_nothing_beside_it_holds_is_never_taken() -> None:
    voice = triplet_bar()
    voice[3].symbols[0] = note("note_8", "upper", "C4")
    voice[4].symbols[0] = note("note_8", "upper", "E4")
    voice[5].symbols[0] = note("note_8", "upper", "G4")
    for index in (4, 5):
        voice[index].symbols[1] = note("note_8", "lower", voice[index].symbols[1].pitch)
    # Every other head is a plain eighth now: the lower voice does not overlap at all.
    assert repair_tuplet_overlaps(voice) is voice


def test_a_voice_that_does_not_overlap_itself_is_left_alone() -> None:
    voice = triplet_bar(misread="note_12")
    assert repair_tuplet_overlaps(voice) is voice
