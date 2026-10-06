"""A coin flip between the key's pitch and an accidental the key does not give.

eerovil/musescore-choir-plugins#274, Lempilintu system 9, in made-up notes: a
low whole note in D major read sharp at 0.58 against the plain note's 0.42.
"""

from homr.score_reconstruction import (
    SymbolChord,
    accidentals_to_the_key_when_unsure,
)
from homr.transformer.vocabulary import EncodedSymbol


def note(pitch: str, lift: str, odds: tuple[tuple[str, float], ...] = ()) -> EncodedSymbol:
    symbol = EncodedSymbol("note_1", pitch, lift, position="upper")
    if odds:
        symbol.confidence = {
            "lift": {
                "value": odds[0][0],
                "alternatives": [{"value": v, "probability": p} for v, p in odds],
            }
        }
    return symbol


def stream(*notes: EncodedSymbol, fifths: int = 2) -> list[SymbolChord]:
    out = [SymbolChord([EncodedSymbol(f"keySignature_{fifths}")])]
    for n in notes:
        out += [SymbolChord([n]), SymbolChord([EncodedSymbol("barline")])]
    return out


def lifts(voice: list[SymbolChord]) -> list[str]:
    return [s.lift for c in voice for s in c.symbols if s.rhythm.startswith("note")]


def test_a_coin_flip_sharp_the_key_does_not_give_goes_back_to_the_key() -> None:
    voice = stream(note("E3", "#", (("#", 0.58), ("_", 0.42))))
    assert lifts(accidentals_to_the_key_when_unsure(voice)) == ["_"]


def test_a_sure_accidental_is_kept() -> None:
    voice = stream(note("E3", "#", (("#", 0.9), ("_", 0.1))))
    assert lifts(accidentals_to_the_key_when_unsure(voice)) == ["#"]


def test_a_coin_flip_between_two_accidentals_is_kept() -> None:
    voice = stream(note("E3", "#", (("#", 0.5), ("b", 0.4))))
    assert lifts(accidentals_to_the_key_when_unsure(voice)) == ["#"]


def test_the_keys_own_sharp_is_never_touched() -> None:
    voice = stream(note("F3", "#", (("#", 0.55), ("N", 0.45))))
    assert lifts(accidentals_to_the_key_when_unsure(voice)) == ["#"]


def test_an_accidental_already_printed_earlier_in_the_bar_holds() -> None:
    sure = note("E3", "#", (("#", 0.95), ("_", 0.05)))
    unsure = note("E3", "#", (("#", 0.55), ("_", 0.45)))
    voice = [
        SymbolChord([EncodedSymbol("keySignature_2")]),
        SymbolChord([sure]),
        SymbolChord([unsure]),
        SymbolChord([EncodedSymbol("barline")]),
    ]
    assert lifts(accidentals_to_the_key_when_unsure(voice)) == ["#", "#"]
