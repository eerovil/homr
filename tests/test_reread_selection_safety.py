"""A re-read may recover music, but confidence must never buy omissions."""

from __future__ import annotations

from homr import reread
from homr.transformer.vocabulary import EncodedSymbol


def event(
    rhythm: str,
    pitch: str = "C4",
    probability: float = 0.9,
    position: str = "upper",
) -> EncodedSymbol:
    return EncodedSymbol(
        rhythm=rhythm,
        pitch=pitch,
        position=position,
        confidence={"rhythm": {"value": rhythm, "probability": probability}},
    )


def barline() -> EncodedSymbol:
    return EncodedSymbol("barline")


def test_higher_confidence_cannot_pay_for_a_dropped_note() -> None:
    fused = [event("note_4", "C4", 0.3), event("note_4", "D4", 0.3)]
    reread_candidate = [event("note_2", "C4", 0.99)]

    kept, replaced = reread.better_of(fused, reread_candidate)

    assert not replaced
    assert kept is fused


def test_gain_on_one_staff_cannot_pay_for_a_loss_on_the_other() -> None:
    fused = [
        event("note_4", "C5", 0.3, "upper"),
        event("note_4", "D5", 0.3, "upper"),
        event("note_4", "C3", 0.3, "lower"),
        event("note_4", "D3", 0.3, "lower"),
    ]
    reread_candidate = [
        event("note_4", "C5", 0.99, "upper"),
        event("note_4", "D5", 0.99, "upper"),
        event("note_4", "E5", 0.99, "upper"),
        event("note_2", "C3", 0.99, "lower"),
    ]

    kept, replaced = reread.better_of(fused, reread_candidate)

    assert not replaced
    assert kept is fused


def test_a_reread_cannot_change_the_number_of_bars() -> None:
    fused = [event("note_4", probability=0.3), barline(), event("note_4", probability=0.3)]
    reread_candidate = [event("note_4", probability=0.99), event("note_4", probability=0.99)]

    kept, replaced = reread.better_of(fused, reread_candidate)

    assert not replaced
    assert kept is fused


def test_a_timed_rest_cannot_disappear_behind_higher_confidence() -> None:
    fused = [event("rest_4", probability=0.3), event("note_4", probability=0.3)]
    reread_candidate = [event("note_2", probability=0.99)]

    kept, replaced = reread.better_of(fused, reread_candidate)

    assert not replaced
    assert kept is fused


def test_a_note_cannot_turn_into_only_a_rest() -> None:
    fused = [event("note_4", probability=0.3)]
    reread_candidate = [event("rest_4", probability=0.99)]

    kept, replaced = reread.better_of(fused, reread_candidate)

    assert not replaced
    assert kept is fused


def test_recovered_notes_can_win_when_coverage_only_grows() -> None:
    fused = [
        event("note_2", "C5", 0.31, "upper"),
        event("note_2", "C3", 0.31, "lower"),
    ]
    reread_candidate = [
        event("note_4", "C5", 0.88, "upper"),
        event("note_4", "D5", 0.88, "upper"),
        event("note_4", "C3", 0.88, "lower"),
        event("note_4", "D3", 0.88, "lower"),
    ]

    kept, replaced = reread.better_of(fused, reread_candidate)

    assert replaced
    assert kept is reread_candidate


def test_more_music_still_has_to_earn_replacement_by_confidence() -> None:
    fused = [event("note_2", "C5", 0.8, "upper")]
    reread_candidate = [
        event("note_4", "C5", 0.6, "upper"),
        event("note_4", "D5", 0.6, "upper"),
    ]

    kept, replaced = reread.better_of(fused, reread_candidate)

    assert not replaced
    assert kept is fused


def test_coverage_is_checked_per_bar_not_only_for_the_whole_system() -> None:
    fused = [
        event("note_4", "C5", 0.3),
        event("note_4", "D5", 0.3),
        barline(),
        event("note_4", "E5", 0.3),
    ]
    reread_candidate = [
        event("note_2", "C5", 0.99),
        barline(),
        event("note_4", "E5", 0.99),
        event("note_4", "F5", 0.99),
    ]

    kept, replaced = reread.better_of(fused, reread_candidate)

    assert not replaced
    assert kept is fused
