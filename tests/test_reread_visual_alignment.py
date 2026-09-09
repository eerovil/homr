from __future__ import annotations

from fractions import Fraction

import numpy as np

from homr import reread
from homr.transformer.vocabulary import EncodedSymbol


def placed(rhythm: str, pitch: str, x: float) -> EncodedSymbol:
    return EncodedSymbol(
        rhythm=rhythm,
        pitch=pitch,
        position="upper",
        coordinates=np.array([x, 100.0]),
    )


def chord() -> EncodedSymbol:
    return EncodedSymbol("chord")


def signature(symbols: list[EncodedSymbol]) -> list[tuple[str, str, str]]:
    return [(symbol.rhythm, symbol.pitch, symbol.position) for symbol in symbols]


def unstable_bar(first_rhythm: str) -> tuple[list[EncodedSymbol], list[EncodedSymbol]]:
    """The Sammon column where a CPU-dependent rhythm changed the next onset."""
    upper = [
        placed(first_rhythm, "B4", 693.4),
        chord(),
        placed("note_2.", "B4", 695.0),
        placed("rest_4", ".", 823.7),
    ]
    lower = [
        placed("note_2.", "G3", 687.0),
        chord(),
        placed("note_4", "E3", 695.9),
        placed("note_4", "B2", 741.1),
        placed("note_4", "B2", 783.5),
        placed("note_4", "D3", 820.7),
    ]
    return upper, lower


def test_page_columns_keep_the_next_moment_stable_when_a_duration_flips() -> None:
    dotted = reread.splice(*unstable_bar("note_2."))
    half = reread.splice(*unstable_bar("note_2"))
    assert dotted is not None and half is not None

    # The disputed duration remains the decoder's reading. What is stabilized is
    # which cross-staff notes share the following printed column.
    for result in (dotted, half):
        rest = next(index for index, symbol in enumerate(result) if symbol.rhythm == "rest_4")
        assert signature(result[rest - 1 : rest + 3]) == [
            ("note_4", "B2", "lower"),
            ("rest_4", ".", "upper"),
            ("chord", ".", "."),
            ("note_4", "D3", "lower"),
        ]


def test_visual_merge_uses_the_measured_twelve_pixel_boundary() -> None:
    upper = [(Fraction(0), [placed("note_4", "C5", 100.0)])]
    within = [(Fraction(1, 4), [placed("note_4", "C3", 112.0)])]
    outside = [(Fraction(1, 4), [placed("note_4", "C3", 112.1)])]

    merged = reread._merge_timed_by_page_x(upper, within)
    separate = reread._merge_timed_by_page_x(upper, outside)
    assert merged is not None and len(merged) == 1
    assert separate is not None and len(separate) == 2


def test_missing_attention_keeps_the_existing_duration_fallback() -> None:
    upper = [
        EncodedSymbol("note_2", pitch="C5", position="upper"),
        EncodedSymbol("note_2", pitch="D5", position="upper"),
    ]
    lower = [
        EncodedSymbol("note_4", pitch="C3", position="upper"),
        EncodedSymbol("note_4", pitch="D3", position="upper"),
        EncodedSymbol("note_4", pitch="E3", position="upper"),
        EncodedSymbol("note_4", pitch="F3", position="upper"),
    ]

    spliced = reread.splice(upper, lower)
    assert spliced is not None
    assert signature(spliced) == [
        ("note_2", "C5", "upper"),
        ("chord", ".", "."),
        ("note_4", "C3", "lower"),
        ("note_4", "D3", "lower"),
        ("note_2", "D5", "upper"),
        ("chord", ".", "."),
        ("note_4", "E3", "lower"),
        ("note_4", "F3", "lower"),
    ]


def test_visual_merge_preserves_each_staffs_decoded_order() -> None:
    upper = [
        (Fraction(0), [placed("note_4", "C5", 100.0)]),
        (Fraction(1, 4), [placed("note_4", "D5", 99.5)]),
    ]
    lower = [(Fraction(0), [placed("note_4", "C3", 100.2)])]

    merged = reread._merge_timed_by_page_x(upper, lower)
    assert merged is not None
    assert [symbol.pitch for moment in merged for symbol in moment if symbol.rhythm != "chord"] == [
        "C5",
        "C3",
        "D5",
    ]
