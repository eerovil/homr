from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from homr import staff_parsing as parsing
from homr.errors import IncompleteRecognitionError
from homr.model import MultiStaff
from homr.transformer.vocabulary import EncodedSymbol


def rows(count: int = 2) -> list[MultiStaff]:
    return [Mock(staffs=[Mock()]) for _ in range(count)]


def prepare(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parsing, "_ensure_same_number_of_staffs", lambda value: value)
    monkeypatch.setattr(parsing, "StaffRegions", Mock())
    monkeypatch.setattr(parsing, "_reread_if_doubtful", lambda *args: args[3])
    monkeypatch.setattr(parsing, "remove_duplicated_symbols", lambda value: value)


@pytest.mark.parametrize(
    "reading",
    [
        [EncodedSymbol("clef_G2")],
        [EncodedSymbol("keySignature_C"), EncodedSymbol("barline")],
        [EncodedSymbol("timeSignature/4"), EncodedSymbol("barline")],
    ],
)
def test_header_only_staff_is_not_reported_as_complete(
    monkeypatch: pytest.MonkeyPatch, reading: list[EncodedSymbol]
) -> None:
    prepare(monkeypatch)
    read = Mock(return_value=reading)
    monkeypatch.setattr(parsing, "parse_staff_image", read)

    with pytest.raises(IncompleteRecognitionError, match="Staff 0: no notes or rests"):
        parsing.parse_staffs(Mock(), rows(1), np.zeros((1, 1)), Mock())

    read.assert_called_once()


def test_a_rest_is_enough_to_make_a_staff_musically_nonempty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare(monkeypatch)
    monkeypatch.setattr(
        parsing,
        "parse_staff_image",
        Mock(return_value=[EncodedSymbol("clef_G2"), EncodedSymbol("rest_1")]),
    )

    result = parsing.parse_staffs(Mock(), rows(1), np.zeros((1, 1)), Mock())
    assert [symbol.rhythm for symbol in result[0]] == ["clef_G2", "rest_1", "newline"]


def test_out_of_range_selected_staff_fails_before_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare(monkeypatch)
    read = Mock()
    monkeypatch.setattr(parsing, "parse_staff_image", read)

    with pytest.raises(
        IncompleteRecognitionError,
        match="Selected staff 2 does not exist; page has 2 staff rows",
    ):
        parsing.parse_staffs(Mock(), rows(2), np.zeros((1, 1)), Mock(), selected_staff=2)

    read.assert_not_called()


def test_last_existing_selected_staff_is_still_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare(monkeypatch)
    read = Mock(return_value=[EncodedSymbol("note_4", "C4")])
    monkeypatch.setattr(parsing, "parse_staff_image", read)

    result = parsing.parse_staffs(Mock(), rows(2), np.zeros((1, 1)), Mock(), selected_staff=1)

    read.assert_called_once()
    assert [symbol.rhythm for symbol in result[0]] == ["note_4", "newline"]
