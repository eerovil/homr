from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from homr import music_xml_generator as xmlgen
from homr.main import _write_confidence
from homr.score_reconstruction import (
    ReconstructionChange,
    SymbolChord,
    infer_meter_changes,
    reconstruct_voice,
    repair_bar_arithmetic,
)
from homr.transformer.vocabulary import EncodedSymbol


def note(
    rhythm: str,
    position: str = "upper",
    alternatives: tuple[str, ...] = (),
    pitch: str = "C4",
) -> EncodedSymbol:
    confidence = None
    if alternatives:
        confidence = {
            "rhythm": {
                "value": rhythm,
                "alternatives": [{"value": value, "probability": 0.01} for value in alternatives],
            }
        }
    return EncodedSymbol(
        rhythm=rhythm,
        pitch=pitch,
        position=position,
        confidence=confidence,
        stem_direction="up",
    )


def moment(*symbols: EncodedSymbol) -> SymbolChord:
    return SymbolChord(list(symbols))


def barline() -> SymbolChord:
    return SymbolChord([EncodedSymbol("barline")])


def even_bar(quarters: int = 2) -> list[SymbolChord]:
    return [moment(note("note_4", "upper"), note("note_4", "lower")) for _ in range(quarters)] + [
        barline()
    ]


def broken_bar() -> list[SymbolChord]:
    return [
        moment(
            note("note_2", "upper", alternatives=("note_4.",), pitch="D5"),
            note("note_2", "lower", pitch="G2"),
        ),
        moment(note("rest_16", "upper")),
        moment(note("note_16", "upper", pitch="D5")),
        barline(),
    ]


def test_bar_arithmetic_records_what_changed_and_why() -> None:
    voice = [*even_bar(), *broken_bar(), *even_bar()]
    changes: list[ReconstructionChange] = []

    repaired = repair_bar_arithmetic(voice, changes)

    assert repaired[3].symbols[0].rhythm == "note_4."
    assert changes == [
        ReconstructionChange(
            kind="rhythm_repair",
            bar=2,
            group=3,
            symbol=0,
            staff="upper",
            pitch="D5",
            before="note_2",
            after="note_4.",
            reason="the only alternative that makes its bar add up",
        )
    ]


def test_refused_or_unneeded_reconstruction_records_nothing() -> None:
    clean = [*even_bar(), *even_bar(), *even_bar()]
    changes: list[ReconstructionChange] = []
    assert repair_bar_arithmetic(clean, changes) is clean
    assert changes == []


def test_inferred_meter_records_the_signature_and_bar() -> None:
    first = [
        SymbolChord([EncodedSymbol("timeSignature/4")]),
        *even_bar(3),
    ]
    second = even_bar(5)
    voice = [*first, *second]
    changes: list[ReconstructionChange] = []

    inferred = infer_meter_changes(voice, changes)

    assert changes == [
        ReconstructionChange(
            kind="meter_inference",
            bar=2,
            group=len(first),
            symbol=None,
            staff=None,
            pitch=None,
            before="3/4",
            after="5/4",
            reason="every detected staff agrees on the bar length",
        )
    ]
    assert inferred[len(first)].symbols[0].rhythm == "timeSignature/4"


def test_reconstructed_voice_always_exposes_an_immutable_change_list() -> None:
    result = reconstruct_voice([EncodedSymbol("note_4"), EncodedSymbol("barline")])
    assert result.changes == ()
    assert isinstance(result.changes, tuple)


def test_generate_xml_collects_the_changes_from_each_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    change = ReconstructionChange(
        kind="rhythm_repair",
        bar=2,
        group=3,
        symbol=0,
        staff="upper",
        pitch="D5",
        before="note_2",
        after="note_4.",
        reason="test",
    )

    def fake_build_part(
        args: xmlgen.XmlGeneratorArguments,
        voice: list[EncodedSymbol],
        index: int,
        has_two_staves: bool,
        reconstruction_changes: list[ReconstructionChange] | None = None,
    ) -> ET.Element:
        assert reconstruction_changes is not None
        reconstruction_changes.append(change)
        return ET.Element("part")

    monkeypatch.setattr(xmlgen, "build_part", fake_build_part)
    collected: list[list[ReconstructionChange]] = []
    xmlgen.generate_xml(xmlgen.XmlGeneratorArguments(), [[]], "", collected)
    assert collected == [[change]]


def test_confidence_sidecar_adds_provenance_without_breaking_version_one(tmp_path: Path) -> None:
    symbol = EncodedSymbol(
        "note_4",
        pitch="C4",
        confidence={"rhythm": {"value": "note_4", "probability": 0.9}},
    )
    change = ReconstructionChange(
        kind="rhythm_repair",
        bar=2,
        group=3,
        symbol=0,
        staff="upper",
        pitch="D5",
        before="note_2",
        after="note_4.",
        reason="the only alternative that makes its bar add up",
    )
    path = tmp_path / "page.confidence.json"

    _write_confidence(str(path), [[symbol]], [[change]])
    data = json.loads(path.read_text())

    assert data["version"] == 1
    assert data["symbols"][0]["token"]["rhythm"] == "note_4"
    assert data["reconstruction"] == {
        "version": 1,
        "changes": [
            {
                "staff": 0,
                "kind": "rhythm_repair",
                "bar": 2,
                "group": 3,
                "symbol": 0,
                "position": "upper",
                "pitch": "D5",
                "before": "note_2",
                "after": "note_4.",
                "reason": "the only alternative that makes its bar add up",
            }
        ],
    }
