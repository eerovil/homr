from __future__ import annotations

import json
from pathlib import Path

import pytest

from homr import main
from homr.music_xml_generator import XmlGeneratorArguments
from homr.system_crops import DEFAULT_SYSTEM_PAD, SystemBounds, SystemCrop


def _config(write_confidence: bool = False) -> main.ProcessingConfig:
    return main.ProcessingConfig(
        enable_debug=False,
        enable_cache=False,
        write_staff_positions=False,
        write_confidence=write_confidence,
        score_settings=None,
        read_staff_positions=False,
        selected_staff=-1,
        transformer_use_gpu=False,
        segnet_use_gpu=False,
        coreml_encoder=False,
        title_detection=False,
    )


def _bounds(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "systems": [
                    {"index": 2, "page": 1, "top": 0.1, "bottom": 0.3},
                    {"index": 5, "page": 1, "top": 0.4, "bottom": 0.7},
                    {"index": 9, "page": 2, "top": 0.2, "bottom": 0.6},
                ]
            }
        )
    )


def _fake_crops(
    monkeypatch: pytest.MonkeyPatch, seen: list[tuple[list[int], float]]
) -> None:
    def render(
        pdf: str,
        bounds: list[SystemBounds],
        out_dir: str,
        dpi: int,
        pad: float,
    ) -> list[SystemCrop]:
        seen.append(([bound.index for bound in bounds], pad))
        result = []
        for bound in bounds:
            path = Path(out_dir) / f"system-{bound.index:03d}.png"
            path.write_bytes(b"image")
            result.append(SystemCrop(bound, str(path)))
        return result

    monkeypatch.setattr(main, "render_system_crops", render)


def test_every_system_is_processed_independently_and_published_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "score.pdf"
    pdf.write_bytes(b"pdf")
    bounds = tmp_path / ".systems.json"
    _bounds(bounds)
    seen: list[tuple[list[int], float]] = []
    _fake_crops(monkeypatch, seen)
    calls: list[str] = []

    def process(path: str, config: main.ProcessingConfig, args: XmlGeneratorArguments) -> None:
        calls.append(Path(path).stem)
        Path(main.replace_extension(path, ".musicxml")).write_text(Path(path).stem)

    monkeypatch.setattr(main, "process_image", process)

    outputs = main.process_system_bounds(
        str(pdf), str(bounds), _config(), XmlGeneratorArguments(), dpi=200
    )

    assert seen == [([2, 5, 9], DEFAULT_SYSTEM_PAD)]
    assert calls == ["system-002", "system-005", "system-009"]
    assert [Path(path).name for path in outputs] == [
        "score_system-002.musicxml",
        "score_system-005.musicxml",
        "score_system-009.musicxml",
    ]
    assert [Path(path).read_text() for path in outputs] == calls


def test_one_system_index_preserves_the_apps_short_lease_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "score.pdf"
    pdf.write_bytes(b"pdf")
    bounds = tmp_path / ".systems.json"
    _bounds(bounds)
    seen: list[tuple[list[int], float]] = []
    _fake_crops(monkeypatch, seen)

    def process(path: str, config: main.ProcessingConfig, args: XmlGeneratorArguments) -> None:
        Path(main.replace_extension(path, ".musicxml")).write_text("ok")

    monkeypatch.setattr(main, "process_image", process)
    outputs = main.process_system_bounds(
        str(pdf),
        str(bounds),
        _config(),
        XmlGeneratorArguments(),
        dpi=200,
        system_index=5,
        pad=0.03,
    )

    assert seen == [([5], 0.03)]
    assert [Path(path).name for path in outputs] == ["score_system-005.musicxml"]


def test_failed_system_leaves_no_stale_or_partial_published_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "score.pdf"
    pdf.write_bytes(b"pdf")
    bounds = tmp_path / ".systems.json"
    _bounds(bounds)
    seen: list[tuple[list[int], float]] = []
    _fake_crops(monkeypatch, seen)
    for index in (2, 5, 9):
        (tmp_path / f"score_system-{index:03d}.musicxml").write_text("stale")
        (tmp_path / f"score_system-{index:03d}.confidence.json").write_text("stale")

    def process(path: str, config: main.ProcessingConfig, args: XmlGeneratorArguments) -> None:
        index = int(Path(path).stem.split("-")[-1])
        if index == 5:
            raise RuntimeError("system failed")
        Path(main.replace_extension(path, ".musicxml")).write_text("temporary")

    monkeypatch.setattr(main, "process_image", process)
    with pytest.raises(RuntimeError, match="system failed"):
        main.process_system_bounds(
            str(pdf), str(bounds), _config(write_confidence=True), XmlGeneratorArguments(), dpi=200
        )

    assert seen == [([2, 5, 9], DEFAULT_SYSTEM_PAD)]
    for index in (2, 5, 9):
        assert not (tmp_path / f"score_system-{index:03d}.musicxml").exists()
        assert not (tmp_path / f"score_system-{index:03d}.confidence.json").exists()


def test_confidence_sidecars_follow_their_system_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "score.pdf"
    pdf.write_bytes(b"pdf")
    bounds = tmp_path / ".systems.json"
    _bounds(bounds)
    _fake_crops(monkeypatch, [])

    def process(path: str, config: main.ProcessingConfig, args: XmlGeneratorArguments) -> None:
        Path(main.replace_extension(path, ".musicxml")).write_text("xml")
        Path(main.replace_extension(path, ".confidence.json")).write_text("confidence")

    monkeypatch.setattr(main, "process_image", process)
    outputs = main.process_system_bounds(
        str(pdf), str(bounds), _config(write_confidence=True), XmlGeneratorArguments(), system_index=2
    )

    assert len(outputs) == 1
    assert (tmp_path / "score_system-002.confidence.json").read_text() == "confidence"
