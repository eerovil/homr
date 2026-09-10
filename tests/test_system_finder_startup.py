from __future__ import annotations

import sys
from pathlib import Path

import pytest

from homr import main
from homr.system_crops import SystemBounds


def test_finder_cli_initializes_segnet_without_transformer_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "score.pdf"
    pdf.write_bytes(b"pdf")
    downloads: list[tuple[bool, bool, bool, bool]] = []

    def download(
        segnet_use_gpu: bool,
        transformer_use_gpu: bool,
        coreml_encoder: bool,
        include_transformer: bool = True,
    ) -> None:
        downloads.append((segnet_use_gpu, transformer_use_gpu, coreml_encoder, include_transformer))

    monkeypatch.setattr(main, "download_weights", download)
    monkeypatch.setattr(
        main,
        "find_system_bounds",
        lambda *args, **kwargs: [SystemBounds(index=1, page=1, top=0.1, bottom=0.9)],
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["homer", str(pdf), "--gpu", "no", "--find-system-bounds"],
    )

    main.main()

    assert downloads == [(False, False, False, False)]
