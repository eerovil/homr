"""The model prerequisite policy is testable without ONNX or real weights."""

from pathlib import Path

import pytest

from tests.model_requirements import require_model


@pytest.mark.parametrize("required", [False, True])
def test_existing_model_is_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, required: bool
) -> None:
    monkeypatch.setenv("HOMR_REQUIRE_MODELS", "1" if required else "0")
    model = tmp_path / "segnet.onnx"
    model.write_bytes(b"test model placeholder")
    require_model(model)


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("kind", ["missing", "directory", "broken-link"])
def test_unavailable_model_fails_when_required_otherwise_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, required: bool, kind: str
) -> None:
    monkeypatch.setenv("HOMR_REQUIRE_MODELS", "1" if required else "0")
    model = tmp_path / "segnet.onnx"
    if kind == "directory":
        model.mkdir()
    elif kind == "broken-link":
        model.symlink_to(tmp_path / "missing.onnx")
    outcome = pytest.fail.Exception if required else pytest.skip.Exception
    with pytest.raises(outcome, match="Run homr --init --gpu no"):
        require_model(model)


def test_model_free_local_run_still_skips_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HOMR_REQUIRE_MODELS", raising=False)
    with pytest.raises(pytest.skip.Exception, match="Segmentation model is not installed"):
        require_model(tmp_path / "missing.onnx")
