"""Allow model-free local tests, but never silently skip required CI fixtures."""

import os
from pathlib import Path

import pytest


def require_model(path: Path) -> None:
    if path.is_file():
        return
    message = (
        f"Segmentation model is not installed: {path}. "
        "Run homr --init --gpu no before model-backed tests."
    )
    if os.environ.get("HOMR_REQUIRE_MODELS") == "1":
        pytest.fail(message, pytrace=False)
    pytest.skip(message)
