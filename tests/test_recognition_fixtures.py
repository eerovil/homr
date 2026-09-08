# ruff: noqa: S101

"""The five real MusicXML acceptance cases, not just segmentation checks.

Use the existing comparator and frozen per-case memory without ratcheting it.
Missing models may skip locally; HOMR_REQUIRE_MODELS=1 makes CI fail instead.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fixturecheck import cases, references
from fixturecheck.compare import compare_output
from homr.segmentation.config import segnet_path_onnx
from homr.transformer.configs import Config
from tests.model_requirements import require_model

EXPECTED_CASES = {
    "hanget-soi",
    "kolme-kakea",
    "laulun-aika-s2",
    "sammon-ryosto",
    "system4",
}


@pytest.fixture(scope="module")
def acceptance_context() -> dict:
    roster = cases.fixture_cases()
    assert {case.name for case in roster} == EXPECTED_CASES
    assert len(roster) == len(EXPECTED_CASES)
    assert references.drift(roster) == {"changed": [], "unfrozen": []}
    memory = references.accepted()
    assert set(memory) >= EXPECTED_CASES
    paths = Config().filepaths
    models = [Path(name) for name in (segnet_path_onnx, paths.encoder_path, paths.decoder_path)]
    for model in models:
        require_model(model)
    return {
        "sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cases.ROOT, text=True  # noqa: S607
        ).strip(),
        "memory": memory,
        "manifest": references.MANIFEST.read_bytes(),
        "models": {
            model.name: hashlib.sha256(model.read_bytes()).hexdigest() for model in models
        },
    }


@pytest.mark.parametrize("case", cases.fixture_cases(), ids=lambda case: case.name)
def test_real_recognition_meets_frozen_acceptance(
    case: cases.Case, tmp_path: Path, acceptance_context: dict
) -> None:
    before = references.fingerprint(case)
    image = tmp_path / "input.png"
    shutil.copyfile(case.image, image)
    # Never run homr beside the reference: its output shares the image stem.
    run = subprocess.run(  # noqa: S603 -- this interpreter and fixed CLI, in a scratch directory
        [
            sys.executable,
            "-c",
            "from homr.main import main; main()",
            "--gpu",
            "no",
            "--no-title",
            str(image),
        ],
        cwd=cases.ROOT,
        env={**os.environ, "PYTHONPATH": str(cases.ROOT)},
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    evidence = Path(os.environ.get("HOMR_FIXTURE_EVIDENCE", str(tmp_path))) / case.name
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "recognition.log").write_text(run.stdout + "\n" + run.stderr)
    assert references.fingerprint(case) == before, "Fixture input or reference was modified"
    assert references.MANIFEST.read_bytes() == acceptance_context["manifest"]
    assert run.returncode == 0, "\n".join(run.stderr.splitlines()[-25:])
    output = image.with_suffix(".musicxml")
    assert output.is_file(), "Recognition produced no MusicXML"
    shutil.copyfile(output, evidence / "output.musicxml")
    result = compare_output(case.reference, output, case.name)
    counts = {
        name: getattr(result, name)
        for name in (
            "agree",
            "voice",
            "pitch",
            "size",
            "timing",
            "meter",
            "unison",
            "staves_page",
            "staves_homr",
        )
    }
    marks = references.marks(counts)
    accepted = acceptance_context["memory"][case.name]
    failures = references.worse(marks, accepted)
    record = {
        "case": case.name,
        "sha": acceptance_context["sha"],
        "inputs": before,
        "models": acceptance_context["models"],
        "counts": counts,
        "accepted": accepted,
        "measured": marks,
        "failures": failures,
        "faults": [
            {"where": row.where, "page": row.page, "homr": row.homr, "kind": row.kind}
            for row in result.rows
            if row.kind != "agree"
        ],
    }
    (evidence / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, sort_keys=True))  # noqa: T201 -- per-case evidence in CI logs
    assert not failures, f"{case.name}: " + "; ".join(failures)
