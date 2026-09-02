"""Check homr's detected stem directions against hand-corrected scores.

Each fixture is one printed system: a scan, and a MusicXML reference somebody
read off that scan by hand.  The reference is the only ground truth here -- an
earlier oracle guessed stem directions from where a note sat on the staff, and
it disagreed with the page.

Every reference notehead must be detected with exactly the stems drawn on it,
including a head that carries both because two voices meet on it.  A detection
the reference has no note for fails as well: a stem direction on a note that
is not there would drive a voice just as confidently as a real one.  What a
fixture cannot yet meet is listed in ``known_gaps`` in the manifest, with the
reason, so this file is a guard against losing ground rather than a wish.
"""

import json
import os
from pathlib import Path

import pytest

from homr.main import ProcessingConfig, detect_staffs_in_image
from homr.model import Note, Staff
from homr.segmentation.config import segnet_path_onnx
from tests.fixture_matching import check_fixture
from tests.fixture_reference import reference_staffs

FIXTURES = Path(__file__).parent.parent / "fixtures"
MANIFEST = FIXTURES / "stem-direction-fixtures.json"


def fixture_names() -> list[str]:
    return sorted(json.loads(MANIFEST.read_text())["fixtures"])


def staff_notes(staff: Staff) -> list[dict]:
    """One printed staff's detected noteheads, left to right."""
    notes = [
        {
            "x": round(float(note.center[0]), 1),
            "y": round(float(note.center[1]), 1),
            "w": round(float(note.box.size[0]), 1),
            "h": round(float(note.box.size[1]), 1),
            "position": int(note.position),
            "stems": sorted(item.name.lower() for item in note.stem_directions),
        }
        for note in staff.symbols
        if isinstance(note, Note)
    ]
    notes.sort(key=lambda note: (note["x"], -note["position"]))
    return notes


def detect(image: Path) -> list[list[dict]]:
    config = ProcessingConfig(
        enable_debug=False,
        enable_cache=False,
        write_staff_positions=False,
        write_confidence=False,
        score_settings=None,
        read_staff_positions=False,
        selected_staff=-1,
        transformer_use_gpu=False,
        segnet_use_gpu=False,
        coreml_encoder=False,
        title_detection=False,
    )
    _, _, _, _, staffs = detect_staffs_in_image(str(image), config)
    return [staff_notes(staff) for staff in sorted(staffs, key=lambda staff: staff.min_y)]


@pytest.mark.skipif(
    not os.path.exists(segnet_path_onnx), reason="the segmentation model is not installed"
)
@pytest.mark.parametrize("name", fixture_names())
def test_stem_directions_match_the_reference_score(name: str) -> None:
    entry = json.loads(MANIFEST.read_text())["fixtures"][name]
    reference = reference_staffs(FIXTURES / entry["reference"])
    detected = detect(FIXTURES / entry["image"])
    allowed = {gap["failure"] for gap in entry.get("known_gaps", [])}
    failures = [
        failure
        for result in check_fixture(reference, detected)
        for failure in result.failures
    ]
    unexpected = [failure for failure in failures if failure not in allowed]
    fixed = allowed - set(failures)
    assert not unexpected, "\n".join([f"{name}: new stem-direction failures:", *unexpected])
    assert not fixed, "\n".join(
        [f"{name}: these known gaps are fixed, remove them from the manifest:", *sorted(fixed)]
    )
