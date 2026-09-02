#!/usr/bin/env python3
"""Dump detected physical noteheads and the reference notes for a fixture."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from homr.main import ProcessingConfig, detect_staffs_in_image  # noqa: E402
from homr.model import Note, Staff  # noqa: E402
from tests.fixture_reference import reference_staffs  # noqa: E402

def staff_notes(staff: Staff) -> list[dict]:
    """One printed staff's noteheads, left to right."""
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


def probe(image_path: Path) -> list[list[dict]]:
    processing = ProcessingConfig(
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
    _, _, _, _, staffs = detect_staffs_in_image(str(image_path), processing)
    return [staff_notes(staff) for staff in sorted(staffs, key=lambda staff: staff.min_y)]


def main() -> None:
    fixtures = Path(__file__).parent / "fixtures"
    manifest = json.loads((fixtures / "stem-direction-fixtures.json").read_text())
    names = sys.argv[1:] or sorted(manifest["fixtures"])
    for name in names:
        entry = manifest["fixtures"][name]
        data = {"detected": probe(fixtures / entry["image"])}
        data["reference"] = reference_staffs(fixtures / entry["reference"])
        out = fixtures / f"{name}.fixture-probe.json"
        out.write_text(json.dumps(data, indent=1) + "\n")
        detected = [len(staff) for staff in data["detected"]]
        expected = [len(staff["notes"]) for staff in data["reference"]]
        print(f"{name}: detected {detected} reference {expected}")


if __name__ == "__main__":
    main()
