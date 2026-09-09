from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from homr import staff_parsing
from homr.main import GpuSupport, ProcessingConfig, XmlGeneratorArguments, process_image

SOURCE = Path("fixtures/sammon-ryosto.png").resolve()
ORIGINAL = staff_parsing.parse_staff_image
calls = 0


def compact(symbol):
    probability = None
    confidence = symbol.confidence or {}
    rhythm = confidence.get("rhythm") if isinstance(confidence, dict) else None
    if isinstance(rhythm, dict):
        probability = rhythm.get("probability")
    coordinates = None
    if symbol.coordinates is not None:
        try:
            coordinates = [round(float(symbol.coordinates[0]), 4), round(float(symbol.coordinates[1]), 4)]
        except Exception:
            pass
    return {
        "rhythm": symbol.rhythm,
        "pitch": symbol.pitch,
        "position": symbol.position,
        "stem": symbol.stem_direction,
        "p": probability,
        "xy": coordinates,
    }


def traced(debug, index, staff, image, regions, config):
    global calls
    calls += 1
    result = ORIGINAL(debug, index, staff, image, regions, config)
    print(
        "TRACE "
        + json.dumps(
            {
                "call": calls,
                "index": index,
                "grand": bool(staff.is_grandstaff),
                "merged": staff.merged_from is not None,
                "symbols": [compact(symbol) for symbol in result],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


staff_parsing.parse_staff_image = traced
with tempfile.TemporaryDirectory(prefix="homr-reread-61-") as directory:
    image = Path(directory) / "sammon-ryosto.png"
    shutil.copyfile(SOURCE, image)
    config = ProcessingConfig(
        enable_debug=False,
        enable_cache=False,
        write_staff_positions=False,
        write_confidence=True,
        score_settings=None,
        read_staff_positions=False,
        selected_staff=-1,
        transformer_use_gpu=False,
        segnet_use_gpu=False,
        coreml_encoder=False,
        title_detection=False,
    )
    process_image(str(image), config, XmlGeneratorArguments())
    output = image.with_suffix(".musicxml")
    target = Path(os.environ["TRACE_OUTPUT"])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output, target)
