from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODE = os.environ.get("ORT_MODE", "default")
_REAL_SESSION = ort.InferenceSession


def _session(path, sess_options=None, *args, **kwargs):
    """Apply diagnostic CPU options to transformer sessions only."""
    path_text = os.fspath(path) if isinstance(path, os.PathLike) else str(path)
    is_transformer = "encoder" in path_text or "decoder" in path_text
    if is_transformer and MODE != "default":
        options = sess_options or ort.SessionOptions()
        if MODE in {"deterministic", "both"}:
            options.use_deterministic_compute = True
        if MODE in {"single", "both"}:
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options = options
    return _REAL_SESSION(path, sess_options, *args, **kwargs)


ort.InferenceSession = _session

from homr import main as homr_main  # noqa: E402
from homr import staff_parsing  # noqa: E402
from homr.main import ProcessingConfig, XmlGeneratorArguments, process_image  # noqa: E402

SOURCE = Path("fixtures/sammon-ryosto.png").resolve()
ORIGINAL_PARSE = staff_parsing.parse_staff_image
ORIGINAL_PREDICTIONS = homr_main.get_predictions
parse_calls = 0


def _hash(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def traced_predictions(*args, **kwargs):
    result = ORIGINAL_PREDICTIONS(*args, **kwargs)
    print(
        "SEGMENTATION "
        + json.dumps(
            {
                name: _hash(getattr(result, name))
                for name in ("staff", "symbols", "stems_rest", "notehead", "clefs_keys")
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def compact(symbol):
    confidence = symbol.confidence or {}
    rhythm_confidence = confidence.get("rhythm") if isinstance(confidence, dict) else None
    coordinates = None
    if symbol.coordinates is not None:
        try:
            coordinates = [round(float(symbol.coordinates[0]), 6), round(float(symbol.coordinates[1]), 6)]
        except Exception:
            pass
    return {
        "rhythm": symbol.rhythm,
        "pitch": symbol.pitch,
        "position": symbol.position,
        "stem": symbol.stem_direction,
        "rhythm_confidence": rhythm_confidence,
        "xy": coordinates,
    }


def traced_parse(debug, index, staff, image, regions, config):
    global parse_calls
    parse_calls += 1
    result = ORIGINAL_PARSE(debug, index, staff, image, regions, config)
    print(
        "TRACE "
        + json.dumps(
            {
                "call": parse_calls,
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


homr_main.get_predictions = traced_predictions
staff_parsing.parse_staff_image = traced_parse

cpu = ""
try:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
except OSError:
    pass
print(
    "ENV "
    + json.dumps(
        {
            "mode": MODE,
            "cpu": cpu or platform.processor(),
            "machine": platform.machine(),
            "onnxruntime": ort.__version__,
        },
        sort_keys=True,
    ),
    flush=True,
)

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
