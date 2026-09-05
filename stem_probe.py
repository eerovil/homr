#!/usr/bin/env python3
"""One-off diagnostic: compare decoded notes with detected stem directions."""

import json
import math
import sys
from pathlib import Path

import cv2
from homr.main import ProcessingConfig, detect_staffs_in_image
from homr.model import Note
from homr.staff_parsing import (
    _ensure_same_number_of_staffs,
    _get_number_of_voices,
    prepare_staff_image,
)
from homr.staff_parsing_tromr import parse_staff_tromr
from homr.staff_regions import StaffRegions
from homr.transformer.configs import Config


def direction(note: Note) -> str | None:
    return None if note.stem_direction is None else note.stem_direction.name.lower()


def stem_geometry(note: Note) -> dict[str, list[float] | float] | None:
    if note.stem is None:
        return None
    return {
        "center": [round(float(value), 1) for value in note.stem.center],
        "size": [round(float(value), 1) for value in note.stem.size],
        "angle": round(float(note.stem.angle), 1),
    }


def main(image_path: str) -> None:
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
    multi_staffs, image, debug, _, _ = detect_staffs_in_image(image_path, processing)
    multi_staffs = _ensure_same_number_of_staffs(multi_staffs)
    regions = StaffRegions(multi_staffs)
    config = Config()
    config.use_gpu_inference = False
    records = []
    wrong_stem_crop = None

    for voice_index in range(_get_number_of_voices(multi_staffs)):
        for system_index, multi_staff in enumerate(multi_staffs):
            staff = multi_staff.staffs[voice_index]
            staff_image, transformed_staff = prepare_staff_image(
                debug, len(records), staff, image, regions
            )
            detected = [symbol for symbol in transformed_staff.symbols if isinstance(symbol, Note)]
            decoded = parse_staff_tromr(transformed_staff, staff_image, config)
            detector_notes = [
                {
                    "x": round(float(note.center[0]), 1),
                    "y": round(float(note.center[1]), 1),
                    "raw_notehead_center": [round(float(value), 1) for value in note.box.center],
                    "staff_position": int(note.position),
                    "stem": direction(note),
                    "stem_directions": [item.name.lower() for item in note.stem_directions],
                    "stem_geometry": stem_geometry(note),
                }
                for note in detected
            ]
            decoded_notes = []
            for token_index, symbol in enumerate(decoded):
                if not symbol.rhythm.startswith("note") or symbol.coordinates is None:
                    continue
                x, y = symbol.coordinates[:2]
                if not math.isfinite(x) or not math.isfinite(y):
                    continue
                nearest = min(
                    detector_notes,
                    key=lambda note: (note["x"] - x) ** 2 + (note["y"] - y) ** 2,
                    default=None,
                )
                distance = None
                if nearest is not None:
                    distance = round(math.hypot(nearest["x"] - x, nearest["y"] - y), 1)
                decoded_notes.append(
                    {
                        "token": token_index,
                        "pitch": symbol.pitch,
                        "rhythm": symbol.rhythm,
                        "attention": [round(float(x), 1), round(float(y), 1)],
                        "nearest_detector_note": nearest,
                        "distance": distance,
                    }
                )
            if voice_index == 0 and system_index == 0:
                marked = cv2.cvtColor(staff_image, cv2.COLOR_GRAY2BGR)
                # In the reference score these are measure 9, lower staff, voice 5:
                # G3 then C4. MusicXML records both with up-stems.
                for marker, note in enumerate(decoded_notes, start=1):
                    if note["token"] not in (8, 13):
                        continue
                    x, y = (
                        int(note["nearest_detector_note"]["x"]),
                        int(note["nearest_detector_note"]["y"]),
                    )
                    cv2.circle(marked, (x, y), 10, (0, 0, 255), 2)
                    cv2.putText(
                        marked,
                        str(note["token"]),
                        (x - 6, y - 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )
                wrong_stem_crop = cv2.resize(marked[140:225, 60:175], None, fx=4, fy=4)
                wrong_stem_crop = cv2.copyMakeBorder(
                    wrong_stem_crop, 0, 85, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
                )
                height = wrong_stem_crop.shape[0]
                detected_by_token = {note["token"]: note["nearest_detector_note"]["stem"] for note in decoded_notes}
                cv2.putText(
                    wrong_stem_crop,
                    f"8: G3, m9 voice 5 - reference UP; matcher {detected_by_token.get(8)}",
                    (10, height - 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 180),
                    1,
                )
                cv2.putText(
                    wrong_stem_crop,
                    f"13: C4, m9 voice 5 - reference UP; matcher {detected_by_token.get(13)}",
                    (10, height - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 180),
                    1,
                )
            records.append(
                {
                    "voice_group": voice_index,
                    "system": system_index,
                    "detector_notes": detector_notes,
                    "decoded_notes": decoded_notes,
                }
            )

    output = Path(image_path).with_suffix(".stem-probe.json")
    output.write_text(json.dumps({"staffs": records}, indent=2) + "\n")
    if wrong_stem_crop is not None:
        cv2.imwrite(str(Path(image_path).with_suffix(".stem-probe-fixed-stems.png")), wrong_stem_crop)
    print(output)


if __name__ == "__main__":
    main(sys.argv[1])
