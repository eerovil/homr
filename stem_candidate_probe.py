#!/usr/bin/env python3
"""One-off report of raw stem-mask candidates around detected noteheads."""

import json
import sys
from pathlib import Path

from homr.main import load_and_preprocess_predictions, predict_symbols


def geometry(box: object) -> dict[str, list[float] | float]:
    center = box.center  # type: ignore[attr-defined]
    size = box.size  # type: ignore[attr-defined]
    return {
        "center": [round(float(value), 1) for value in center],
        "size": [round(float(value), 1) for value in size],
        "angle": round(float(box.angle), 1),  # type: ignore[attr-defined]
    }


def main(image_path: str) -> None:
    predictions, debug = load_and_preprocess_predictions(image_path, False, False, False)
    symbols = predict_symbols(debug, predictions)
    records = []
    for notehead in symbols.noteheads:
        nearby = [stem for stem in symbols.stems_rest if stem.is_overlapping(notehead.make_box_thicker(15))]
        plausible = [
            stem
            for stem in nearby
            if stem.size[1] >= max(notehead.size[1] * 1.5, notehead.size[0])
            and stem.size[0] <= notehead.size[0] * 0.75
            and abs(stem.center[0] - notehead.center[0]) <= (notehead.size[0] + stem.size[0]) / 2
        ]
        records.append(
            {
                "debug_id": notehead.debug_id,
                "notehead": geometry(notehead),
                "nearby": [geometry(stem) for stem in nearby],
                "plausible": [geometry(stem) for stem in plausible],
            }
        )
    output = Path(image_path).with_suffix(".stem-candidates.json")
    output.write_text(json.dumps({"notes": records}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main(sys.argv[1])
