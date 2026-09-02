#!/usr/bin/env python3
"""Why one notehead got the stem directions it did.

Usage: stem_diagnose.py <fixture> <scan x> <scan y> [radius]
"""

import sys
from pathlib import Path

import numpy as np

from homr.main import load_and_preprocess_predictions, predict_symbols
from homr.note_detection import (
    is_attached,
    is_plausible_stem,
    source_stem_candidates,
    split_notehead_ellipse,
    stem_direction,
    vertical_ink,
)


def main() -> None:
    name, target_x, target_y = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    radius = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0
    image = Path(__file__).parent / "fixtures" / f"{name}.png"
    predictions, _ = load_and_preprocess_predictions(str(image), False, False, False)
    symbols = predict_symbols(_Quiet(), predictions)
    unit_size = float(np.median([notehead.size[1] for notehead in symbols.noteheads]))
    noteheads = [
        split
        for notehead in symbols.noteheads
        for split in split_notehead_ellipse(notehead, predictions.notehead, unit_size)
    ]
    ink = vertical_ink(predictions.preprocessed)
    for notehead in noteheads:
        if abs(notehead.center[0] - target_x) > radius or abs(notehead.center[1] - target_y) > radius:
            continue
        print(
            f"notehead center {notehead.center} size"
            f" ({notehead.size[0]:.1f}, {notehead.size[1]:.1f})"
        )
        thickened = notehead.make_box_thicker(15)
        for stem in symbols.stems_rest:
            if not stem.is_overlapping(thickened):
                continue
            print(
                f"  learned candidate center ({stem.center[0]:.1f}, {stem.center[1]:.1f})"
                f" size ({stem.size[0]:.1f}, {stem.size[1]:.1f})"
                f" side={stem_direction(notehead, stem)}"
                f" attached={is_attached(notehead, stem)}"
                f" plausible={is_plausible_stem(notehead, stem)}"
            )
        for stem in source_stem_candidates(notehead, ink):
            print(
                f"  scan candidate center ({stem.center[0]:.1f}, {stem.center[1]:.1f})"
                f" size ({stem.size[0]:.1f}, {stem.size[1]:.1f})"
                f" side={stem_direction(notehead, stem)}"
            )


class _Quiet:
    def write_threshold_image(self, *args: object) -> None:
        pass

    def write_bounding_boxes(self, *args: object) -> None:
        pass


if __name__ == "__main__":
    main()
