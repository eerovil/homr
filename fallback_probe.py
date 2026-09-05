#!/usr/bin/env python3
"""Measure source-image vertical ink near stem-mask-missed noteheads."""

import cv2
import numpy as np

from homr.main import load_and_preprocess_predictions, predict_symbols
from homr.note_detection import combine_noteheads_with_stems


def main() -> None:
    predictions, debug = load_and_preprocess_predictions("system-4.png", False, False, False)
    symbols = predict_symbols(debug, predictions)
    combined = combine_noteheads_with_stems(symbols.noteheads, symbols.stems_rest)
    image = predictions.preprocessed
    ink = (image < 180).astype(np.uint8)
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, 30), np.uint8))
    ink = ink & (1 - horizontal)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((1, 7), np.uint8))
    for item in combined:
        if item.stem is not None:
            continue
        head = item.notehead
        x, y = map(int, head.center)
        width, height = map(int, head.size)
        for direction, xs, ys in (
            ("up", range(x, x + width + 16), range(max(0, y - 80), y + 1)),
            ("down", range(max(0, x - width - 16), x + 1), range(y, min(ink.shape[0], y + 81))),
        ):
            best = (0, 0, 0, 0)
            for column in xs:
                values = ink[list(ys), column].astype(int)
                changes = np.diff(np.pad(values, (1, 1)))
                for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True):
                    if end - start > best[0]:
                        best = (int(end - start), column, ys[start], ys[end - 1])
            print("head", head.center, head.size, direction, "longest", best)


if __name__ == "__main__":
    main()
