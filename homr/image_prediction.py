"""Shared segmentation and symbol-bound prediction primitives.

These are the pixel-level stages used by both full recognition and lightweight
system-bound proposal. They deliberately stop before staff parsing or transformer
decoding.
"""

from __future__ import annotations

import cv2
import numpy as np

from homr.bar_line_detection import prepare_bar_line_image
from homr.bounding_boxes import (
    BoundingEllipse,
    RotatedBoundingBox,
    create_bounding_ellipses,
    create_rotated_bounding_boxes,
)
from homr.debug import Debug
from homr.model import InputPredictions
from homr.segmentation.inference_segnet import extract
from homr.simple_logging import eprint
from homr.type_definitions import NDArray


class PredictedSymbols:
    def __init__(
        self,
        noteheads: list[BoundingEllipse],
        staff_fragments: list[RotatedBoundingBox],
        clefs_keys: list[RotatedBoundingBox],
        stems_rest: list[RotatedBoundingBox],
        bar_lines: list[RotatedBoundingBox],
    ) -> None:
        self.noteheads = noteheads
        self.staff_fragments = staff_fragments
        self.clefs_keys = clefs_keys
        self.stems_rest = stems_rest
        self.bar_lines = bar_lines


def get_predictions(
    original: NDArray,
    preprocessed: NDArray,
    img_path: str,
    enable_cache: bool,
    segnet_use_gpu: bool,
) -> InputPredictions:
    result = extract(
        preprocessed,
        img_path,
        step_size=320,
        use_cache=enable_cache,
        use_gpu_inference=segnet_use_gpu,
    )
    original_image = cv2.resize(original, (result.staff.shape[1], result.staff.shape[0]))
    preprocessed_image = cv2.resize(preprocessed, (result.staff.shape[1], result.staff.shape[0]))
    return InputPredictions(
        original=original_image,
        preprocessed=preprocessed_image,
        notehead=result.notehead.astype(np.uint8),
        symbols=result.symbols.astype(np.uint8),
        staff=result.staff.astype(np.uint8),
        clefs_keys=result.clefs_keys.astype(np.uint8),
        stems_rest=result.stems_rests.astype(np.uint8),
    )


def predict_symbols(debug: Debug, predictions: InputPredictions) -> PredictedSymbols:
    eprint("Creating bounds for noteheads")
    noteheads = create_bounding_ellipses(predictions.notehead, min_size=(4, 4))
    eprint("Creating bounds for staff_fragments")
    staff_fragments = create_rotated_bounding_boxes(
        predictions.staff, skip_merging=True, min_size=(5, 1), max_size=(10000, 100)
    )

    eprint("Creating bounds for clefs_keys")
    clefs_keys = create_rotated_bounding_boxes(
        predictions.clefs_keys, min_size=(20, 40), max_size=(1000, 1000)
    )
    eprint("Creating bounds for stems_rest")
    stems_rest = create_rotated_bounding_boxes(predictions.stems_rest)
    eprint("Creating bounds for bar_lines")
    bar_line_img = prepare_bar_line_image(predictions.stems_rest)
    debug.write_threshold_image("bar_line_img", bar_line_img)
    bar_lines = create_rotated_bounding_boxes(bar_line_img, skip_merging=True, min_size=(1, 5))

    return PredictedSymbols(noteheads, staff_fragments, clefs_keys, stems_rest, bar_lines)
