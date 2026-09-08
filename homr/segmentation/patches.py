"""Stitch segmentation labels without treating class IDs as numeric measurements."""

import numpy as np

from homr.type_definitions import NDArray


def merge_patches(
    patches: list[NDArray], image_shape: tuple[int, int], win_size: int, step_size: int
) -> NDArray:
    """Vote per class in overlaps; ties choose the lowest predicted class ID.

    Inputs are argmax label maps, not probabilities. Averaging labels can invent
    a class neither tile predicted (e.g. notehead 2 + staff 4 becomes clef 3).
    Count one class at a time to avoid a page-sized buffer for every class.
    Padded pixels are excluded, matching extract_patch's top-left placement.
    """
    if not patches:
        raise ValueError("Cannot merge an empty list of segmentation patches")
    reconstructed = np.zeros(image_shape, dtype=patches[0].dtype)
    best_count = np.zeros(image_shape, dtype=np.uint32)
    votes = np.zeros(image_shape, dtype=np.uint32)
    number_of_classes = max(int(patch.max()) for patch in patches) + 1

    for class_id in range(number_of_classes):
        votes.fill(0)
        idx = 0
        for iy in range(0, image_shape[0], step_size):
            y = min(iy, image_shape[0] - win_size)
            y0 = max(y, 0)
            y1 = min(y + win_size, image_shape[0])

            for ix in range(0, image_shape[1], step_size):
                x = min(ix, image_shape[1] - win_size)
                x0 = max(x, 0)
                x1 = min(x + win_size, image_shape[1])

                patch = patches[idx]
                ph = y1 - y0
                pw = x1 - x0
                votes[y0:y1, x0:x1] += patch[:ph, :pw] == class_id
                idx += 1

        wins = votes > best_count
        reconstructed[wins] = class_id
        best_count[wins] = votes[wins]

    return reconstructed
