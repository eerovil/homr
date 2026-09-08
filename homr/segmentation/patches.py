"""Blend segmentation class probabilities before choosing a pixel's label."""

import numpy as np


def merge_patches(
    patches: list[np.ndarray], image_shape: tuple[int, int], win_size: int, step_size: int
) -> np.ndarray:
    """Combine class logits, retaining runner-up evidence in overlapping tiles.

    Class IDs are categories, not quantities. Voting on each tile's argmax also
    loses evidence: two tiles can disagree on their winners while both give the
    same runner-up substantial probability. Blend softmax probabilities instead.
    A pixel has the same overlap count for every class, so normalization by that
    count cannot change the final argmax and need not allocate a weight buffer.
    """
    if not patches:
        raise ValueError("Cannot merge an empty list of segmentation patches")
    if patches[0].ndim != 3:
        raise ValueError("Expected class logits with shape (classes, height, width)")
    classes = patches[0].shape[0]
    reconstructed = np.zeros((classes, *image_shape), dtype=np.float32)
    index = 0
    for iy in range(0, image_shape[0], step_size):
        y = min(iy, image_shape[0] - win_size)
        y0 = max(y, 0)
        y1 = min(y + win_size, image_shape[0])
        for ix in range(0, image_shape[1], step_size):
            x = min(ix, image_shape[1] - win_size)
            x0 = max(x, 0)
            x1 = min(x + win_size, image_shape[1])
            patch = patches[index]
            if patch.ndim != 3 or patch.shape[0] != classes:
                raise ValueError("All segmentation patches must use the same class axis")
            # Copy to float32 before subtracting: do not mutate callers or overflow fp16.
            logits = patch[:, : y1 - y0, : x1 - x0].astype(np.float32)
            logits -= logits.max(axis=0, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=0, keepdims=True)
            reconstructed[:, y0:y1, x0:x1] += probabilities
            index += 1
    if index != len(patches):
        raise ValueError("Segmentation patch count does not match the image grid")
    return reconstructed.argmax(axis=0)
