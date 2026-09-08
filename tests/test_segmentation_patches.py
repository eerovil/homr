# ruff: noqa: S101

"""Blend class evidence, never class IDs; no model runtime or weights needed."""

import numpy as np
import pytest

from homr.segmentation.patches import merge_patches


def logits_for_labels(labels: np.ndarray) -> np.ndarray:
    scores = np.full((6, *labels.shape), -10, dtype=np.float32)
    np.put_along_axis(scores, labels[None, ...], 10, axis=0)
    return scores


def test_overlap_never_averages_class_ids() -> None:
    tiles = [logits_for_labels(np.full((2, 2), label)) for label in (2, 4)]
    np.testing.assert_array_equal(merge_patches(tiles, (2, 3), 2, 2), [[2, 2, 4]] * 2)


def test_shared_runner_up_survives_disagreeing_tile_winners() -> None:
    # Background and notehead win their tiles, but both strongly support a stem.
    probabilities = ([0.51, 0.48, 0.01], [0.01, 0.48, 0.51])
    tiles = [np.broadcast_to(np.log(p)[:, None, None], (3, 2, 2)) for p in probabilities]
    np.testing.assert_array_equal(merge_patches(tiles, (2, 3), 2, 2), [[0, 1, 2]] * 2)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_large_logits_are_stable_and_inputs_are_not_mutated(dtype: type) -> None:
    tiles = [np.full((3, 2, 2), 30000, dtype=dtype) for _ in range(2)]
    tiles[0][1] = 31000
    tiles[1][1] = 32000
    saved = [tile.copy() for tile in tiles]
    np.testing.assert_array_equal(merge_patches(tiles, (2, 3), 2, 2), np.ones((2, 3)))
    for before, after in zip(saved, tiles, strict=True):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1), (2, 3), (3, 5), (6, 8)])
@pytest.mark.parametrize("step_size", [1, 2, 4])
def test_matching_tiles_reconstruct_every_pixel_without_padding(
    shape: tuple[int, int], step_size: int
) -> None:
    height, width = shape
    win_size = 4
    expected = (np.arange(height * width).reshape(shape) % 6).astype(np.uint8)
    patches = []
    for iy in range(0, height, step_size):
        y0 = max(0, min(iy, height - win_size))
        for ix in range(0, width, step_size):
            x0 = max(0, min(ix, width - win_size))
            source = expected[y0 : y0 + win_size, x0 : x0 + win_size]
            labels = np.full((win_size, win_size), 5, dtype=np.uint8)
            labels[: source.shape[0], : source.shape[1]] = source
            patches.append(logits_for_labels(labels))
    np.testing.assert_array_equal(merge_patches(patches, shape, win_size, step_size), expected)


def test_empty_patch_list_has_a_clear_error() -> None:
    with pytest.raises(ValueError, match="empty list"):
        merge_patches([], (2, 2), 2, 2)


def test_argmax_label_maps_are_not_accepted_as_logits() -> None:
    with pytest.raises(ValueError, match="class logits"):
        merge_patches([np.ones((2, 2))], (2, 2), 2, 2)


def test_extra_tiles_are_not_silently_ignored() -> None:
    with pytest.raises(ValueError, match="patch count"):
        merge_patches([np.zeros((6, 2, 2))] * 2, (2, 2), 2, 2)
