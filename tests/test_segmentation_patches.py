# ruff: noqa: S101

"""Categorical tile stitching; no inference runtime or model weights needed."""

import numpy as np
import pytest

from homr.segmentation.patches import merge_patches


def test_overlap_never_invents_an_intermediate_class() -> None:
    patches = [np.full((2, 2), label, dtype=np.uint8) for label in (2, 4)]
    result = merge_patches(patches, (2, 3), win_size=2, step_size=2)
    np.testing.assert_array_equal(result, [[2, 2, 4], [2, 2, 4]])


def test_ties_are_independent_of_tile_order() -> None:
    patches = [np.full((2, 2), label, dtype=np.uint8) for label in (4, 2)]
    result = merge_patches(patches, (2, 3), win_size=2, step_size=2)
    np.testing.assert_array_equal(result, [[4, 2, 2], [4, 2, 2]])


def test_majority_wins_instead_of_an_average_label() -> None:
    patches = [np.full((2, 2), label, dtype=np.uint8) for label in (2, 4, 4)]
    result = merge_patches(patches, (1, 3), win_size=2, step_size=1)
    np.testing.assert_array_equal(result, [[2, 4, 4]])


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
            # Unused pixels must not vote, even when they predict a real class.
            patch = np.full((win_size, win_size), 5, dtype=np.uint8)
            patch[: source.shape[0], : source.shape[1]] = source
            patches.append(patch)
    actual = merge_patches(patches, shape, win_size, step_size)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.int64])
def test_disjoint_tiles_preserve_labels_and_dtype(dtype: type) -> None:
    patches = [np.full((2, 2), label, dtype=dtype) for label in (2, 4)]
    result = merge_patches(patches, (2, 4), win_size=2, step_size=2)
    np.testing.assert_array_equal(result, [[2, 2, 4, 4], [2, 2, 4, 4]])
    assert result.dtype == dtype


def test_empty_patch_list_has_a_clear_error() -> None:
    with pytest.raises(ValueError, match="empty list"):
        merge_patches([], (2, 2), win_size=2, step_size=2)
