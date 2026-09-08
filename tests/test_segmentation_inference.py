# ruff: noqa: S101

"""Tiling and cache integration with a deterministic fake model, not real weights."""

import hashlib
import lzma
from pathlib import Path

import numpy as np
import pytest

from homr.segmentation import inference_segnet as segnet
from homr.type_definitions import NDArray


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_short_image_uses_the_same_grid_for_inference_and_merging(
    monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    class FakeSegnet:
        seen = 0

        def run(self, batch: NDArray) -> NDArray:
            labels = (2, 4, 4)
            output = np.zeros((len(batch), 6, 2, 2), dtype=np.float32)
            for index in range(len(batch)):
                output[index, labels[self.seen]] = 1
                self.seen += 1
            return output

    model = FakeSegnet()
    monkeypatch.setattr(segnet, "_segnet_inference", model)
    staff, symbols, rests, heads, clefs = segnet.inference(
        np.full((1, 3), 255, dtype=np.uint8), False, batch_size, 1, 2
    )
    assert model.seen == 3
    np.testing.assert_array_equal(staff, [[0, 1, 1]])
    np.testing.assert_array_equal(heads, [[1, 0, 0]])
    for mask in (symbols, rests, clefs):
        np.testing.assert_array_equal(mask, np.zeros((1, 3), dtype=np.uint8))


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_inference_retains_runner_up_scores_through_full_and_partial_batches(
    monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    class FakeSegnet:
        seen = 0

        def run(self, batch: NDArray) -> NDArray:
            probabilities = ([0.51, 0.48, 0.01], [0.01, 0.48, 0.51])
            output = []
            for _ in batch:
                logits = np.log(probabilities[self.seen])[:, None, None]
                output.append(np.broadcast_to(logits, (3, 2, 2)))
                self.seen += 1
            return np.stack(output)

    model = FakeSegnet()
    monkeypatch.setattr(segnet, "_segnet_inference", model)
    _, _, stems, heads, _ = segnet.inference(
        np.full((2, 3), 255, dtype=np.uint8), False, batch_size, 2, 2
    )
    assert model.seen == 2
    np.testing.assert_array_equal(stems, [[0, 1, 0]] * 2)
    np.testing.assert_array_equal(heads, [[0, 0, 1]] * 2)


@pytest.mark.parametrize("old_algorithm", ["legacy", "categorical-vote-v1"])
def test_old_cache_is_replaced_and_new_cache_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old_algorithm: str
) -> None:
    image = np.full((2, 3), 255, dtype=np.uint8)
    image_path = tmp_path / "page.png"
    cache_path = image_path.with_suffix(".npy")
    empty = np.zeros(image.shape, dtype=np.uint8)
    marker = segnet.segmentation_version
    if old_algorithm != "legacy":
        marker += f":{old_algorithm}:320:-1"
    with lzma.open(cache_path, "wb") as cache:
        for _ in range(5):
            np.save(cache, empty)
        cache.write((hashlib.sha256(image.tobytes()).hexdigest() + "\n").encode())
        cache.write((marker + "\n").encode())

    calls = 0

    def fake_inference(
        original: NDArray,
        use_gpu_inference: bool,
        batch_size: int,
        step_size: int,
        win_size: int,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        nonlocal calls
        calls += 1
        return np.ones(original.shape, dtype=np.uint8), empty, empty, empty, empty

    monkeypatch.setattr(segnet, "inference", fake_inference)
    first = segnet.extract(image, str(image_path), use_cache=True)
    second = segnet.extract(image, str(image_path), use_cache=True)
    assert calls == 1
    np.testing.assert_array_equal(first.staff, np.ones(image.shape, dtype=np.uint8))
    np.testing.assert_array_equal(second.staff, first.staff)
    np.testing.assert_array_equal(second.notehead, empty)
    segnet.extract(image, str(image_path), use_cache=True, step_size=160)
    assert calls == 2
