"""Request-aware segmentation reuse, without loading or mocking package imports."""

import hashlib
import lzma
from pathlib import Path
from threading import Lock
from unittest.mock import Mock

import numpy as np
import pytest

from homr.segmentation import inference_segnet as segnet
from homr.segmentation.config import segmentation_version
from homr.type_definitions import NDArray


@pytest.fixture
def models(monkeypatch: pytest.MonkeyPatch) -> Mock:
    monkeypatch.setattr(segnet, "_segnet_inference", None)
    monkeypatch.setattr(segnet, "_segnet_key", None, raising=False)
    monkeypatch.setattr(segnet, "_segnet_lock", Lock(), raising=False)

    def construct(use_gpu_inference: bool) -> Mock:
        def run(batch: NDArray) -> NDArray:
            output = np.zeros((len(batch), 6, *batch.shape[-2:]), dtype=np.float32)
            output[:, 4 if use_gpu_inference else 2] = 1
            return output

        # A GPU request may fall back to CPU. Cache by the request, not this flag.
        return Mock(use_gpu=False, run=Mock(side_effect=run))

    factory = Mock(side_effect=construct)
    monkeypatch.setattr(segnet, "Segnet", factory)
    return factory


def read(use_gpu: bool, batch_size: int = 2) -> None:
    masks = segnet.inference(np.full((1, 3), 255, dtype=np.uint8), use_gpu, batch_size, 1, 2)
    for index, mask in enumerate(masks):
        # CPU -> noteheads; GPU request -> staff, including the final partial batch.
        expected = int(index == (0 if use_gpu else 3))
        np.testing.assert_array_equal(mask, np.full((1, 3), expected, dtype=np.uint8))


@pytest.mark.parametrize("use_gpu", [False, True])
def test_equivalent_requests_reuse_the_same_model(models: Mock, use_gpu: bool) -> None:
    read(use_gpu)
    previous = segnet._segnet_inference
    # Batch size is a request setting, not a reason to reload model weights.
    read(use_gpu, batch_size=1)
    read(use_gpu, batch_size=8)
    assert segnet._segnet_inference is previous
    models.assert_called_once_with(use_gpu)


@pytest.mark.parametrize("first", [False, True])
def test_switching_mode_never_reuses_the_first_requests_model(models: Mock, first: bool) -> None:
    for mode in (first, not first, first):
        read(mode)
    assert [call.args for call in models.call_args_list] == [(first,), (not first,), (first,)]
    # Only the latest model is retained, rather than a growing pool of sessions.
    read(first)
    assert models.call_count == 3


@pytest.mark.parametrize("path", ["segnet_path_onnx", "segnet_path_onnx_fp16"])
def test_changed_model_paths_invalidate_reuse(
    models: Mock, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    read(False)
    previous = segnet._segnet_inference
    monkeypatch.setattr(segnet, path, getattr(segnet, path) + ".changed")
    read(False)
    assert models.call_count == 2
    assert segnet._segnet_inference is not previous


def test_failed_replacement_keeps_the_previous_model_and_key(models: Mock) -> None:
    read(False)
    previous = segnet._segnet_inference
    key = segnet._segnet_key
    construct = models.side_effect
    error = RuntimeError("model could not load")
    models.side_effect = error
    with pytest.raises(RuntimeError, match="could not load") as caught:
        read(True)
    assert caught.value is error
    assert segnet._segnet_inference is previous
    assert segnet._segnet_key == key
    assert not segnet._segnet_lock.locked()
    read(False)
    models.side_effect = construct
    read(True)
    assert models.call_count == 3


def test_initialization_and_every_batch_share_the_lock(models: Mock) -> None:
    states = []
    construct = models.side_effect

    def locked_construct(use_gpu: bool) -> Mock:
        states.append(("load", segnet._segnet_lock.locked()))
        model = construct(use_gpu)
        run = model.run.side_effect

        def locked_run(batch: NDArray) -> NDArray:
            states.append(("batch", segnet._segnet_lock.locked()))
            return run(batch)

        model.run.side_effect = locked_run
        return model

    models.side_effect = locked_construct
    read(False)  # full batch and tail
    read(False)
    assert states == [("load", True), *[("batch", True)] * 4]
    assert not segnet._segnet_lock.locked()


def test_failed_prediction_releases_lock_and_can_reuse_model(models: Mock) -> None:
    read(False)
    model = segnet._segnet_inference
    assert isinstance(model, Mock)
    run = model.run.side_effect
    error = RuntimeError("inference failed")
    model.run.side_effect = error
    with pytest.raises(RuntimeError, match="inference failed") as caught:
        read(False)
    assert caught.value is error
    assert not segnet._segnet_lock.locked()
    model.run.side_effect = run
    read(False)
    assert models.call_count == 1


@pytest.fixture
def cached_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Mock, Path]:
    def fake_inference(
        image: NDArray, use_gpu_inference: bool, batch_size: int, step_size: int, win_size: int
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        empty = np.zeros(image.shape, dtype=np.uint8)
        full = np.ones(image.shape, dtype=np.uint8)
        return (
            (full, empty, empty, empty, empty)
            if use_gpu_inference
            else (empty, empty, empty, full, empty)
        )

    run = Mock(side_effect=fake_inference)
    monkeypatch.setattr(segnet, "inference", run)
    return run, tmp_path / "page.png"


@pytest.mark.parametrize("first", [False, True])
def test_disk_cache_does_not_hide_a_changed_mode(
    cached_read: tuple[Mock, Path], first: bool
) -> None:
    run, path = cached_read
    image = np.zeros((2, 3), dtype=np.uint8)
    for mode in (first, not first, first):
        for _ in range(2):
            result = segnet.extract(image, str(path), use_cache=True, use_gpu_inference=mode)
            np.testing.assert_array_equal(result.staff, np.full(image.shape, int(mode)))
            np.testing.assert_array_equal(result.notehead, np.full(image.shape, int(not mode)))
    assert [call.kwargs["use_gpu_inference"] for call in run.call_args_list] == [
        first,
        not first,
        first,
    ]


@pytest.mark.parametrize(
    ("batch_size", "step_size", "win_size"), [(2, -1, 320), (8, 2, 320), (8, -1, 2)]
)
def test_disk_cache_identifies_request_parameters(
    cached_read: tuple[Mock, Path], batch_size: int, step_size: int, win_size: int
) -> None:
    run, path = cached_read
    image = np.zeros((2, 3), dtype=np.uint8)
    segnet.extract(image, str(path), use_cache=True)
    segnet.extract(
        image,
        str(path),
        use_cache=True,
        batch_size=batch_size,
        step_size=step_size,
        win_size=win_size,
    )
    assert run.call_count == 2


@pytest.mark.parametrize("setting", ["segnet_path_onnx", "segnet_path_onnx_fp16"])
def test_disk_cache_identifies_model_paths(
    cached_read: tuple[Mock, Path], monkeypatch: pytest.MonkeyPatch, setting: str
) -> None:
    run, path = cached_read
    image = np.zeros((2, 3), dtype=np.uint8)
    segnet.extract(image, str(path), use_cache=True)
    monkeypatch.setattr(segnet, setting, getattr(segnet, setting) + ".changed")
    segnet.extract(image, str(path), use_cache=True)
    assert run.call_count == 2


@pytest.mark.parametrize("change", ["shape", "dtype", "pixels"])
def test_image_metadata_and_pixels_both_participate_in_the_cache(
    cached_read: tuple[Mock, Path], change: str
) -> None:
    run, path = cached_read
    image = np.zeros((2, 3), dtype=np.uint16)
    if change == "shape":
        changed = image.reshape(3, 2)
    elif change == "dtype":
        changed = image.view(np.int16)
    else:
        changed = np.ones_like(image)
    if change != "pixels":
        assert changed.tobytes() == image.tobytes()
    segnet.extract(image, str(path), use_cache=True)
    result = segnet.extract(changed, str(path), use_cache=True)
    assert run.call_count == 2
    assert result.staff.shape == changed.shape


@pytest.mark.parametrize("old_format", ["original", "categorical-vote-v1"])
def test_caches_without_model_request_identity_are_invalidated(
    cached_read: tuple[Mock, Path], old_format: str
) -> None:
    run, path = cached_read
    image = np.zeros((2, 3), dtype=np.uint8)
    marker = segmentation_version
    if old_format != "original":
        marker += f":{old_format}:320:-1"
    with lzma.open(path.with_suffix(".npy"), "wb") as file:
        for _ in range(5):
            np.save(file, np.zeros(image.shape, dtype=np.uint8))
        file.write((hashlib.sha256(image.tobytes()).hexdigest() + "\n" + marker + "\n").encode())
    for _ in range(2):
        result = segnet.extract(image, str(path), use_cache=True)
        np.testing.assert_array_equal(result.staff, np.ones(image.shape, dtype=np.uint8))
    assert run.call_count == 1
