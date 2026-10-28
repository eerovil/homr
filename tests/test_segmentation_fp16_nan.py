"""A GPU that gives NaN for the fp16 segmentation model falls back to the fp32 one."""

from unittest.mock import Mock

import numpy as np
import pytest

from homr import download_utils
from homr.segmentation import inference_segnet as segnet
from homr.segmentation.config import segnet_path_onnx, segnet_path_onnx_fp16


class FakeSession:
    """An ONNX session whose output is NaN for the fp16 model and ones for fp32."""

    def __init__(self, path: str, providers: list) -> None:
        self.path = path
        self.providers = providers
        self.binding = Mock()

        def outputs() -> list:
            value = np.nan if path == segnet_path_onnx_fp16 else 1.0
            return [Mock(numpy=Mock(return_value=np.full((1, 6, 2, 2), value, np.float32)))]

        self.binding.get_outputs.side_effect = outputs

    def io_binding(self) -> Mock:
        return self.binding

    def run_with_iobinding(self, binding: Mock) -> None:
        pass


def gpu_segnet(providers: list | None) -> segnet.Segnet:
    model = segnet.Segnet.__new__(segnet.Segnet)
    model.use_gpu = providers is not None
    model.providers = providers
    model.model = FakeSession(segnet_path_onnx_fp16, providers or [])
    model.fp16 = True
    model.io_binding = model.model.io_binding()
    return model


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch) -> Mock:
    monkeypatch.setattr("homr.segmentation.inference_segnet.ort.InferenceSession", FakeSession)
    download = Mock()
    monkeypatch.setattr(download_utils, "download_models", download)
    return download


def test_nan_on_the_gpu_switches_to_fp32_on_the_same_providers(downloads: Mock) -> None:
    model = gpu_segnet(["CUDAExecutionProvider"])
    out = model.run(np.zeros((1, 3, 2, 2), np.float32))
    assert not np.isnan(out).any()
    assert model.fp16 is False
    assert model.model.path == segnet_path_onnx
    assert model.model.providers == ["CUDAExecutionProvider"]
    downloads.assert_called_once_with([segnet_path_onnx])
    # The fp32 input is bound from then on.
    assert model.io_binding.bind_cpu_input.call_args.args[1].dtype == np.float32


def test_nan_on_the_cpu_is_returned_as_it_is(downloads: Mock) -> None:
    model = gpu_segnet(None)
    out = model.run(np.zeros((1, 3, 2, 2), np.float32))
    assert np.isnan(out).all()
    assert model.fp16 is True
    downloads.assert_not_called()
