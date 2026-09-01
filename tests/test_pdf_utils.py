# ruff: noqa: S101

import pathlib
from collections.abc import Iterator

import cv2
import numpy as np
import pytest
from PIL import Image

from homr.pdf_utils import _pad_to_width, _vstack_pages, render_pdf_to_image


def test_pad_to_width_centres_a_narrow_page_on_white() -> None:
    page = np.zeros((2, 2, 3), dtype=np.uint8)

    padded = _pad_to_width(page, 5)

    assert padded.shape == (2, 5, 3)
    assert np.all(padded[:, 0] == 255)
    assert np.all(padded[:, 1:3] == 0)
    assert np.all(padded[:, 3:] == 255)


def test_vstack_pages_preserves_order_and_centres_different_widths() -> None:
    first = np.full((1, 2, 3), 10, dtype=np.uint8)
    second = np.full((2, 4, 3), 20, dtype=np.uint8)

    stacked = _vstack_pages([first, second])

    assert stacked.shape == (3, 4, 3)
    assert np.all(stacked[0, 1:3] == 10)
    assert np.all(stacked[1:] == 20)


def test_render_pdf_uses_300_dpi_colour_pages_in_document_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    pdf_path = tmp_path / "score.pdf"
    pdf_path.write_bytes(b"%PDF")
    render_scales: list[float] = []

    class FakeBitmap:
        def __init__(self, colour: tuple[int, int, int]) -> None:
            self.colour = colour

        def to_pil(self) -> Image.Image:
            return Image.new("RGB", (2, 1), self.colour)

    class FakePage:
        def __init__(self, colour: tuple[int, int, int]) -> None:
            self.colour = colour

        def render(self, scale: float) -> FakeBitmap:
            render_scales.append(scale)
            return FakeBitmap(self.colour)

    class FakePdf:
        def __init__(self, path: str) -> None:
            assert path == str(pdf_path)
            self.closed = False

        def __bool__(self) -> bool:
            return True

        def __iter__(self) -> Iterator[FakePage]:
            return iter([FakePage((255, 0, 0)), FakePage((0, 255, 0))])

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("homr.pdf_utils.pdfium.PdfDocument", FakePdf)
    monkeypatch.setattr("homr.pdf_utils.autocrop", lambda page: page)

    render_pdf_to_image(str(pdf_path))

    output = cv2.imread(str(tmp_path / "score.png"))
    assert render_scales == [300 / 72.0, 300 / 72.0]
    assert output.shape == (2, 2, 3)
    assert np.all(output[0] == (0, 0, 255))
    assert np.all(output[1] == (0, 255, 0))
