# ruff: noqa: S101

"""Completion is an EOS, not exhaustion of the decoder's token budget."""

from unittest.mock import Mock

import numpy as np
import pytest

from homr.errors import IncompleteRecognitionError
from homr.transformer.decoder_inference import ScoreDecoder


def make_decoder(tokens: list[int], limit: int) -> tuple[ScoreDecoder, Mock]:
    config = Mock(
        max_seq_len=limit,
        eos_token=2,
        rhythm_vocab={"PAD": 0, "BOS": 1, "EOS": 2, "note_4": 3},
        pitch_vocab={"C4": 0},
        lift_vocab={"_": 0},
        articulation_vocab={"_": 0},
        slur_vocab={"_": 0},
        position_vocab={"upper": 0},
        forbidden_rhythm_tokens=set(),
        record_confidence=False,
        decoder_heads=1,
        decoder_dim=1,
        decoder_depth=0,
    )
    session = Mock()
    outputs = []
    for token in tokens:
        rhythm = np.zeros((1, 1, 4), dtype=np.float32)
        rhythm[0, 0, token] = 1
        heads = [rhythm] + [np.ones((1, 1, 1), dtype=np.float32) for _ in range(5)]
        heads.append(np.array([10.0, 20.0], dtype=np.float32))
        outputs.append([Mock(numpy=Mock(return_value=head)) for head in heads])
    session.io_binding.return_value.get_outputs.side_effect = outputs
    decoder = ScoreDecoder(session, fp16=False, use_gpu=False, config=config)
    return decoder, session


def generate(decoder: ScoreDecoder) -> list:
    return decoder.generate(
        np.array([[1]], dtype=np.int64),
        np.array([[0]], dtype=np.int64),
        context=np.zeros((1, 2, 3), dtype=np.float32),
    )


@pytest.mark.parametrize("eos_step", [0, 1, 2])
def test_eos_completes_the_read_even_on_the_last_allowed_step(eos_step: int) -> None:
    decoder, session = make_decoder([3] * eos_step + [2], limit=3)
    result = generate(decoder)
    assert [symbol.rhythm for symbol in result] == ["note_4"] * eos_step
    assert session.run_with_iobinding.call_count == eos_step + 1


@pytest.mark.parametrize("limit", [0, 1, 3])
def test_budget_exhaustion_never_returns_partial_symbols(limit: int) -> None:
    decoder, session = make_decoder([3] * limit, limit)
    with pytest.raises(IncompleteRecognitionError, match=f"{limit}-step limit") as error:
        generate(decoder)
    assert f"{limit} partial symbols" in str(error.value)
    assert session.run_with_iobinding.call_count == limit


def test_an_eos_after_the_budget_does_not_hide_truncation() -> None:
    decoder, session = make_decoder([3, 3, 2], limit=2)
    with pytest.raises(IncompleteRecognitionError):
        generate(decoder)
    assert session.run_with_iobinding.call_count == 2
