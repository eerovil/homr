# ruff: noqa: S101

"""Configuration-aware transformer reuse; no ONNX sessions or weights are loaded."""

from threading import Lock
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest

from homr import staff_parsing_tromr as parsing
from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol


@pytest.fixture
def models(monkeypatch: pytest.MonkeyPatch) -> Mock:
    # Reset the process cache for each test, including the original implementation.
    monkeypatch.setattr(parsing, "inference", None)
    monkeypatch.setattr(parsing, "_inference_key", None, raising=False)
    monkeypatch.setattr(parsing, "_inference_lock", Lock(), raising=False)

    def construct(config: Config) -> Mock:
        return Mock(config=config, predict=Mock(return_value=[]))

    factory = Mock(side_effect=construct)
    monkeypatch.setattr(parsing, "Staff2Score", factory)
    return factory


def cached_model() -> Mock:
    assert parsing.inference is not None
    return cast(Mock, parsing.inference)


def read(config: Config, grandstaff: bool = True) -> list[EncodedSymbol]:
    return parsing.parse_staff_tromr(
        Mock(is_grandstaff=grandstaff), np.zeros((1, 1), dtype=np.uint8), config
    )


def test_equivalent_configs_reuse_one_model_across_staffs(models: Mock) -> None:
    config = Config()
    read(config)
    first = cached_model()
    read(config)
    read(Config())
    assert models.call_count == 1
    assert parsing.inference is first
    assert first.predict.call_count == 3


@pytest.mark.parametrize(
    "setting",
    [
        "use_gpu_inference",
        "use_coreml_encoder",
        "filepaths.encoder_path",
        "filepaths.encoder_path_fp16",
        "filepaths.decoder_path",
        "filepaths.decoder_path_fp16",
        "filepaths.rhythmtokenizer",
        "max_seq_len",
        "eos_token",
        "decoder_heads",
        "decoder_dim",
        "decoder_depth",
        "record_confidence",
        "forbidden_rhythm_tokens",
        "rhythm_vocab",
        "pitch_vocab",
        "lift_vocab",
        "position_vocab",
        "articulation_vocab",
        "slur_vocab",
    ],
)
def test_inference_setting_edits_replace_the_model(models: Mock, setting: str) -> None:
    config = Config()
    read(config)
    first = cached_model()
    target = config.filepaths if setting.startswith("filepaths.") else config
    name = setting.rsplit(".", 1)[-1]
    value = getattr(target, name)
    if isinstance(value, bool):
        setattr(target, name, not value)
    elif isinstance(value, int):
        setattr(target, name, value + 1)
    elif isinstance(value, str):
        setattr(target, name, value + ".changed")
    elif isinstance(value, set):
        value.add(config.rhythm_vocab["note_4"])
    else:
        # Modify an existing mapping entry, not only its size or identity.
        key = next(iter(value))
        value[key] += 1
    read(config)
    assert models.call_count == 2, setting
    assert parsing.inference is not first


def test_different_request_constraints_do_not_leak_or_stick(models: Mock) -> None:
    constrained, ordinary = Config(), Config()
    forbidden = constrained.rhythm_vocab["note_4"]
    constrained.forbidden_rhythm_tokens.add(forbidden)
    read(constrained)
    assert cached_model().config.forbidden_rhythm_tokens == {forbidden}
    read(ordinary)
    assert cached_model().config.forbidden_rhythm_tokens == set()
    read(constrained)
    assert cached_model().config.forbidden_rhythm_tokens == {forbidden}
    # Only the most recent model is cached, not one per possible score setting.
    assert models.call_count == 3


def test_caller_edits_cannot_mutate_the_live_model_config(models: Mock) -> None:
    config = Config()
    read(config)
    saved = models.call_args.args[0]
    assert saved is not config
    config.forbidden_rhythm_tokens.add(config.rhythm_vocab["note_4"])
    config.pitch_vocab["C4"] += 1
    config.filepaths.encoder_path = "different.onnx"
    assert saved.forbidden_rhythm_tokens == set()
    assert saved.pitch_vocab["C4"] != config.pitch_vocab["C4"]
    assert saved.filepaths.encoder_path != config.filepaths.encoder_path


def test_mapping_order_does_not_force_a_reload(models: Mock) -> None:
    config = Config()
    read(config)
    config.pitch_vocab = dict(reversed(list(config.pitch_vocab.items())))
    read(config)
    assert models.call_count == 1


def test_non_inference_settings_do_not_force_a_reload(models: Mock) -> None:
    config = Config()
    read(config)
    config.use_stem_voice_hints = not config.use_stem_voice_hints
    config.scheduled_sampling_start_prob = 0.5
    read(config)
    assert models.call_count == 1


@pytest.mark.parametrize("grandstaff", [False, True])
def test_staff_filtering_is_unchanged(models: Mock, grandstaff: bool) -> None:
    config = Config()
    read(config)
    symbols = [
        EncodedSymbol("note_4", "C4", position="upper"),
        EncodedSymbol("note_4", "C3", position="lower"),
    ]
    cached_model().predict.return_value = symbols
    result = read(config, grandstaff)
    assert result == (symbols if grandstaff else symbols[:1])
    assert models.call_count == 1


def test_failed_replacement_does_not_publish_a_wrong_cache_key(models: Mock) -> None:
    original, changed = Config(), Config()
    changed.record_confidence = not original.record_confidence
    read(original)
    previous = parsing.inference
    models.side_effect = RuntimeError("construction failed")
    with pytest.raises(RuntimeError, match="construction failed"):
        read(changed)
    assert parsing.inference is previous
    # The previous model must remain usable with its own settings.
    read(original)
    models.side_effect = lambda config: Mock(config=config, predict=Mock(return_value=[]))
    read(changed)
    assert models.call_count == 3
    assert cached_model().config.record_confidence == changed.record_confidence


def test_initialization_and_prediction_share_one_lock(models: Mock) -> None:
    lock = parsing._inference_lock
    states: list[tuple[str, bool]] = []

    def predict(image: np.ndarray) -> list:
        states.append(("predict", lock.locked()))
        return []

    def construct(config: Config) -> Mock:
        states.append(("construct", lock.locked()))
        return Mock(config=config, predict=Mock(side_effect=predict))

    models.side_effect = construct
    read(Config())
    read(Config())
    assert states == [("construct", True), ("predict", True), ("predict", True)]
    assert not lock.locked()


def test_prediction_failure_releases_lock_and_preserves_the_error(models: Mock) -> None:
    config = Config()
    read(config)
    error = RuntimeError("prediction failed")
    cached_model().predict.side_effect = error
    with pytest.raises(RuntimeError, match="prediction failed") as caught:
        read(config)
    assert caught.value is error
    assert not parsing._inference_lock.locked()
    cached_model().predict.side_effect = None
    assert read(config) == []
    assert models.call_count == 1
