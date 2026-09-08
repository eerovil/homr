from copy import deepcopy
from threading import Lock

from homr.model import Staff
from homr.transformer.configs import Config
from homr.transformer.staff2score import Staff2Score
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray

# Keep only the latest configuration, not an unbounded cache of model sessions.
inference: Staff2Score | None = None
_inference_key: tuple | None = None
_inference_lock = Lock()


def _configuration_key(config: Config) -> tuple:
    """Values consumed by Staff2Score, Encoder and ScoreDecoder at inference time.

    Config.to_dict() is a training export: it omits providers, constraints and
    several vocabularies/model paths. Object identity also misses in-place edits.
    Keep this key aligned with those inference classes when adding new settings.
    """
    return (
        config.use_gpu_inference,
        config.use_coreml_encoder,
        config.filepaths.encoder_path,
        config.filepaths.encoder_path_fp16,
        config.filepaths.decoder_path,
        config.filepaths.decoder_path_fp16,
        config.filepaths.rhythmtokenizer,
        config.max_seq_len,
        config.eos_token,
        config.decoder_heads,
        config.decoder_dim,
        config.decoder_depth,
        config.record_confidence,
        frozenset(config.forbidden_rhythm_tokens),
        tuple(sorted(config.rhythm_vocab.items())),
        tuple(sorted(config.pitch_vocab.items())),
        tuple(sorted(config.lift_vocab.items())),
        tuple(sorted(config.position_vocab.items())),
        tuple(sorted(config.articulation_vocab.items())),
        tuple(sorted(config.slur_vocab.items())),
    )


def parse_staff_tromr(staff: Staff, staff_image: NDArray, config: Config) -> list[EncodedSymbol]:
    return predict_best(staff_image, staff=staff, config=config)


def predict_best(org_image: NDArray, staff: Staff, config: Config) -> list[EncodedSymbol]:
    global inference, _inference_key  # noqa: PLW0603
    # ONNX encoder/decoder I/O bindings are mutable. Serialize initialization and
    # prediction together, not just cache lookup. A failed call releases the lock.
    with _inference_lock:
        snapshot = deepcopy(config)
        key = _configuration_key(snapshot)
        if inference is None or _inference_key != key:
            # Snapshot first, publish only after successful construction. Caller
            # edits cannot mutate an existing model's retained configuration.
            inference = Staff2Score(snapshot)
            _inference_key = key
        result = inference.predict(org_image)
    if staff.is_grandstaff:
        return result
    return [r for r in result if r.position != "lower"]
