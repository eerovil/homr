"""Opt-in score knowledge that can safely restrict decoder rhythm choices."""

from dataclasses import dataclass
from typing import Any

from homr.transformer.vocabulary import EncodedSymbol


@dataclass(frozen=True)
class RhythmSettings:
    minimum_duration: int | None = None
    allow_tuplets: bool = True
    allow_grace_notes: bool = True
    stem_voice_hints: bool = True

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "RhythmSettings":
        unknown_root = set(data) - {"decoder", "postprocessing"}
        if unknown_root:
            raise ValueError("unknown score setting(s): " + ", ".join(sorted(unknown_root)))
        decoder = data.get("decoder", {})
        if not isinstance(decoder, dict):
            raise ValueError("score settings 'decoder' must be an object")
        unknown = set(decoder) - {"minimum_duration", "allow_tuplets", "allow_grace_notes"}
        if unknown:
            raise ValueError("unknown decoder setting(s): " + ", ".join(sorted(unknown)))
        minimum = decoder.get("minimum_duration")
        if minimum is not None:
            try:
                minimum = int(minimum)
            except (TypeError, ValueError) as error:
                raise ValueError("minimum_duration must be a note denominator such as '16'") from error
            if minimum not in {1, 2, 4, 8, 16, 32, 64, 128}:
                raise ValueError("minimum_duration must be a power-of-two note denominator")
        for name in ("allow_tuplets", "allow_grace_notes"):
            if name in decoder and not isinstance(decoder[name], bool):
                raise ValueError(f"{name} must be true or false")
        postprocessing = data.get("postprocessing", {})
        if not isinstance(postprocessing, dict):
            raise ValueError("score settings 'postprocessing' must be an object")
        unknown = set(postprocessing) - {"stem_voice_hints"}
        if unknown:
            raise ValueError("unknown postprocessing setting(s): " + ", ".join(sorted(unknown)))
        stem_voice_hints = postprocessing.get("stem_voice_hints", True)
        if not isinstance(stem_voice_hints, bool):
            raise ValueError("stem_voice_hints must be true or false")
        return cls(
            minimum_duration=minimum,
            allow_tuplets=decoder.get("allow_tuplets", True),
            allow_grace_notes=decoder.get("allow_grace_notes", True),
            stem_voice_hints=stem_voice_hints,
        )

    def forbidden_rhythm_tokens(self, vocabulary: dict[str, int]) -> set[int]:
        forbidden = set()
        for token, token_id in vocabulary.items():
            if not token.startswith(("note_", "rest_")):
                continue
            kern = token.split("_", 1)[1]
            digits = "".join(character for character in kern if character.isdigit())
            denominator = int(digits) if digits else 4
            symbol = EncodedSymbol(token)
            if not self.allow_grace_notes and "G" in kern:
                forbidden.add(token_id)
            elif not self.allow_tuplets and symbol.is_tuplet():
                forbidden.add(token_id)
            # This is the written base note value: 16 allows dotted 16ths and
            # 16th-note tuplets, but rules out 32nds and smaller values.
            elif (
                self.minimum_duration is not None
                and not symbol.is_tuplet()
                and denominator > self.minimum_duration
            ):
                forbidden.add(token_id)
        return forbidden
