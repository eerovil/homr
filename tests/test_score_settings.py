from homr.transformer.score_settings import RhythmSettings
from homr.transformer.vocabulary import Vocabulary


def test_minimum_duration_rejects_32nds_but_keeps_dotted_16ths_and_tuplets():
    forbidden = RhythmSettings.from_json(
        {"decoder": {"minimum_duration": "16"}}
    ).forbidden_rhythm_tokens(Vocabulary().rhythm)
    vocab = Vocabulary().rhythm

    assert vocab["note_32"] in forbidden
    assert vocab["rest_64"] in forbidden
    assert vocab["note_16."] not in forbidden
    assert vocab["note_24"] not in forbidden


def test_settings_can_forbid_tuplets_and_grace_notes():
    forbidden = RhythmSettings.from_json(
        {"decoder": {"allow_tuplets": False, "allow_grace_notes": False}}
    ).forbidden_rhythm_tokens(Vocabulary().rhythm)
    vocab = Vocabulary().rhythm

    assert vocab["note_12"] in forbidden
    assert vocab["rest_8G"] in forbidden
    assert vocab["note_16"] not in forbidden
