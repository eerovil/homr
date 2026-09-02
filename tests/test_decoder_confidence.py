import numpy as np

from homr.transformer.decoder_inference import confidence_for_logits


def test_confidence_reports_selected_token_alternatives_and_margin():
    report = confidence_for_logits(
        np.array([[0.0, 2.0, 1.0]]), {0: "32nd", 1: "16th", 2: "eighth"}
    )

    assert report["value"] == "16th"
    assert [item["value"] for item in report["alternatives"]] == ["16th", "eighth", "32nd"]
    assert report["probability"] == report["alternatives"][0]["probability"]
    assert report["margin"] == report["alternatives"][0]["probability"] - report["alternatives"][1]["probability"]
