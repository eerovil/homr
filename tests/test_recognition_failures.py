# ruff: noqa: S101

"""Incomplete reads must be visible through assembly, output, and the CLI.

No weights are loaded: inference, geometry preparation, and downloads are mocked.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import onnxruntime as ort
import pytest

from homr import main as cli
from homr import reread
from homr import staff_parsing as parsing
from homr.errors import IncompleteRecognitionError
from homr.model import MultiStaff
from homr.transformer.vocabulary import EncodedSymbol


@pytest.mark.parametrize("empty_index", [0, 1])
def test_an_empty_requested_staff_is_not_silently_omitted(
    monkeypatch: pytest.MonkeyPatch, empty_index: int
) -> None:
    rows: list[MultiStaff] = [Mock(staffs=[Mock()]), Mock(staffs=[Mock()])]
    monkeypatch.setattr(parsing, "_ensure_same_number_of_staffs", lambda value: value)
    monkeypatch.setattr(parsing, "StaffRegions", Mock())
    readings = [[EncodedSymbol("note_4", "C4")], [EncodedSymbol("note_4", "D4")]]
    readings[empty_index] = []
    read = Mock(side_effect=readings)
    monkeypatch.setattr(parsing, "parse_staff_image", read)
    monkeypatch.setattr(parsing, "_reread_if_doubtful", lambda *args: args[3])
    monkeypatch.setattr(parsing, "remove_duplicated_symbols", lambda value: value)
    with pytest.raises(IncompleteRecognitionError, match=f"Staff {empty_index}: no symbols"):
        parsing.parse_staffs(Mock(), rows, np.zeros((1, 1)), Mock())
    assert read.call_count == empty_index + 1


def test_explicitly_unselected_staffs_are_still_not_read(monkeypatch: pytest.MonkeyPatch) -> None:
    rows: list[MultiStaff] = [Mock(staffs=[Mock()]), Mock(staffs=[Mock()])]
    monkeypatch.setattr(parsing, "_ensure_same_number_of_staffs", lambda value: value)
    monkeypatch.setattr(parsing, "StaffRegions", Mock())
    read = Mock(return_value=[EncodedSymbol("note_4", "C4")])
    monkeypatch.setattr(parsing, "parse_staff_image", read)
    monkeypatch.setattr(parsing, "_reread_if_doubtful", lambda *args: args[3])
    monkeypatch.setattr(parsing, "remove_duplicated_symbols", lambda value: value)
    result = parsing.parse_staffs(Mock(), rows, np.zeros((1, 1)), Mock(), selected_staff=0)
    assert read.call_count == 1
    assert [symbol.rhythm for symbol in result[0]] == ["note_4", "newline"]


def test_truncation_identifies_the_staff_and_keeps_the_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, staff = np.zeros((1, 1)), Mock()
    monkeypatch.setattr(parsing, "prepare_staff_image", Mock(return_value=(image, staff)))
    original = IncompleteRecognitionError("decoder exhausted")
    monkeypatch.setattr(parsing, "parse_staff_tromr", Mock(side_effect=original))
    with pytest.raises(IncompleteRecognitionError, match="Staff 7: decoder exhausted") as error:
        parsing.parse_staff_image(Mock(), 7, staff, image, Mock(), Mock())
    assert error.value.__cause__ is original


@pytest.mark.parametrize("failed_half", [0, 1])
def test_incomplete_optional_reread_keeps_the_complete_fused_reading(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, failed_half: int
) -> None:
    staff = Mock(merged_from=(Mock(), Mock()))
    fused = [EncodedSymbol("note_4", "C4")]
    monkeypatch.setattr(parsing, "StaffRegions", Mock())
    monkeypatch.setattr(parsing, "MultiStaff", Mock())
    monkeypatch.setattr(reread, "doubtful", Mock(return_value=True))
    attempts: list[object] = [fused] * failed_half
    attempts.append(IncompleteRecognitionError("decoder exhausted"))
    read = Mock(side_effect=attempts)
    monkeypatch.setattr(parsing, "parse_staff_image", read)
    splice = Mock()
    monkeypatch.setattr(reread, "splice", splice)
    result = parsing._reread_if_doubtful(Mock(), 0, staff, fused, np.zeros((1, 1)), Mock())
    assert result is fused
    assert read.call_count == failed_half + 1
    splice.assert_not_called()
    assert "optional re-read was incomplete" in capsys.readouterr().err


def test_optional_reread_does_not_swallow_unrelated_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parsing, "StaffRegions", Mock())
    monkeypatch.setattr(parsing, "MultiStaff", Mock())
    monkeypatch.setattr(reread, "doubtful", Mock(return_value=True))
    monkeypatch.setattr(parsing, "parse_staff_image", Mock(side_effect=ValueError("unrelated")))
    with pytest.raises(ValueError, match="unrelated"):
        parsing._reread_if_doubtful(
            Mock(), 0, Mock(merged_from=(Mock(), Mock())), [], np.zeros((1, 1)), Mock()
        )


def test_complete_rereads_still_use_the_existing_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    upper: list[EncodedSymbol] = [Mock()]
    lower: list[EncodedSymbol] = [Mock()]
    fused: list[EncodedSymbol] = [Mock()]
    chosen: list[EncodedSymbol] = [Mock()]
    monkeypatch.setattr(parsing, "StaffRegions", Mock())
    monkeypatch.setattr(parsing, "MultiStaff", Mock())
    monkeypatch.setattr(reread, "doubtful", Mock(return_value=True))
    monkeypatch.setattr(parsing, "parse_staff_image", Mock(side_effect=[upper, lower]))
    splice, better = Mock(return_value=chosen), Mock(return_value=(chosen, True))
    monkeypatch.setattr(reread, "splice", splice)
    monkeypatch.setattr(reread, "better_of", better)
    result = parsing._reread_if_doubtful(
        Mock(), 0, Mock(merged_from=(Mock(), Mock())), fused, np.zeros((1, 1)), Mock()
    )
    assert result is chosen
    splice.assert_called_once_with(upper, lower)
    better.assert_called_once_with(fused, chosen)


@pytest.mark.parametrize("previous_output", [False, True])
def test_incomplete_page_never_reaches_xml_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, previous_output: bool
) -> None:
    image_path = tmp_path / "page.png"
    output = image_path.with_suffix(".musicxml")
    if previous_output:
        output.write_text("old output")
    debug = Mock()
    detection: tuple = ([], np.zeros((1, 1)), debug, Mock(), [])
    monkeypatch.setattr(cli, "detect_staffs_in_image", Mock(return_value=detection))
    monkeypatch.setattr(
        cli, "parse_staffs", Mock(side_effect=IncompleteRecognitionError("partial"))
    )
    writer = Mock()
    monkeypatch.setattr(cli, "generate_xml", writer)
    config = Mock(read_staff_positions=False, score_settings=None)
    with pytest.raises(IncompleteRecognitionError, match="partial"):
        cli.process_image(str(image_path), config, Mock())
    writer.assert_not_called()
    assert not output.exists()
    debug.clean_debug_files_from_previous_runs.assert_called_once()


@pytest.fixture
def no_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "download_weights", Mock())
    monkeypatch.setattr(cli, "download_ocr_weights", Mock())
    monkeypatch.setattr(ort, "set_default_logger_severity", Mock())


def test_single_file_incompleteness_exits_nonzero_with_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    no_downloads: None,
) -> None:
    image = tmp_path / "page.png"
    image.touch()
    monkeypatch.setattr(sys, "argv", ["homr", str(image), "--gpu", "no"])
    monkeypatch.setattr(
        cli, "process_image", Mock(side_effect=IncompleteRecognitionError("partial"))
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 1
    stderr = capsys.readouterr().err
    assert str(image) in stderr
    assert "Incomplete recognition" in stderr
    assert "partial" in stderr


@pytest.mark.parametrize("error_type", [IncompleteRecognitionError, RuntimeError])
def test_batch_attempts_remaining_files_but_exits_nonzero_after_any_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    no_downloads: None,
    error_type: type[Exception],
) -> None:
    bad, good = tmp_path / "a.png", tmp_path / "b.png"
    bad.touch()
    good.touch()
    monkeypatch.setattr(sys, "argv", ["homr", str(tmp_path), "--gpu", "no"])
    process = Mock(side_effect=[error_type("failed reading"), None])
    monkeypatch.setattr(cli, "process_image", process)
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 1
    assert [call.args[0] for call in process.call_args_list] == [str(bad), str(good)]
    assert "Errors occurred while processing" in capsys.readouterr().err


@pytest.mark.parametrize("directory", [False, True])
def test_successful_cli_runs_still_succeed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_downloads: None, directory: bool
) -> None:
    image = tmp_path / "page.png"
    image.touch()
    path = tmp_path if directory else image
    monkeypatch.setattr(sys, "argv", ["homr", str(path), "--gpu", "no"])
    process = Mock()
    monkeypatch.setattr(cli, "process_image", process)
    cli.main()  # Success returns normally; failures raise SystemExit.
    process.assert_called_once()


def test_invalid_input_keeps_its_existing_exit_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_downloads: None
) -> None:
    image = tmp_path / "page.png"
    image.touch()
    monkeypatch.setattr(sys, "argv", ["homr", str(image), "--gpu", "no"])
    monkeypatch.setattr(
        cli, "process_image", Mock(side_effect=cli.InvalidProgramArgumentException())
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2
