# ruff: noqa: S101, S314

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import Mock

import pytest

from homr import main
from homr.transformer.vocabulary import EncodedSymbol


def symbol(name: str) -> EncodedSymbol:
    return EncodedSymbol(name)


def page(part_names: list[str], lower_parts: set[int] | None = None) -> list[list[EncodedSymbol]]:
    lower_parts = lower_parts or set()
    return [
        [
            EncodedSymbol("note_4", position="lower" if index in lower_parts else "upper"),
            EncodedSymbol("barline"),
            EncodedSymbol("newline"),
        ]
        for index, _name in enumerate(part_names)
    ]


def test_combine_pages_keeps_part_identity_and_page_order() -> None:
    pages = [
        [
            [
                EncodedSymbol("note_4", position="upper"),
                symbol("barline"),
                symbol("newline"),
            ],
            [
                EncodedSymbol("rest_4", position="upper"),
                symbol("barline"),
                symbol("newline"),
            ],
        ],
        [
            [EncodedSymbol("note_8", position="upper"), symbol("barline")],
            [EncodedSymbol("rest_8", position="upper"), symbol("barline")],
        ],
    ]

    combined = main.combine_page_staffs(pages, ["page-1.png", "page-2.png"])

    assert [[item.rhythm for item in part] for part in combined] == [
        ["note_4", "barline", "pagebreak", "note_8", "barline"],
        ["rest_4", "barline", "pagebreak", "rest_8", "barline"],
    ]


def test_combine_pages_refuses_a_part_count_mismatch() -> None:
    pages = [
        page(["page1-part1", "page1-part2"]),
        [[symbol("page2-part1")]],
    ]

    with pytest.raises(main.InvalidProgramArgumentException, match="Page 2 .* produced 1 parts"):
        main.combine_page_staffs(pages, ["page-1.png", "page-2.png"])


def test_combine_pages_refuses_a_part_topology_mismatch() -> None:
    pages = [
        page(["page1-part1"], lower_parts={0}),
        page(["page2-part1"]),
    ]

    with pytest.raises(main.InvalidProgramArgumentException, match="part topology"):
        main.combine_page_staffs(pages, ["page-1.png", "page-2.png"])


def test_combine_pages_refuses_an_unclosed_page_boundary() -> None:
    pages = [
        [[EncodedSymbol("note_4", position="upper"), EncodedSymbol("newline")]],
        page(["page2-part1"]),
    ]

    with pytest.raises(main.InvalidProgramArgumentException, match="does not end at a barline"):
        main.combine_page_staffs(pages, ["page-1.png", "page-2.png"])


def test_single_image_keeps_an_unclosed_score_for_legacy_callers() -> None:
    staffs = [[EncodedSymbol("note_4", position="upper"), EncodedSymbol("newline")]]

    assert main.combine_page_staffs([staffs], ["page.png"]) == staffs


def test_combine_pages_refuses_misaligned_parts() -> None:
    pages = [
        [
            [
                EncodedSymbol("note_4", position="upper"),
                EncodedSymbol("barline"),
                EncodedSymbol("newline"),
            ],
            [
                EncodedSymbol("note_4", position="upper"),
                EncodedSymbol("barline"),
                EncodedSymbol("note_4", position="upper"),
                EncodedSymbol("barline"),
                EncodedSymbol("newline"),
            ],
        ],
        page(["page2-part1", "page2-part2"]),
    ]

    with pytest.raises(main.InvalidProgramArgumentException, match="misaligned parts"):
        main.combine_page_staffs(pages, ["page-1.png", "page-2.png"])


def test_combine_pages_refuses_an_empty_part_on_the_final_page() -> None:
    pages = [
        page(["page1-part1", "page1-part2"]),
        [page(["page2-part1"])[0], []],
    ]

    with pytest.raises(main.InvalidProgramArgumentException, match=r"no music for parts \[2\]"):
        main.combine_page_staffs(pages, ["page-1.png", "page-2.png"])


def test_combined_pages_have_one_clock_and_an_explicit_page_break(tmp_path: Path) -> None:
    pages = [
        [
            [
                EncodedSymbol("timeSignature/4"),
                EncodedSymbol("note_4", position="upper"),
                EncodedSymbol("barline"),
                EncodedSymbol("newline"),
            ]
        ],
        [
            [
                EncodedSymbol("keySignature_1"),
                EncodedSymbol("note_8", position="upper"),
                EncodedSymbol("barline"),
                EncodedSymbol("newline"),
            ]
        ],
    ]
    combined = main.combine_page_staffs(pages, ["page-1.png", "page-2.png"])
    output = tmp_path / "combined.musicxml"

    main.generate_xml(main.XmlGeneratorArguments(), combined, "Title").write(str(output))

    root = ET.parse(output).getroot()
    measures = root.findall("./part/measure")
    assert [measure.attrib["number"] for measure in measures] == ["1", "2"]
    divisions = root.findall("./part/measure/attributes/divisions")
    assert len(divisions) == 1
    assert divisions[0].text == "2"
    assert [duration.text for duration in root.findall("./part/measure/note/duration")] == [
        "2",
        "1",
    ]
    assert len(root.findall("./part/measure/attributes/time")) == 1
    assert measures[1].find("./attributes/key/fifths").text == "1"
    assert measures[1].find("./print").attrib["new-page"] == "yes"


def patch_main_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "cuda_available", lambda: False)
    monkeypatch.setattr(main, "coreml_available", lambda: False)
    monkeypatch.setattr(main, "download_weights", lambda *_args: None)
    monkeypatch.setattr(main.ort, "set_default_logger_severity", lambda _level: None)


@pytest.mark.parametrize("page_count", [1, 2])
def test_cli_passes_file_pages_in_argument_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, page_count: int
) -> None:
    patch_main_setup(monkeypatch)
    pages = [tmp_path / f"page-{number}.png" for number in range(1, page_count + 1)]
    for page_path in pages:
        page_path.touch()
    process_images = Mock()
    monkeypatch.setattr(main, "process_images", process_images)
    monkeypatch.setattr(sys, "argv", ["homr", *[str(page_path) for page_path in pages]])

    main.main()

    assert process_images.call_args.args[0] == [str(page_path) for page_path in pages]


def test_cli_keeps_directory_input_as_separate_scores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    patch_main_setup(monkeypatch)
    folder = tmp_path / "pages"
    folder.mkdir()
    pages = [str(folder / "page-1.png"), str(folder / "page-2.png")]
    monkeypatch.setattr(main, "get_all_image_files_in_folder", lambda _folder: pages)
    process_image = Mock()
    monkeypatch.setattr(main, "process_image", process_image)
    monkeypatch.setattr(sys, "argv", ["homr", str(folder)])

    main.main()

    assert [call.args[0] for call in process_image.call_args_list] == pages


def test_process_images_writes_one_score_using_the_first_page_title(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = [tmp_path / "page-1.png", tmp_path / "page-2.png"]
    for page in pages:
        page.touch()

    parsed = {
        str(pages[0]): (
            [[EncodedSymbol("note_4", position="upper"), symbol("barline"), symbol("newline")]],
            "The title",
        ),
        str(pages[1]): (
            [[EncodedSymbol("note_8", position="upper"), symbol("barline")]],
            "Wrong page title",
        ),
    }
    monkeypatch.setattr(main, "parse_image", lambda path, _config: parsed[path])
    xml = Mock()
    generate_xml = Mock(return_value=xml)
    monkeypatch.setattr(main, "generate_xml", generate_xml)

    main.process_images([str(page) for page in pages], Mock(), Mock())

    generated_staffs = generate_xml.call_args.args[1]
    assert [item.rhythm for item in generated_staffs[0]] == [
        "note_4",
        "barline",
        "pagebreak",
        "note_8",
        "barline",
    ]
    assert generate_xml.call_args.args[2] == "The title"
    xml.write.assert_called_once_with(str(tmp_path / "page-1.musicxml"))


def test_process_images_writes_nothing_when_a_later_page_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = [tmp_path / "page-1.png", tmp_path / "page-2.png"]
    output = tmp_path / "page-1.musicxml"
    output.write_text("stale")

    def parse(path: str, _config: object) -> tuple[list[list[EncodedSymbol]], str]:
        if path == str(pages[1]):
            raise RuntimeError("page failed")
        return [[symbol("page1")]], "Title"

    monkeypatch.setattr(main, "parse_image", parse)
    generate_xml = Mock()
    monkeypatch.setattr(main, "generate_xml", generate_xml)

    with pytest.raises(RuntimeError, match="page failed"):
        main.process_images([str(page) for page in pages], Mock(), Mock())

    assert not output.exists()
    generate_xml.assert_not_called()
