"""The index page, where the committed fixtures sit apart from the songs.

In one worst-first table the five gate fixtures read as five more songs and the
songs read as gated. The split is presentation only: the totals above the
tables still count every judged note, whichever table its case is in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fixturecheck import report, series


def _entry(name: str, **over: object) -> dict:
    entry = {
        "name": name,
        "page": "",
        "score": 100.0,
        "agree": 10,
        "voice": 0,
        "pitch": 0,
        "size": 0,
        "timing": 0,
        "structure": 0,
        "staves_page": 1,
        "staves_homr": 1,
        "at_fault": "",
        "unison": 0,
        "meter": 0,
        "before": None,
    }
    entry.update(over)
    return entry


def _index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entries: list[dict]) -> str:
    monkeypatch.setattr(series, "SERIES", tmp_path / "series.jsonl")
    monkeypatch.setattr(report, "OUT", tmp_path / "report")
    return report.index_page(entries, "all").read_text()


def test_committed_fixtures_get_their_own_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = _index(
        tmp_path,
        monkeypatch,
        [_entry("hanget-soi", agree=7, pitch=3), _entry("some-song-s1", agree=5, voice=2)],
    )

    fixtures, songs = page.split("The committed fixtures")[1].split("Song systems")
    assert "hanget-soi" in fixtures and "hanget-soi" not in songs
    assert "some-song-s1" in songs and "some-song-s1" not in fixtures
    # Presentation only: the totals above still count both cases' notes.
    assert "<b>12</b>agree" in page
    assert "<b>3</b>wrong pitch" in page and "<b>2</b>wrong voice" in page


def test_no_fixture_section_when_none_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = _index(tmp_path, monkeypatch, [_entry("some-song-s1")])
    assert "The committed fixtures" not in page
    assert "some-song-s1" in page
