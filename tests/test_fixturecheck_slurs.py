"""The slur score: what counts as found, missed and invented (fixturecheck/slurs.py)."""

from __future__ import annotations

import json
from pathlib import Path

from fixturecheck import slurs


def score(tmp_path: Path, notes: str) -> Path:
    """One part, one staff, 4/4 with divisions 1; ``notes`` is the measures' body."""
    path = tmp_path / "s.musicxml"
    path.write_text(
        "<score-partwise><part id='P1'>"
        "<measure number='1'><attributes><divisions>1</divisions></attributes>"
        + notes + "</part></score-partwise>")
    return path


def note(kind: str = "") -> str:
    slur = f"<notations><slur type='{kind}' number='1'/></notations>" if kind else ""
    return f"<note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration>{slur}</note>"


def test_a_slur_is_its_staff_and_both_ends(tmp_path: Path) -> None:
    path = score(tmp_path, note() + note("start") + note() + note("stop") + "</measure>")
    assert slurs.read_slurs(path) == [(1, ("1", 1.0), ("1", 3.0))]


def test_loose_ends_are_slurs_over_the_system_edge(tmp_path: Path) -> None:
    path = score(tmp_path, note("stop") + note() + note() + note() + "</measure>"
                 "<measure number='2'>" + note() + note() + note() + note("start") + "</measure>")
    assert sorted(slurs.read_slurs(path), key=str) == sorted(
        [(1, None, ("1", 0.0)), (1, ("2", 3.0), None)], key=str)


def test_found_missed_and_invented() -> None:
    page = [(1, ("1", 0.0), ("1", 2.0)), (1, ("2", 3.0), None)]
    homr = [(1, ("1", 0.0), ("1", 2.0)), (2, ("1", 0.0), ("1", 1.0))]
    result = slurs.compare_slurs(page, homr)
    assert (result.found, result.missed, result.invented) == (1, 1, 1)
    assert (result.edge, result.edge_found) == (1, 0)


def test_a_case_the_key_does_not_list_is_not_judged(tmp_path: Path) -> None:
    key = tmp_path / "slurs.json"
    key.write_text(json.dumps({"cases": {"a": [
        {"staff": 1, "from": {"bar": "1", "onset": 1}, "to": {"bar": "1", "onset": 3}}]}}))
    path = score(tmp_path, note() + note("start") + note() + note("stop") + "</measure>")
    listed = slurs.answer_key(key)
    assert slurs.judge("b", path, listed) is None
    assert slurs.judge("a", path, listed).found == 1
    assert slurs.answer_key(tmp_path / "absent.json") == {}
