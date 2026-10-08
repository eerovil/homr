"""The arc score: slurs and ties found, missed and invented (fixturecheck/slurs.py)."""

from __future__ import annotations

import json
from pathlib import Path

from fixturecheck import slurs


def score(tmp_path: Path, notes: str) -> Path:
    """One part, one staff, divisions 1; ``notes`` is the measures' body."""
    path = tmp_path / "s.musicxml"
    path.write_text(
        "<score-partwise><part id='P1'>"
        "<measure number='1'><attributes><divisions>1</divisions></attributes>"
        + notes
        + "</part></score-partwise>"
    )
    return path


def note(slur: str = "", tie: str = "", step: str = "C") -> str:
    marks = ""
    if slur:
        marks += f"<slur type='{slur}' number='1'/>"
    if tie:
        marks += f"<tied type='{tie}'/>"
    marks = f"<notations>{marks}</notations>" if marks else ""
    return (
        f"<note><pitch><step>{step}</step><octave>4</octave></pitch>"
        f"<duration>1</duration>{marks}</note>"
    )


def test_a_slur_is_its_staff_and_both_ends(tmp_path: Path) -> None:
    path = score(tmp_path, note() + note("start") + note() + note("stop") + "</measure>")
    assert slurs.read_arcs(path) == [("slur", 1, ("1", 1.0), ("1", 3.0), "")]


def test_a_tie_carries_its_pitch(tmp_path: Path) -> None:
    path = score(tmp_path, note(tie="start", step="G") + note(tie="stop", step="G") + "</measure>")
    assert slurs.read_arcs(path) == [("tie", 1, ("1", 0.0), ("1", 1.0), "G")]


def test_loose_ends_are_arcs_over_the_system_edge(tmp_path: Path) -> None:
    path = score(
        tmp_path,
        note("stop") + note() + "</measure>"
        "<measure number='2'>" + note() + note(tie="start") + "</measure>",
    )
    assert sorted(slurs.read_arcs(path), key=str) == sorted(
        [("slur", 1, None, ("1", 0.0), ""), ("tie", 1, ("2", 1.0), None, "C")], key=str
    )


def test_slurs_and_ties_are_counted_apart() -> None:
    page: list[slurs.Arc] = [
        ("slur", 1, ("1", 0.0), ("1", 2.0), ""),
        ("tie", 1, ("1", 2.0), ("1", 3.0), "C"),
    ]
    homr: list[slurs.Arc] = [
        ("slur", 1, ("1", 0.0), ("1", 2.0), ""),
        ("slur", 1, ("1", 2.0), ("1", 3.0), ""),
    ]
    result = slurs.compare_arcs(page, homr)
    assert (result.slur.found, result.slur.missed, result.slur.invented) == (1, 0, 1)
    assert (result.tie.found, result.tie.missed, result.tie.invented) == (0, 1, 0)


def test_an_edge_end_is_found_by_either_kind() -> None:
    """Over a line break homr cannot tell a tie from a slur: it sees one end."""
    page: list[slurs.Arc] = [("tie", 1, ("8", 1.0), None, "A")]
    homr: list[slurs.Arc] = [("slur", 1, ("8", 1.0), None, "")]
    result = slurs.compare_arcs(page, homr)
    assert (result.tie.found, result.slur.invented) == (1, 0)
    assert (result.edge, result.edge_found) == (1, 1)


def test_a_case_listed_with_no_arcs_catches_an_invented_one(tmp_path: Path) -> None:
    key = tmp_path / "slurs.json"
    key.write_text(
        json.dumps(
            {
                "cases": {
                    "empty": [],
                    "a": [
                        {
                            "kind": "slur",
                            "staff": 1,
                            "from": {"bar": "1", "onset": 1},
                            "to": {"bar": "1", "onset": 3},
                        }
                    ],
                }
            }
        )
    )
    path = score(tmp_path, note() + note("start") + note() + note("stop") + "</measure>")
    listed = slurs.answer_key(key)
    assert slurs.judge("absent", path, listed) is None
    found = slurs.judge("a", path, listed)
    assert found is not None and found.slur.found == 1
    empty = slurs.judge("empty", path, listed)
    assert empty is not None and empty.slur.invented == 1
    assert slurs.answer_key(tmp_path / "absent.json") == {}
