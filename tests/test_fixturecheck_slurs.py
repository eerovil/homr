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
    assert slurs.read_arcs(path) == [("slur", 1, 1, ("1", 1.0), ("1", 3.0), "")]


def test_a_tie_carries_its_pitch(tmp_path: Path) -> None:
    path = score(tmp_path, note(tie="start", step="G") + note(tie="stop", step="G") + "</measure>")
    assert slurs.read_arcs(path) == [("tie", 1, 1, ("1", 0.0), ("1", 1.0), "G")]


def test_loose_ends_are_arcs_over_the_system_edge(tmp_path: Path) -> None:
    path = score(
        tmp_path,
        note("stop") + note() + "</measure>"
        "<measure number='2'>" + note() + note(tie="start") + "</measure>",
    )
    assert sorted(slurs.read_arcs(path), key=str) == sorted(
        [("slur", 1, 1, None, ("1", 0.0), ""), ("tie", 1, 1, ("2", 1.0), None, "C")], key=str
    )


def test_slurs_and_ties_are_counted_apart() -> None:
    page: list[slurs.Arc] = [
        ("slur", 1, 1, ("1", 0.0), ("1", 2.0), ""),
        ("tie", 1, 1, ("1", 2.0), ("1", 3.0), "C"),
    ]
    homr: list[slurs.Arc] = [
        ("slur", 1, 1, ("1", 0.0), ("1", 2.0), ""),
        ("slur", 1, 1, ("1", 2.0), ("1", 3.0), ""),
    ]
    result = slurs.compare_arcs(page, homr)
    assert (result.slur.found, result.slur.missed, result.slur.invented) == (1, 0, 1)
    assert (result.tie.found, result.tie.missed, result.tie.invented) == (0, 1, 0)


def test_an_edge_end_is_found_by_either_kind() -> None:
    """Over a line break homr cannot tell a tie from a slur: it sees one end."""
    page: list[slurs.Arc] = [("tie", 1, 1, ("8", 1.0), None, "A")]
    homr: list[slurs.Arc] = [("slur", 1, 1, ("8", 1.0), None, "")]
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
    assert slurs.judge("absent", path, path, listed) is None
    found = slurs.judge("a", path, path, listed)
    assert found is not None and found.slur.found == 1
    empty = slurs.judge("empty", path, path, listed)
    assert empty is not None and empty.slur.invented == 1
    assert slurs.answer_key(tmp_path / "absent.json") == {}


def two_voices(tmp_path: Path, name: str, slur_voice: str, upper: str, lower: str) -> Path:
    """One bar, one staff: an upper line and a lower line under it, written
    with the given voice numbers, and a slur over notes 2-3 of ``slur_voice``."""

    def line(voice: str, step: str, octave: str) -> str:
        notes = ""
        for n in range(1, 5):
            kind = {2: "start", 3: "stop"}.get(n, "") if voice == slur_voice else ""
            mark = f"<notations><slur type='{kind}' number='1'/></notations>" if kind else ""
            notes += (
                f"<note><pitch><step>{step}</step><octave>{octave}</octave></pitch>"
                f"<duration>1</duration><voice>{voice}</voice>{mark}</note>"
            )
        return notes

    path = tmp_path / f"{name}.musicxml"
    path.write_text(
        "<score-partwise><part id='P1'><measure number='1'>"
        "<attributes><divisions>1</divisions></attributes>"
        + line(upper, "E", "5")
        + "<backup><duration>4</duration></backup>"
        + line(lower, "C", "4")
        + "</measure></part></score-partwise>"
    )
    return path


def test_a_slur_on_the_wrong_voice_is_not_found(tmp_path: Path) -> None:
    """illan-s05 prints voices 1/2 and 5/6 on its staves. The reference writes
    the slur on the upper line; a reading with the same beats on the lower line
    is a miss and an invention, not a find. The two files number their voices
    differently (1/2 against 5/6), which is why voices are compared by rank."""
    reference = two_voices(tmp_path, "ref", "1", "1", "2")
    key = {
        "case": [
            {
                "kind": "slur",
                "staff": 1,
                "voice": "1",
                "from": {"bar": "1", "onset": 1},
                "to": {"bar": "1", "onset": 2},
            }
        ]
    }
    right = two_voices(tmp_path, "right", "5", "5", "6")
    wrong = two_voices(tmp_path, "wrong", "6", "5", "6")
    hit = slurs.judge("case", right, reference, key)
    miss = slurs.judge("case", wrong, reference, key)
    assert hit is not None and (hit.slur.found, hit.slur.invented) == (1, 0)
    assert miss is not None and (miss.slur.found, miss.slur.missed, miss.slur.invented) == (0, 1, 1)
