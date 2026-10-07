"""A repeat sign homr skipped, which the note comparison could not see.

eerovil/musescore-choir-plugins#312: a start repeat opening a system was left out
of the parse, every note was right, the case read 100%, and the practice track
went back to the wrong bar. Judged per bar, across the whole system, because the
sign is drawn across the whole system.
"""

from __future__ import annotations

import json
from pathlib import Path

from fixturecheck import compare, references, series


def score(tmp_path: Path, name: str, signs: dict[tuple[int, int], list[str]]) -> Path:
    """Two parts of three bars; ``signs[(part, bar)]`` lists the repeats written there."""
    parts = []
    for part in (1, 2):
        bars = []
        for bar in (1, 2, 3):
            barlines = "".join(
                f'<barline location="{"left" if d == "forward" else "right"}">'
                f'<repeat direction="{d}"/></barline>'
                for d in signs.get((part, bar), [])
            )
            bars.append(
                f'<measure number="{bar}"><attributes><divisions>1</divisions></attributes>'
                f"<note><pitch><step>C</step><octave>4</octave></pitch>"
                f"<duration>4</duration><voice>1</voice></note>{barlines}</measure>"
            )
        parts.append(f'<part id="P{part}">{"".join(bars)}</part>')
    path = tmp_path / name
    path.write_text(
        '<?xml version="1.0"?><score-partwise version="3.1"><part-list>'
        '<score-part id="P1"><part-name>A</part-name></score-part>'
        '<score-part id="P2"><part-name>B</part-name></score-part></part-list>'
        f'{"".join(parts)}</score-partwise>'
    )
    return path


def test_a_start_repeat_homr_skipped_is_a_fault(tmp_path: Path) -> None:
    page = score(tmp_path, "page.musicxml", {(1, 1): ["forward"], (2, 1): ["forward"]})
    homr = score(tmp_path, "homr.musicxml", {})
    [row] = compare.compare_repeats(page, homr)
    assert (row.bar, row.page, row.homr, row.kind) == ("1", "start repeat", "none", "repeat")
    assert "did not write" in row.verdict
    result = compare.compare_output(page, homr)
    assert result.repeat == 1
    # Every note is right, which is exactly why it needs its own count.
    assert result.score == 100.0
    assert "1 repeat sign(s) missing or invented" in result.remaining


def test_one_staff_carrying_it_is_the_system_carrying_it(tmp_path: Path) -> None:
    page = score(tmp_path, "page.musicxml", {(1, 1): ["forward"], (2, 1): ["forward"]})
    homr = score(tmp_path, "homr.musicxml", {(2, 1): ["forward"]})
    assert compare.compare_repeats(page, homr) == []


def test_an_invented_end_repeat_is_a_fault_too(tmp_path: Path) -> None:
    page = score(tmp_path, "page.musicxml", {})
    homr = score(tmp_path, "homr.musicxml", {(1, 3): ["backward"]})
    [row] = compare.compare_repeats(page, homr)
    assert (row.bar, row.page, row.homr) == ("3", "none", "end repeat")
    assert "does not print" in row.verdict


def test_the_gate_remembers_repeats_and_reads_old_memories_as_none(tmp_path: Path) -> None:
    counts = dict.fromkeys(series.COUNTS, 0)
    counts.update(agree=4, staves_page=2, staves_homr=2)
    clean = references.marks(counts)
    counts["repeat"] = 1
    skipped = references.marks(counts)
    assert clean["repeat"] == 0 and skipped["repeat"] == 1
    assert references.worse(skipped, clean) == ["repeat 1, against 0 accepted"]
    # A memory written before repeats were counted holds none.
    path = tmp_path / "references.json"
    path.write_text(json.dumps({"cases": {"old": {"score": 100.0, "structure": 0, "meter": 0}}}))
    assert references.accepted(path) == {
        "old": {"score": 100.0, "structure": 0, "meter": 0, "repeat": 0}
    }
