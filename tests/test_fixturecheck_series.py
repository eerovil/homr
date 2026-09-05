"""The record: that a run is kept, keyed by both commits, and never overwritten.

The defect these exist for is not subtle and is worth naming: the harness kept
one file holding the last run, so a three-case run destroyed a ninety-eight-case
sweep and the question "is this getting better over months" could not be asked
at all.
"""

from __future__ import annotations

import json

import pytest

from fixturecheck import quality, references, series
from fixturecheck.compare import Result, Row


def case(name, **counts):
    full = {k: 0 for k in series.COUNTS}
    full.update(counts)
    return series.CaseRecord(name, counts=full)


def test_a_small_run_does_not_destroy_a_large_one(tmp_path):
    """The whole reason this file exists."""
    big = tmp_path / "series.jsonl"
    series.record_run("fixturecheck", "all",
                      [case(f"song-s{i}", agree=10) for i in range(98)],
                      references="ref1", path=big)
    series.record_run("fixturecheck", "one", [case("song-s0", agree=9, pitch=1)],
                      references="ref1", path=big)

    assert len(series.runs(big)) == 2
    # The sweep is still there, in full, and still says what it said.
    sweep = series.runs(big)[0]
    assert len(sweep["cases"]) == 98
    assert sweep["headline"]["judged"] == 980
    # And each case's own last measurement survives the small run.
    standing = series.previous_cases("fixturecheck", big)
    assert len(standing) == 98
    assert standing["song-s0"]["pitch"] == 1     # the one the small run re-read
    assert standing["song-s1"]["agree"] == 10    # untouched by it


def test_a_run_names_both_the_homr_and_the_references(tmp_path):
    """A score that improved because a reference was fixed is not homr improving."""
    path = tmp_path / "series.jsonl"
    run = series.record_run("fixturecheck", "ten", [case("a", agree=1)],
                            references="deadbeef", path=path)
    assert run["references"] == "deadbeef"
    assert run["homr"]          # whatever this checkout is, it is recorded


def test_losing_notes_and_shifting_beats_count_against_the_score():
    """The old denominator let a system missing half its notes read 100%."""
    half_missing = case("x", agree=10, size=10)
    every_beat_moved = case("y", agree=10, timing=10)

    assert series.headline([half_missing])["percent"] == 50.0
    assert series.headline([every_beat_moved])["percent"] == 50.0
    # And the same on the Result the harness actually computes.
    result = Result(agree=10, size=10)
    assert result.judged == 10          # notes matched one against one
    assert result.scored == 20          # ...plus what was lost
    assert result.score == 50.0
    assert not result.perfect


def test_a_case_with_nothing_in_it_is_not_perfect():
    """Or an empty parse would sail through the gate."""
    assert not Result().perfect
    assert Result(agree=5).perfect
    assert not Result(agree=5, staves_page=2, staves_homr=3).perfect


def test_a_case_that_could_not_be_read_is_recorded_not_skipped(tmp_path):
    """Skipping made a case that stopped parsing look like one nobody ran."""
    path = tmp_path / "series.jsonl"
    run = series.record_run(
        "fixturecheck", "ten",
        [case("fine", agree=10),
         series.CaseRecord("crashed", outcome=series.UNREADABLE),
         series.CaseRecord("absent", outcome=series.UNBUILDABLE)],
        references="r", path=path)

    assert run["outcomes"] == {series.READ: 1, series.UNREADABLE: 1,
                               series.UNBUILDABLE: 1}
    assert run["cases"]["crashed"] == {"outcome": series.UNREADABLE}
    # A crash is not folded into the accuracy as a zero: it has no notes to be
    # right or wrong about, and averaging it in would hide it.
    assert run["headline"]["percent"] == 100.0


def test_only_the_disagreements_are_kept(tmp_path):
    """The first three rows of a system are usually three notes that agree."""
    result = Result(rows=[
        Row("m1", "C", "C", "", "agree"),
        Row("m1", "D", "D", "", "agree"),
        Row("m2", "E", "F", "a step out", "pitch"),
        Row("m3", "G", "-", "lost", "size"),
        Row("m4", "A", "B", "a step out", "pitch"),
        Row("m5", "B", "C", "a step out", "pitch"),
    ])
    kept = series.first_faults(result)
    assert [row["where"] for row in kept] == ["m2", "m3", "m4"]
    assert all(row["kind"] != "agree" for row in kept)


def test_an_unchanged_fixture_table_is_recorded_by_reference(tmp_path):
    """The five are gated at 100%, so most runs would repeat the same table."""
    path = tmp_path / "series.jsonl"
    table = [{"where": "m1", "page": "C", "homr": "C", "kind": "agree"}]

    first = series.record_run(
        "fixturecheck", "one",
        [series.CaseRecord("system4", counts={k: 0 for k in series.COUNTS},
                           rows=table)],
        references="r", path=path)

    was, from_run = series.last_rows("system4", path=path)
    assert was == table and from_run == first["at"]

    series.record_run(
        "fixturecheck", "one",
        [series.CaseRecord("system4", counts={k: 0 for k in series.COUNTS},
                           rows_same_as=from_run)],
        references="r", path=path)

    # Followed back to the run that actually holds it, so a long unchanged
    # stretch costs one hop rather than a walk.
    again, held_by = series.last_rows("system4", path=path)
    assert again == table and held_by == first["at"]


def test_a_damaged_line_does_not_take_the_series_with_it(tmp_path):
    path = tmp_path / "series.jsonl"
    series.record_run("fixturecheck", "one", [case("a", agree=1)],
                      references="r", path=path)
    with path.open("a") as writing:
        writing.write("{not json\n")
    series.record_run("fixturecheck", "one", [case("b", agree=1)],
                      references="r", path=path)

    assert len(series.runs(path)) == 2


class Fake:
    def __init__(self, name, image, reference):
        self.name, self.image, self.reference = name, image, reference


def _fixture_case(tmp_path, name, image=b"pixels", reference=b"<score/>"):
    picture, score = tmp_path / f"{name}.png", tmp_path / f"{name}.musicxml"
    picture.write_bytes(image)
    score.write_bytes(reference)
    return Fake(name, picture, score)


def test_a_reference_that_moves_is_visible_in_the_run_key(tmp_path):
    """A number that improved because a cleaned score was fixed must show it."""
    manifest = tmp_path / "references.json"
    one = _fixture_case(tmp_path, "a")
    references.write([one], manifest)
    frozen = references.stamp([one], manifest)
    assert frozen != "unfrozen" and "+" not in frozen

    one.reference.write_bytes(b"<score>corrected</score>")
    moved = references.stamp([one], manifest)
    assert moved.startswith(frozen) and "drift1" in moved
    assert references.drift([one], manifest)["changed"] == ["a"]


def test_a_reference_nobody_froze_is_not_silently_the_manifest(tmp_path):
    manifest = tmp_path / "references.json"
    known = _fixture_case(tmp_path, "known")
    references.write([known], manifest)
    fresh = _fixture_case(tmp_path, "fresh")

    assert "new1" in references.stamp([known, fresh], manifest)
    assert references.drift([known, fresh], manifest)["unfrozen"] == ["fresh"]


def test_freezing_a_few_does_not_drop_the_rest(tmp_path):
    """A ten-case run must not read in the diff as eighty-three deletions."""
    manifest = tmp_path / "references.json"
    references.write([_fixture_case(tmp_path, f"c{i}") for i in range(5)], manifest)
    references.write([_fixture_case(tmp_path, "c0", image=b"redrawn")], manifest)

    held = json.loads(manifest.read_text())["cases"]
    assert len(held) == 5


def test_the_summary_names_which_homr_and_says_what_it_cannot_measure(tmp_path):
    path = tmp_path / "series.jsonl"
    series.record_run("fixturecheck", "all", [case("a", agree=9, pitch=1)],
                      references="ref9", gate={"fixtures": 5, "perfect": 5,
                                               "failing": [], "passed": True},
                      path=path)
    written = quality.render(path)

    assert "90.0%" in written
    assert "ref9" in written
    assert "pass" in written
    # The two things it is not allowed to let a reader assume.
    assert "practice track" in written
    assert "Detection" in written


def test_the_summary_names_the_failing_fixtures(tmp_path):
    path = tmp_path / "series.jsonl"
    series.record_run("fixturecheck", "ten", [case("a", agree=1)],
                      references="r",
                      gate={"fixtures": 5, "perfect": 3,
                            "failing": ["hanget-soi", "sammon-ryosto"],
                            "passed": False},
                      path=path)
    written = quality.render(path)

    assert "FAIL" in written
    assert "hanget-soi" in written and "sammon-ryosto" in written


def test_the_two_harnesses_are_reported_apart(tmp_path):
    """They score different things; one figure over both would mean nothing."""
    path = tmp_path / "series.jsonl"
    series.record_run("fixturecheck", "all", [case("a", agree=8, pitch=2)],
                      references="r", path=path)
    series.record_run("choir-bench", "benchmark", [case("b", agree=5, pitch=5)],
                      references="r", path=path)

    assert series.latest("fixturecheck", path)["headline"]["percent"] == 80.0
    assert series.latest("choir-bench", path)["headline"]["percent"] == 50.0
    written = quality.render(path)
    assert "80.0%" in written and "50.0%" in written
    assert "not** averaged" in written or "not averaged" in written


@pytest.mark.parametrize("outcome", [series.UNREADABLE, series.UNBUILDABLE])
def test_an_unread_case_carries_no_counts(outcome):
    """It scored nothing; writing zeros would read as a case that scored zero."""
    assert series.CaseRecord("x", outcome=outcome).to_json() == {"outcome": outcome}
