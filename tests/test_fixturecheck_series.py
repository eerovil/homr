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
    series.record_run("fixturecheck", "all",
                      [case("a", agree=9, pitch=1)] + [_perfect(n, True) for n in "abcde"],
                      references="ref9", extra={"committed": list("abcde")},
                      path=path)
    written = quality.render(path)

    assert "ref9" in written
    assert "pass" in written
    # The two things it is not allowed to let a reader assume.
    assert "practice track" in written
    assert "Detection" in written


def test_the_summary_names_the_failing_fixtures(tmp_path):
    path = tmp_path / "series.jsonl"
    roster = ["hanget-soi", "kolme-kakea", "laulun-aika-s2", "sammon-ryosto", "system4"]
    series.record_run(
        "fixturecheck", "ten",
        [_perfect(n, n not in ("hanget-soi", "sammon-ryosto")) for n in roster],
        references="r", extra={"committed": roster}, path=path)
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


def test_a_run_that_judged_no_fixtures_cannot_turn_a_failing_gate_green(tmp_path):
    """The bug: a song-only run recorded `passed: true` over a standing FAIL.

    Most runs judge no committed fixture -- a song system, a retry of one case
    -- and reading "this run had nothing to say" as "the gate is fine" replaced
    a real `FAIL - 3/5` with `all 0 committed fixtures are perfect`.
    """
    path = tmp_path / "series.jsonl"
    roster = ["hanget-soi", "kolme-kakea", "laulun-aika-s2", "sammon-ryosto", "system4"]
    series.record_run(
        "fixturecheck", "all",
        [_perfect(n, n not in ("hanget-soi", "sammon-ryosto")) for n in roster],
        references="r", extra={"committed": roster}, path=path)
    # ...then an ordinary run over song systems only, which judges no fixture.
    series.record_run("fixturecheck", "one", [case("song-s4", agree=20)],
                      references="r", gate=None,
                      extra={"committed": roster}, path=path)

    written = quality.render(path)
    assert "FAIL" in written
    assert "hanget-soi" in written and "sammon-ryosto" in written
    assert "all 0 committed fixtures" not in written
    assert "3/5" in written                 # still speaking for all five


def test_a_run_with_no_fixtures_in_it_records_no_gate():
    """`gate_over`'s own rule: no fixtures judged is no opinion, not a pass."""
    from fixturecheck.__main__ import gate_over

    committed = {"system4", "hanget-soi"}
    assert gate_over([case("song-s1", agree=5)], committed) is None
    judged = gate_over([series.CaseRecord("system4", counts={"perfect": True})],
                       committed)
    assert judged == {"fixtures": 1, "perfect": 1, "failing": [], "passed": True}


def test_nothing_recorded_is_not_a_passing_gate(tmp_path):
    path = tmp_path / "series.jsonl"
    series.record_run("fixturecheck", "one", [case("song-s1", agree=5)],
                      references="r", gate=None, path=path)
    written = quality.render(path)
    assert "not evaluated" in written
    assert "pass" not in written.split("## What this does not measure")[0]


def _fixture_run(path, cases_, at_committed=("a", "b", "c", "d", "e"), **kw):
    return series.record_run("fixturecheck", kw.pop("tier", "one"), cases_,
                             references="r",
                             extra={"committed": sorted(at_committed)},
                             path=path, **kw)


def _perfect(name, ok):
    counts = {k: 0 for k in series.COUNTS}
    counts.update({"agree": 10, "perfect": ok})
    if not ok:
        counts["pitch"] = 2
    return series.CaseRecord(name, counts=counts)


def test_rerunning_one_passing_fixture_cannot_turn_a_failing_gate_green(tmp_path):
    """The hole the zero-fixture fix left open.

    `fixturecheck one system4` judges one fixture, passes it, and used to
    publish `1/1 perfect` — so a standing `FAIL - 3/5` went green because
    somebody re-ran a fixture that was never the problem, while both failing
    ones were still failing and nothing said so.
    """
    path = tmp_path / "series.jsonl"
    # A full run: three perfect, two not.
    _fixture_run(path, [_perfect("a", True), _perfect("b", True), _perfect("c", True),
                        _perfect("d", False), _perfect("e", False)], tier="all")
    assert "FAIL" in quality.render(path)

    # ...then one passing fixture on its own, which says nothing about d and e.
    _fixture_run(path, [_perfect("a", True)])

    written = quality.render(path)
    assert "FAIL" in written
    assert "3/5" in written
    assert "`d`" in written and "`e`" in written
    assert "all 1 committed fixtures" not in written


def test_the_published_gate_is_each_fixtures_own_latest_result(tmp_path):
    """A run moves the fixtures it ran, and only those."""
    path = tmp_path / "series.jsonl"
    _fixture_run(path, [_perfect(n, n not in ("d", "e")) for n in "abcde"], tier="all")
    gate = series.published_gate(series.runs(path))
    assert (gate["perfect"], gate["fixtures"]) == (3, 5)
    assert gate["failing"] == ["d", "e"] and not gate["passed"]

    # Fixing one of them, on its own, moves exactly one.
    _fixture_run(path, [_perfect("d", True)])
    gate = series.published_gate(series.runs(path))
    assert (gate["perfect"], gate["fixtures"]) == (4, 5)
    assert gate["failing"] == ["e"] and not gate["passed"]

    # And fixing the last one passes it, because now all five stand perfect.
    _fixture_run(path, [_perfect("e", True)])
    assert series.published_gate(series.runs(path))["passed"]


def test_a_one_case_run_still_reports_every_case(tmp_path, monkeypatch):
    """Re-reading one system used to throw away the view of everything else.

    The index listed exactly what the run judged, so a one-case run produced a
    one-case report while the other pages sat on disk with nothing linking to
    them. The series has held each case's own last measurement all along.
    """
    from fixturecheck import report

    path = tmp_path / "series.jsonl"
    monkeypatch.setattr(series, "SERIES", path)
    monkeypatch.setattr(report, "OUT", tmp_path / "report")

    _fixture_run(path, [_perfect(n, n != "d") for n in "abcde"], tier="all")

    live = [{"name": "a", "page": "a.html", "score": 100.0, "agree": 10,
             "voice": 0, "pitch": 0, "size": 0, "timing": 0, "structure": 0,
             "staves_page": 1, "staves_homr": 1, "at_fault": "", "unison": 0,
             "before": None}]
    combined = report._with_standing(live)

    assert sorted(e["name"] for e in combined) == list("abcde")
    # The one this run read carries no date; the rest carry when they were read.
    fresh = [e for e in combined if not e["measured"]]
    assert [e["name"] for e in fresh] == ["a"]
    assert all(e["measured"] for e in combined if e["name"] != "a")
    # And a carried-forward case keeps its own numbers, including a failing one.
    failing = next(e for e in combined if e["name"] == "d")
    assert failing["pitch"] == 2 and failing["score"] < 100.0


def test_a_fixture_nobody_has_judged_holds_the_gate_open(tmp_path):
    """"We have never looked" is not "we looked and it was fine"."""
    path = tmp_path / "series.jsonl"
    _fixture_run(path, [_perfect(n, True) for n in "abc"])
    gate = series.published_gate(series.runs(path))
    assert not gate["passed"]
    assert gate["unevaluated"] == ["d", "e"]
    assert "Never judged" in quality.render(path)


def test_the_benchmark_records_the_engine_it_measured_not_this_checkout(tmp_path):
    """`choir-bench` can be measuring an install, a worktree or a pod.

    Letting it fall through to the series' own default keyed its numbers to the
    checkout holding the harness -- which is this card's complaint, one level
    down.
    """
    import subprocess as sp

    # A worktree answers with its own git revision, uncommitted work marked.
    tree = tmp_path / "tree"
    tree.mkdir()
    sp.run(["git", "init", "-q"], cwd=tree, check=True)
    (tree / "homr").mkdir()
    (tree / "homr" / "main.py").write_text("x = 1\n")
    sp.run(["git", "add", "-A"], cwd=tree, check=True)
    sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
            "commit", "-qm", "one"], cwd=tree, check=True)
    head = sp.run(["git", "log", "-1", "--format=%h"], cwd=tree,
                  capture_output=True, text=True).stdout.strip()

    assert series.engine_revision(str(tree)) == head
    (tree / "homr" / "main.py").write_text("x = 2\n")
    assert series.engine_revision(str(tree)) == head + "+dirty"

    # An install answers with what pip wrote down when it installed it.
    venv = tmp_path / "venv"
    site = venv / "lib" / "python3.12" / "site-packages" / "homr-0.7.0.post103+ec41559.dist-info"
    site.mkdir(parents=True)
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "homr").write_text("#!/bin/sh\n")
    (site / "direct_url.json").write_text(json.dumps(
        {"url": "https://github.com/eerovil/homr.git",
         "vcs_info": {"commit_id": "ec4155991a5d7c17b8049dec2a690c86cba7db96"}}))

    assert series.engine_revision(None, str(venv / "bin" / "homr")) == "installed:ec41559"

    # Without a direct_url the version still carries the commit.
    (site / "direct_url.json").unlink()
    assert series.engine_revision(None, str(venv / "bin" / "homr")) == "installed:ec41559"

    # And an engine that cannot be identified says so rather than borrowing ours.
    assert series.engine_revision(None, str(tmp_path / "nope" / "bin" / "homr")) == "unknown"


def test_the_ordinary_run_finds_the_install_without_being_told(tmp_path, monkeypatch):
    """The documented invocation sets no `HOMR_BIN` and got `unknown`.

    `scripts/choir-bench.py --benchmark` with no `--tree` is *the* case this
    provenance exists for, and it was the one case that recorded nothing: the
    app resolves its own default venv, so nothing put `HOMR_BIN` in the
    environment for `engine_revision` to read. This exercises that path -- no
    `HOMR_BIN`, no `--tree`, nothing handed in.
    """
    venv = tmp_path / "homr-venv"
    site = venv / "lib" / "python3.12" / "site-packages" / "homr-0.7.0.post103+ec41559.dist-info"
    site.mkdir(parents=True)
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "homr").write_text("#!/bin/sh\n")
    (site / "direct_url.json").write_text(json.dumps(
        {"vcs_info": {"commit_id": "ec4155991a5d7c17b8049dec2a690c86cba7db96"}}))

    monkeypatch.delenv("HOMR_BIN", raising=False)
    monkeypatch.setattr(series, "DEFAULT_VENV", str(venv))

    assert series.installed_binary() == str(venv / "bin" / "homr")
    assert series.engine_revision() == "installed:ec41559"

    # And an explicit HOMR_BIN still wins, the way the app resolves it.
    monkeypatch.setenv("HOMR_BIN", str(venv / "bin" / "homr"))
    assert series.engine_revision() == "installed:ec41559"


def test_no_install_anywhere_is_unknown_rather_than_a_guess(tmp_path, monkeypatch):
    monkeypatch.delenv("HOMR_BIN", raising=False)
    monkeypatch.setattr(series, "DEFAULT_VENV", str(tmp_path / "nothing-here"))
    assert series.installed_binary() == "homr"
    assert series.engine_revision() == "unknown"


def test_a_worktree_is_keyed_by_its_homr_code_not_by_its_head(tmp_path):
    """A harness commit is not a new homr, and must not read as one.

    This pull request is the case: its head is several commits past the last
    change to `homr/`, and every one of them is harness and report. Keying a
    benchmark to the head would file its numbers under a homr that has never
    existed -- and would disagree with `homr_commit` and the parse cache, which
    both key on `homr/`.
    """
    import subprocess as sp

    tree = tmp_path / "tree"
    (tree / "homr").mkdir(parents=True)
    (tree / "fixturecheck").mkdir()
    sp.run(["git", "init", "-q"], cwd=tree, check=True)

    def commit(message):
        sp.run(["git", "add", "-A"], cwd=tree, check=True)
        sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                "commit", "-qm", message], cwd=tree, check=True)
        return sp.run(["git", "log", "-1", "--format=%h"], cwd=tree,
                      capture_output=True, text=True).stdout.strip()

    (tree / "homr" / "main.py").write_text("x = 1\n")
    engine = commit("a real homr change")

    (tree / "fixturecheck" / "report.py").write_text("# harness only\n")
    head = commit("harness only, no homr")
    assert head != engine

    assert series.engine_revision(str(tree)) == engine     # not `head`

    # Uncommitted work in homr/ is still marked, on the homr commit.
    (tree / "homr" / "main.py").write_text("x = 2\n")
    assert series.engine_revision(str(tree)) == engine + "+dirty"

    # ...and uncommitted work outside homr/ is not homr changing.
    sp.run(["git", "checkout", "--", "homr"], cwd=tree, check=True)
    (tree / "fixturecheck" / "report.py").write_text("# edited\n")
    assert series.engine_revision(str(tree)) == engine


def test_an_explicit_revision_beats_the_checkout(tmp_path):
    path = tmp_path / "series.jsonl"
    run = series.record_run("choir-bench", "benchmark", [case("a", agree=1)],
                            references="r", homr="installed:ec41559", path=path)
    assert run["homr"] == "installed:ec41559"


@pytest.mark.parametrize("outcome", [series.UNREADABLE, series.UNBUILDABLE])
def test_an_unread_case_carries_no_counts(outcome):
    """It scored nothing; writing zeros would read as a case that scored zero."""
    assert series.CaseRecord("x", outcome=outcome).to_json() == {"outcome": outcome}
