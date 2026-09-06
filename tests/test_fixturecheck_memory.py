"""The gate: nothing gets worse, per case.

What is under test is the rule the operator settled on — each case remembers the
notes-right score it was last accepted at, a run fails if any case reads below
its own memory, and an improvement takes the memory up with it. Three tiers and
one rule: nothing here knows whether a case is a pin, one of the five, or one of
the ninety-odd song systems, which is the property that stops a win on the
fixtures paying for a loss on real music.

The old gate is pinned too, by its absence: no `perfect` flag, and a case that
reads worse says by how much rather than flipping a light.
"""

from __future__ import annotations

import json

import pytest

from fixturecheck import cases, references, series
from fixturecheck.__main__ import gate_over, ratchet
from fixturecheck.compare import Result


def _counts(**over):
    full = {k: 0 for k in series.COUNTS}
    full.update(over)
    return full


def record(name, **over):
    return series.CaseRecord(name, counts=_counts(**over))


def manifest(tmp_path, **memories):
    """A manifest holding a fingerprint and a memory for each named case."""
    path = tmp_path / "references.json"
    held = {}
    for name, reading in memories.items():
        held[name] = {"image": f"i-{name}", "reference": f"r-{name}", **reading}
    path.write_text(json.dumps({"digest": "d", "cases": held}))
    return path


def reading(score, structure=0, meter=0):
    return {"score": score, "structure": structure, "meter": meter}


# --- what a case is remembered by ----------------------------------------


def test_the_remembered_score_is_the_score_the_report_shows():
    """Two spellings of one number is how they come to disagree.

    `marks` reads the counts as the series records them, because the gate has to
    judge a case out of the record as well as out of a live run. It has to give
    the same answer as the `Result` the report page prints.
    """
    result = Result(agree=90, pitch=5, size=3, timing=2)
    counts = {k: getattr(result, k) for k in series.COUNTS}

    assert references.marks(counts)["score"] == round(result.score, 2)
    assert references.marks(counts)["score"] == 90.0


def test_a_case_is_remembered_by_three_things_not_one():
    """Structure and meter are counted as cases, not as notes.

    Neither is in the notes-right percentage — that is deliberate and predates
    this card — so a memory made only of that percentage would have dropped the
    meter gate on the floor. "Nothing gets worse" has to cover all three.
    """
    was = reading(98.0)
    assert references.worse(reading(97.9), was)
    assert references.worse(reading(98.0, structure=1), was)
    assert references.worse(reading(98.0, meter=3), was)
    assert not references.worse(reading(98.0), was)
    assert not references.worse(reading(99.5), was)


def test_what_fell_is_said_with_its_numbers_in_it():
    """An alarm reading "below its memory" is one people learn to click through."""
    said = references.worse(reading(91.25), reading(97.5))
    assert said == ["91.25% of notes right, against 97.50% accepted"]

    both = references.worse(reading(90.0, meter=2), reading(95.0, meter=0))
    assert len(both) == 2 and "meter 2, against 0 accepted" in both


# --- the alarm -----------------------------------------------------------


def test_a_case_that_falls_fails_and_one_that_holds_does_not(tmp_path):
    path = manifest(tmp_path, a=reading(90.0), b=reading(90.0))

    held = references.judge({"a": reading(90.0), "b": reading(95.0)}, [], path=path)
    assert held["passed"] and held["judged"] == ["a", "b"] and not held["below"]

    fell = references.judge({"a": reading(89.99), "b": reading(95.0)}, [], path=path)
    assert not fell["passed"]
    assert list(fell["below"]) == ["a"]


def test_a_win_on_one_case_does_not_pay_for_a_loss_on_another(tmp_path):
    """Why the memory is per case and not one figure for the run.

    Across the run these two average better than they were accepted at. That is
    exactly the reading the operator ruled out: a change that improves the
    fixtures while ruining a song is a change that must trip the alarm.
    """
    path = manifest(tmp_path, a=reading(90.0), b=reading(90.0))

    gate = references.judge({"a": reading(99.0), "b": reading(85.0)}, [], path=path)

    assert not gate["passed"]
    assert list(gate["below"]) == ["b"]


def test_a_case_homr_can_no_longer_read_at_all_fails(tmp_path):
    """The loudest regression there is, and a score cannot express it."""
    path = manifest(tmp_path, a=reading(90.0))

    gate = references.judge({}, ["a"], path=path)

    assert not gate["passed"] and gate["unreadable"] == ["a"]


def test_a_case_nobody_has_accepted_is_neither_pass_nor_fail(tmp_path):
    """A first sighting has nothing to be compared against."""
    path = manifest(tmp_path, a=reading(90.0))

    gate = references.judge({"a": reading(90.0), "new": reading(10.0)}, [], path=path)

    assert gate["passed"]
    assert gate["judged"] == ["a"] and gate["unremembered"] == ["new"]


def test_a_case_whose_reference_moved_is_held_out_of_the_gate(tmp_path):
    """Its memory is a number about music that has since been edited.

    Failing a run for that would blame homr for somebody correcting a score,
    which is the confusion the reference fingerprint exists to prevent one
    level up.
    """
    path = manifest(tmp_path, a=reading(90.0), b=reading(90.0))

    gate = references.judge({"a": reading(10.0), "b": reading(90.0)},
                            [], adrift=["a"], path=path)

    assert gate["passed"]
    assert gate["adrift"] == ["a"] and gate["judged"] == ["b"]


def test_every_tier_is_judged_by_the_same_rule(monkeypatch, tmp_path):
    """A pin, a fixture and a song system are one object with one rule.

    The old gate looked only at the five committed fixtures, so a change that
    lifted them while costing the corpus passed. Nothing in `gate_over` knows
    which tier a case belongs to.
    """
    path = manifest(tmp_path, pinned=reading(50.0), system4=reading(100.0),
                    **{"song-s3": reading(95.0)})
    monkeypatch.setattr(references, "accepted", lambda *a, **k: json_memory(path))

    gate = gate_over([record("pinned", agree=1),          # 100%, up from 50
                      record("system4", agree=10),        # 100%, holding
                      record("song-s3", agree=9, pitch=1)],   # 90%, down from 95
                     [])

    assert not gate["passed"]
    assert list(gate["below"]) == ["song-s3"]


def json_memory(path):
    held = json.loads(path.read_text())["cases"]
    return {name: {f: entry[f] for f in references.MEMORY}
            for name, entry in held.items()
            if all(f in entry for f in references.MEMORY)}


# --- the ratchet ---------------------------------------------------------


def test_an_improvement_moves_the_memory_up_and_a_fall_never_does(tmp_path):
    """The ratchet turns one way on its own.

    An improvement is written into a committed file, so the next run has to hold
    on to it. Accepting a *fall* is a judgement somebody makes with the report
    open, and `freeze` is where that is expressed — a run must never do it, or
    the alarm would silently re-arm itself one notch lower every time.
    """
    path = manifest(tmp_path, a=reading(90.0), b=reading(90.0))
    memory = json_memory(path)

    raised = references.remember(
        {"a": reading(96.0)}, path=path)          # only what the caller offers
    assert raised == ["a"]
    assert json_memory(path)["a"]["score"] == 96.0
    assert json_memory(path)["b"]["score"] == 90.0

    # And the ratchet itself never offers a fall.
    offered = ratchet([record("a", agree=8, pitch=2)], memory, [])
    assert offered == []
    assert json_memory(path)["a"]["score"] == 96.0


def test_a_first_sighting_is_recorded_so_adding_a_case_costs_one_run(tmp_path):
    path = manifest(tmp_path)
    assert references.remember({"fresh": reading(73.5)}, path=path) == ["fresh"]
    assert json_memory(path)["fresh"]["score"] == 73.5


def test_a_case_whose_reference_moved_is_not_ratcheted(tmp_path):
    """Its new reading has not been looked at by the person who moved it."""
    path = manifest(tmp_path, a=reading(90.0))
    memory = json_memory(path)

    assert ratchet([record("a", agree=10)], memory, adrift=["a"]) == []


def test_the_gate_is_judged_before_the_ratchet_writes(tmp_path, monkeypatch):
    """A ratchet that ran first would find every case standing at its memory.

    This is the ordering bug the run loop is written around, pinned here because
    it would make the gate pass everything, every time, and look entirely fine.
    """
    path = manifest(tmp_path, a=reading(90.0))
    monkeypatch.setattr(references, "MANIFEST", path)
    monkeypatch.setattr(references, "accepted", lambda *a, **k: json_memory(path))

    fallen = [record("a", agree=8, pitch=2)]
    gate = gate_over(fallen, [])
    assert not gate["passed"]

    # ...and the fall is still not written, so the next run fails the same way.
    assert ratchet(fallen, json_memory(path), []) == []
    assert json_memory(path)["a"]["score"] == 90.0


# --- freezing ------------------------------------------------------------


class Fake:
    def __init__(self, name, image, reference):
        self.name, self.image, self.reference = name, image, reference


def _case(tmp_path, name, image=b"pixels", reference=b"<score/>"):
    picture, score = tmp_path / f"{name}.png", tmp_path / f"{name}.musicxml"
    picture.write_bytes(image)
    score.write_bytes(reference)
    return Fake(name, picture, score)


def test_freezing_a_case_whose_files_moved_forgets_its_score(tmp_path):
    """The score was about the picture and reference that have just been replaced.

    Carrying it over would gate a new case against an old case's number, and do
    it at exactly the moment nobody is looking at scores.
    """
    path = tmp_path / "references.json"
    case = _case(tmp_path, "a")
    references.write([case], path)
    references.remember({"a": reading(90.0)}, path=path)
    assert json_memory(path)["a"]["score"] == 90.0

    case.image.write_bytes(b"rescanned")
    _, forgotten = references.write([case], path)

    assert forgotten == ["a"]
    assert "a" not in json_memory(path)


def test_freezing_an_unchanged_case_keeps_its_score(tmp_path):
    path = tmp_path / "references.json"
    case = _case(tmp_path, "a")
    references.write([case], path)
    references.remember({"a": reading(90.0)}, path=path)

    _, forgotten = references.write([case], path)

    assert forgotten == []
    assert json_memory(path)["a"]["score"] == 90.0


def test_a_memory_does_not_change_the_digest_a_run_is_keyed_by(tmp_path):
    """Or every improvement would start a fresh identity.

    Under a new identity every case is `unevaluated`, so the ratchet would erase
    the standing record it exists to keep — the improvement would read as
    "nothing has been judged under this homr".
    """
    path = tmp_path / "references.json"
    case = _case(tmp_path, "a")
    was = references.write([case], path)[0]["digest"]

    references.remember({"a": reading(90.0)}, path=path)

    assert json.loads(path.read_text())["digest"] == was
    # ...and the case is not read as having drifted, either.
    assert references.drift([case], path) == {"changed": [], "unfrozen": []}


def test_a_memory_for_a_case_nobody_froze_is_not_a_drifted_reference(tmp_path):
    """An entry can hold a score and no fingerprint, and that is not a change.

    Read as a fingerprint of nothing it looks like a reference that has been
    emptied, so the case would report as drifted on every later run and be held
    out of the gate for good — the ratchet quietly disarming itself for exactly
    the cases it had just started watching. Its name must stay out of the digest
    for the same reason: a first sighting would otherwise move the identity and
    send every case back to `unevaluated`.
    """
    path = tmp_path / "references.json"
    frozen_case, fresh = _case(tmp_path, "old"), _case(tmp_path, "new")
    was = references.write([frozen_case], path)[0]["digest"]

    references.remember({"new": reading(80.0)}, path=path)

    assert references.drift([fresh], path)["changed"] == []
    assert references.drift([fresh], path)["unfrozen"] == ["new"]
    assert json.loads(path.read_text())["digest"] == was
    # The memory is kept, though — it is the fingerprint that is missing.
    assert json_memory(path)["new"]["score"] == 80.0


# --- a pin ---------------------------------------------------------------


PAGE = """<?xml version="1.0"?>
<score-partwise version="4.0"><part-list><score-part id="P1">
<part-name>V</part-name></score-part></part-list>
<part id="P1">
 <measure number="1"><attributes><divisions>1</divisions></attributes>
  <note><pitch><step>C</step><octave>4</octave></pitch>
  <duration>1</duration><type>quarter</type></note></measure>
 <measure number="2"><note><pitch><step>D</step><octave>4</octave></pitch>
  <duration>1</duration><type>quarter</type></note></measure>
 <measure number="3"><note><pitch><step>E</step><octave>4</octave></pitch>
  <duration>1</duration><type>quarter</type></note></measure>
</part></score-partwise>
"""


def _source(tmp_path):
    from PIL import Image

    picture = tmp_path / "src.png"
    Image.new("RGB", (300, 100), "white").save(picture)
    score = tmp_path / "src.musicxml"
    score.write_text(PAGE)
    return cases.Case("src", picture, score, "fixture")


def test_a_pin_cuts_both_sides_the_same_way(tmp_path, monkeypatch):
    """Which is what makes it a case rather than two pictures.

    A pin is tier 1: the exact bars an error was found on, committed so the fix
    is proven against the thing that was wrong. Making it cheap is the point —
    a pin that costs an afternoon is a pin nobody makes.
    """
    pytest.importorskip("PIL")
    from fixturecheck import bars

    source = _source(tmp_path)
    monkeypatch.setattr(bars, "geometry", lambda *a, **k: {
        "staves": [{"top": 0.2, "bottom": 0.4}],
        "bar_lines": [0.0, 0.33, 0.66, 1.0]})

    pinned = cases.pin("m2-beats", source, 2, 2, "the beats walked",
                       register=tmp_path / "pins.json", into=tmp_path / "pins")

    assert pinned.image.exists() and pinned.reference.exists()
    assert pinned.committed and pinned.why == "the beats walked"
    # The reference holds that bar and only that bar...
    assert bars.bars_in(pinned.reference) == ["2"]
    # ...and it carries the opening attributes, or it would be engraved in
    # whatever MuseScore assumes rather than in the music's own spelling.
    assert "<divisions>1</divisions>" in pinned.reference.read_text()
    # ...and the crop is narrower than the page it came from.
    from PIL import Image
    with Image.open(pinned.image) as cut, Image.open(source.image) as whole:
        assert cut.width < whole.width


def test_a_pin_is_registered_so_it_runs_with_every_tier(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    from fixturecheck import bars

    monkeypatch.setattr(bars, "geometry", lambda *a, **k: {
        "staves": [{"top": 0.2, "bottom": 0.4}],
        "bar_lines": [0.0, 0.33, 0.66, 1.0]})
    register, into = tmp_path / "pins.json", tmp_path / "pins"
    cases.pin("m2-beats", _source(tmp_path), 2, 2, "why", register, into)

    monkeypatch.setattr(cases, "PINS", register)
    monkeypatch.setattr(cases, "PIN_FILES", into)

    named = [case.name for case in cases.pinned_cases()]
    assert named == ["m2-beats"]
    # ...and a pin is a committed case, so it travels with a clone.
    assert cases.pinned_cases()[0].committed
    assert json.loads(register.read_text())["pins"]["m2-beats"]["from"] == "src"


def test_a_pin_refuses_rather_than_cutting_the_wrong_bars(tmp_path, monkeypatch):
    """A crop of the wrong bars is worse than no crop.

    The detected barlines have to imply the bars the reference says the system
    holds. Where they do not, which bars a picture held would be a guess — and a
    confident picture of the wrong music is the mistake this harness exists to
    catch one level up.
    """
    pytest.importorskip("PIL")
    from fixturecheck import bars

    source = _source(tmp_path)
    monkeypatch.setattr(bars, "geometry", lambda *a, **k: None)

    with pytest.raises(cases.CannotPin):
        cases.pin("nope", source, 2, 2, "why",
                  tmp_path / "pins.json", tmp_path / "pins")

    # And nothing is left behind on disk for a later run to pick up.
    assert not (tmp_path / "pins" / "nope.musicxml").exists()
    assert not (tmp_path / "pins.json").exists()


def test_a_pin_outside_the_case_is_refused(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    source = _source(tmp_path)

    with pytest.raises(cases.CannotPin) as refused:
        cases.pin("nope", source, 2, 9, "why",
                  tmp_path / "pins.json", tmp_path / "pins")

    assert "3 bar(s)" in str(refused.value)


# --- the run, end to end -------------------------------------------------


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """The whole run with homr and the comparison stubbed out.

    What is under test is the loop's own arithmetic — the order the gate and the
    ratchet happen in, what the exit status is, and what ends up in the manifest
    — so the two expensive parts are replaced and everything else is real.
    """
    pytest.importorskip("PIL")
    from PIL import Image

    from fixturecheck import __main__ as main
    from fixturecheck import quality, report

    manifest_path = tmp_path / "references.json"
    manifest_path.write_text(json.dumps({"digest": "d", "cases": {}}))
    monkeypatch.setattr(references, "MANIFEST", manifest_path)
    monkeypatch.setattr(references, "accepted",
                        lambda *a, **k: json_memory(manifest_path))
    monkeypatch.setattr(references, "remember",
                        lambda readings, path=None: _remember(manifest_path, readings))
    monkeypatch.setattr(references, "drift", lambda *a, **k: {"changed": [],
                                                             "unfrozen": []})
    monkeypatch.setattr(references, "stamp", lambda *a, **k: "refX")
    monkeypatch.setattr(series, "SERIES", tmp_path / "series.jsonl")
    monkeypatch.setattr(quality, "QUALITY", tmp_path / "QUALITY.md")
    monkeypatch.setattr(report, "OUT", tmp_path / "report")
    monkeypatch.setattr(main, "code_fingerprint", lambda: "abc123")
    # The engraving needs MuseScore and is pinned where it lives. What is under
    # test here is the loop, and a page that says "(could not engrave)" carries
    # everything this asks about.
    monkeypatch.setattr(report, "_engrave", lambda *a, **k: "")
    monkeypatch.setattr(report, "_bar_detail_row", lambda *a, **k: "")

    picture = tmp_path / "c.png"
    Image.new("RGB", (40, 20), "white").save(picture)
    score = tmp_path / "c.musicxml"
    score.write_text(PAGE)
    built = cases.Case("c", picture, score, "fixture")
    monkeypatch.setattr(cases, "resolve", lambda names: [built])
    monkeypatch.setattr(cases, "committed_cases", lambda: [built])
    monkeypatch.setattr(main, "parse", lambda case, fp: score)

    scores = {}

    def score_it(agree, pitch):
        scores["result"] = Result(agree=agree, pitch=pitch, staves_page=1,
                                  staves_homr=1)
        monkeypatch.setattr(main, "compare_output",
                            lambda *a, **k: scores["result"])

    return main, manifest_path, score_it


def _remember(path, readings):
    held = json.loads(path.read_text())
    cases_held = held.setdefault("cases", {})
    moved = []
    for name, reading in readings.items():
        entry = cases_held.setdefault(name, {})
        if all(entry.get(f) == reading[f] for f in references.MEMORY):
            continue
        entry.update(reading)
        moved.append(name)
    path.write_text(json.dumps(held))
    return sorted(moved)


def test_a_run_records_a_first_sighting_then_gates_against_it(harness):
    """The migration path, and the two runs it takes.

    A case nobody has accepted is recorded as it stands and the run passes,
    because there was no claim to fall below. The next run is judged against it.
    """
    main, manifest_path, score_it = harness

    score_it(agree=9, pitch=1)
    assert main.run_cases(["c"], "fixtures") == 0
    assert json_memory(manifest_path)["c"]["score"] == 90.0

    # Standing exactly where it was accepted: still a pass, nothing rewritten.
    assert main.run_cases(["c"], "fixtures") == 0
    assert json_memory(manifest_path)["c"]["score"] == 90.0


def test_a_run_that_reads_worse_exits_non_zero_and_writes_nothing(harness, capsys):
    """The alarm, and the thing that makes it an alarm rather than a shrug.

    A fall must not be recorded, or the gate would re-arm itself one notch lower
    every run and the second run would pass — which is the same defect as
    ratcheting before judging, arriving from the other side.
    """
    main, manifest_path, score_it = harness

    score_it(agree=9, pitch=1)
    assert main.run_cases(["c"], "fixtures") == 0

    score_it(agree=7, pitch=3)
    assert main.run_cases(["c"], "fixtures") == 1
    assert json_memory(manifest_path)["c"]["score"] == 90.0
    said = capsys.readouterr().out
    assert "GATE FAILED" in said
    assert "70.00%" in said and "90.00%" in said

    # ...and it keeps failing, rather than the fall having become the new normal.
    assert main.run_cases(["c"], "fixtures") == 1


def test_an_improvement_is_written_where_the_next_run_will_hold_it(harness, capsys):
    main, manifest_path, score_it = harness

    score_it(agree=7, pitch=3)
    main.run_cases(["c"], "fixtures")
    score_it(agree=10, pitch=0)
    assert main.run_cases(["c"], "fixtures") == 0

    assert json_memory(manifest_path)["c"]["score"] == 100.0
    assert "now remembered higher" in capsys.readouterr().out
    # And falling back to where it used to be acceptable now fails.
    score_it(agree=7, pitch=3)
    assert main.run_cases(["c"], "fixtures") == 1


def test_a_redirected_run_does_not_write_to_the_real_series(harness, tmp_path):
    """It did, while this file was being written, and committed nine runs.

    `series.runs` reads `SERIES` at call time and says in its own docstring why.
    `append`, `record_run` and `quality.write` bound it as a default instead, so
    a test that redirected the harness still *wrote* the host's committed record
    and rewrote `QUALITY.md` with invented numbers. Reading the wrong file is a
    test that means nothing; writing it is worse.
    """
    main, _, score_it = harness

    score_it(agree=9, pitch=1)
    main.run_cases(["c"], "fixtures")

    assert (tmp_path / "series.jsonl").exists()
    assert (tmp_path / "QUALITY.md").exists()
    assert len(series.runs(tmp_path / "series.jsonl")) == 1


def test_the_series_path_is_read_when_it_is_used_not_when_it_is_defined(
        tmp_path, monkeypatch):
    """The mechanism behind the test above, checked on its own.

    A default bound at import cannot be redirected at all, so this is what makes
    the redirection real rather than decorative.
    """
    from fixturecheck import quality

    monkeypatch.setattr(series, "SERIES", tmp_path / "elsewhere.jsonl")
    monkeypatch.setattr(quality, "QUALITY", tmp_path / "ELSEWHERE.md")

    series.append({"at": "now", "harness": "fixturecheck", "cases": {}})
    quality.write()

    assert (tmp_path / "elsewhere.jsonl").exists()
    assert (tmp_path / "ELSEWHERE.md").exists()


def test_the_case_page_lists_what_is_still_wrong_and_names_no_perfection(harness):
    """The stage-1 error list, where the operator actually reads it."""
    main, manifest_path, score_it = harness
    from fixturecheck import report

    score_it(agree=8, pitch=2)
    main.run_cases(["c"], "fixtures")

    page = (report.OUT / "c.html").read_text()
    assert "What is still wrong here" in page
    assert "2 note(s) at the wrong pitch" in page
    assert "perfect" not in page.lower()
