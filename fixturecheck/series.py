"""Every run, kept — so "is this getting better?" can be asked at all.

`check-report/results.json` held the last run and only the last run. It was
overwritten by whatever ran next, including a three-case run, which is how a
sweep of ninety-eight systems was lost during the session that raised this. What
survived of a baseline was hand-copied into `/var/tmp` under invented names. So
there was no series, and the question this file exists for could not be asked.

**One run is one line.** JSONL, appended, committed: a run adds a line and
touches nothing above it, so the history diffs as what it is and two runs made
from different checkouts do not collide in the middle of a file.

**A run is keyed by both commits, not one.** The reference for a song system is
that song's cleaned score imploded back to the printed shape, and those scores
are edited — nineteen of them were wrong about their staves on the day this was
written. A number that improved because somebody fixed a cleaned score must not
read as homr improving, so every run records the homr it measured *and* the
state of the references it measured against (`references.py`).

**What a run keeps is deliberately less than what it shows.** Per case: the
counts, and the first three disagreements — enough to tell "homr misread this"
from "our reference is wrong here" months later without re-running the harness.
The whole note-by-note table is kept only for the committed fixtures, which are
five small systems this repository owns outright; for the songs it would be a
transcription of music this repository is not allowed to hold. See
`fixturecheck/README.md`, "What is committed".

**A fixture's table is written only when it changes.** The five are gated at
100%, so most runs would otherwise append the same clean table forever; a run
that matches the last recorded one says so and names it instead. A change is
then the only table in the file, which is the point.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERIES = Path(__file__).resolve().parent / "series.jsonl"

#: How many disagreements are kept per case. Three, because the question a
#: stored row answers is "which side is wrong here", and the first fault
#: usually settles it -- a reference that has lost a staff is wrong from its
#: first row, and a note homr misread is wrong on its own.
KEPT_FAULTS = 3

#: The counts a case carries. Same names the report uses, so nothing has to be
#: translated between what is shown and what is recorded.
COUNTS = ("agree", "voice", "pitch", "size", "timing", "unison",
          "staves_page", "staves_homr", "staves_printed")

#: A case that was not compared, and why. Kept apart from a case that scored
#: badly: "homr crashed on this" and "homr read it and got everything wrong"
#: are different failures, and neither is "nobody ran it". Skipping them --
#: which is what happened before -- made a case that stopped parsing look
#: exactly like a case that was not in the run.
READ = "read"
UNREADABLE = "unreadable"        # homr ran and produced nothing usable
UNBUILDABLE = "unbuildable"      # the case's own picture or reference is missing


def homr_commit() -> str:
    """The homr under test: its last commit, plus a mark if it is uncommitted.

    The last commit that touched `homr/` rather than `HEAD`, matching the parse
    cache's own key -- a commit to the harness does not change a parse and must
    not look in the series like a new measurement of new software.
    """
    head = subprocess.run(["git", "log", "-1", "--format=%h", "--", "homr"],
                          cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "diff", "--stat", "HEAD", "--", "homr"],
                           cwd=ROOT, capture_output=True, text=True).stdout
    return (head or "nogit") + ("+dirty" if dirty.strip() else "")


#: Where `scripts/install-homr.sh` puts the one install, and therefore what the
#: choir app runs when nothing says otherwise. Kept in step with
#: `src/song_app/omr.py`'s `DEFAULT_VENV` by hand, because this repository
#: cannot import that one.
DEFAULT_VENV = os.path.join(os.path.expanduser("~"), ".local", "share",
                            "musescore-choir-plugins", "homr-venv")


def installed_binary() -> str:
    """The homr the app would run: `HOMR_BIN`, else the venv the installer writes.

    The same order `omr.homr_binary` resolves in, and it has to be, because the
    point is to name *the binary that read the music*. Looking only at
    `HOMR_BIN` was the defect: the documented benchmark invocation sets nothing
    and lets the app find its own default, so the ordinary run recorded
    `unknown` while measuring a perfectly identifiable install.

    Falls back to the bare name, which resolves on `PATH` — and if there is no
    venv behind it there is nothing to read a revision out of, so that ends up
    `unknown`, honestly this time.
    """
    configured = os.environ.get("HOMR_BIN")
    if configured:
        return configured
    default = os.path.join(DEFAULT_VENV, "bin", "homr")
    return default if os.path.exists(default) else "homr"


def engine_revision(tree: str | None = None, binary: str | None = None) -> str:
    """The revision of a homr that is **not** this checkout.

    `homr_commit` above answers for the code running here, which is the right
    answer only when the harness and the engine are the same homr —
    `fixturecheck` runs it in its own interpreter, so for it they are.
    `choir-bench` they are not: it measures the installed venv, a worktree, or a
    pod, and keying its numbers to the checkout holding the harness would name a
    homr that did not produce them. Which is this card's own complaint, one
    level down.

    A worktree answers with **the last commit that touched `homr/`**, not with
    its `HEAD`, uncommitted work in `homr/` marked. Those are different commits
    and the difference is not hypothetical: the branch this was written on sits
    several commits past `ec41559` and every one of them is harness and report,
    so keying a benchmark to its `HEAD` would name a homr that has never existed.
    `homr_commit` and the parse cache already key this way, and all three have to
    agree or a cached parse and the run that used it are filed under different
    engines.

    An install answers with what pip wrote down: `direct_url.json` carries the
    commit, and the distribution's own version carries it too
    (`0.7.0.post103+ec41559`), which is the fallback when the install was not
    made from a URL. **Which install is found the way the app finds it** —
    `HOMR_BIN` if it is set, else the venv `scripts/install-homr.sh` writes.
    Requiring `HOMR_BIN` meant the documented invocation, which sets nothing and
    lets the app resolve its own default, recorded `unknown` every time: the
    ordinary case, the one this is for.

    `unknown` rather than a guess when none of it can be read. A run whose
    engine cannot be identified is still worth recording — the numbers happened
    — but not worth attributing to a revision nobody checked.
    """
    if tree:
        rev = subprocess.run(["git", "log", "-1", "--format=%h", "--", "homr"],
                             cwd=tree, capture_output=True, text=True).stdout.strip()
        if not rev:
            return "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "homr"],
                               cwd=tree, capture_output=True, text=True).stdout
        return rev + ("+dirty" if dirty.strip() else "")

    binary = binary or installed_binary()
    venv = os.path.dirname(os.path.dirname(binary)) if binary else ""
    if not venv or not os.path.isdir(venv):
        return "unknown"
    for info in sorted(Path(venv).glob("lib/python*/site-packages/homr-*.dist-info")):
        direct = info / "direct_url.json"
        if direct.exists():
            try:
                commit = json.loads(direct.read_text()).get(
                    "vcs_info", {}).get("commit_id", "")
                if commit:
                    return f"installed:{commit[:7]}"
            except (ValueError, OSError):
                pass
        version = info.name[len("homr-"):-len(".dist-info")]
        if "+" in version:
            return "installed:" + version.rsplit("+", 1)[1]
        return f"installed:{version}"
    return "unknown"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class CaseRecord:
    """One case in one run."""

    name: str
    outcome: str = READ
    counts: dict = field(default_factory=dict)
    at_fault: str = ""
    faults: list = field(default_factory=list)
    rows: list | None = None          # committed fixtures only
    rows_same_as: str | None = None   # ...or the run whose table this repeats

    def to_json(self) -> dict:
        kept: dict = {"outcome": self.outcome}
        if self.outcome != READ:
            return kept
        kept.update(self.counts)
        if self.at_fault:
            kept["at_fault"] = self.at_fault
        if self.faults:
            kept["faults"] = self.faults
        if self.rows is not None:
            kept["rows"] = self.rows
        elif self.rows_same_as:
            kept["rows_same_as"] = self.rows_same_as
        return kept


def row_json(row) -> dict:
    """One note-by-note row, as little of it as still reads as evidence."""
    return {"where": row.where, "page": row.page, "homr": row.homr, "kind": row.kind}


def first_faults(result, keep: int = KEPT_FAULTS) -> list[dict]:
    """The first few rows where the two sides disagree.

    The *disagreements*, not the first few rows. A system opens with the notes
    both sides got right, so the first three rows of an ordinary case say
    nothing at all; what a stored row is for is telling a real misreading from a
    reference that has gone wrong, and only a fault does that.
    """
    return [row_json(row) for row in result.rows
            if row.kind not in ("agree", "unison")][:keep]


def headline(cases: list[CaseRecord]) -> dict:
    """One number for this harness: of everything judged, how much was right.

    **Notes homr lost and beats it moved count against it.** The older score was
    `agree / (agree + voice + pitch)`, which asks only "of the notes homr
    wrote, how many are right" -- so a system whose notes were half missing, or
    whose every beat had walked out of step, could read 100%. Both are wrong
    parses and the singer meets both. They are in the denominator now, and the
    consequence is that no number here is comparable with one quoted before
    2026-09-05.

    A case homr could not read at all is not folded in as a zero -- it has no
    notes to be right or wrong about, and averaging a crash into an accuracy
    would hide it. It is counted separately, by `outcomes`.
    """
    read = [c for c in cases if c.outcome == READ]
    right = sum(c.counts.get("agree", 0) for c in read)
    # A case may state its own denominator. `choir-bench` does: it judges
    # staves, bars and note counts against the printed page, and calling those
    # "wrong pitches" to fit this vocabulary would put a made-up word in the
    # record. Where nothing says otherwise the denominator is the note counts.
    judged = sum(
        c.counts["judged"] if "judged" in c.counts
        else sum(c.counts.get(k, 0) for k in ("agree", "voice", "pitch", "size", "timing"))
        for c in read)
    return {"right": right, "judged": judged,
            "percent": round(100.0 * right / judged, 2) if judged else 0.0}


def outcomes(cases: list[CaseRecord]) -> dict:
    """How many cases were read, and how many were not — never silently dropped."""
    tally = {READ: 0, UNREADABLE: 0, UNBUILDABLE: 0}
    for case in cases:
        tally[case.outcome] = tally.get(case.outcome, 0) + 1
    return tally


def append(record: dict, path: Path = SERIES) -> dict:
    """Add one run. Nothing above it is rewritten, ever."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as writing:
        writing.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def runs(path: Path = SERIES) -> list[dict]:
    """Every run recorded, oldest first. A damaged line is skipped, not fatal."""
    if not path.exists():
        return []
    found = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            found.append(json.loads(line))
        except ValueError:
            continue
    return found


def latest(harness: str, path: Path = SERIES) -> dict | None:
    """The last run of one harness, which is what "how good is it now" means."""
    for run in reversed(runs(path)):
        if run.get("harness") == harness:
            return run
    return None


def previous_cases(harness: str, path: Path = SERIES) -> dict:
    """Each case as it last stood, taken across runs rather than from one.

    What moved since the last run of *the same case* is the question, and a run
    of ten cases does not say anything about the other eighty-three. Reading the
    series backwards gives every case its own last measurement, which is why
    this can replace the single overwritten file without losing what that file
    was for.
    """
    standing: dict = {}
    for run in runs(path):
        if run.get("harness") != harness:
            continue
        for name, case in run.get("cases", {}).items():
            standing[name] = case
    return standing


def last_rows(name: str, harness: str = "fixturecheck", path: Path = SERIES) -> tuple:
    """The last recorded note-by-note table for a case, and the run it came from.

    Follows a `rows_same_as` back to the run that actually holds the table, so a
    long stretch of unchanged runs costs one hop rather than a walk.
    """
    for run in reversed(runs(path)):
        if run.get("harness") != harness:
            continue
        case = run.get("cases", {}).get(name)
        if not case:
            continue
        if "rows" in case:
            return case["rows"], run.get("at", "")
        if "rows_same_as" in case:
            for older in runs(path):
                if older.get("at") == case["rows_same_as"]:
                    held = older.get("cases", {}).get(name, {})
                    if "rows" in held:
                        return held["rows"], older.get("at", "")
            return None, ""
    return None, ""


def record_run(harness: str, tier: str, cases: list[CaseRecord],
               references: str, gate: dict | None = None,
               extra: dict | None = None, homr: str | None = None,
               path: Path = SERIES) -> dict:
    """Write one run of one harness into the series.

    `homr` is the revision that actually read the music, and a caller that can
    know it must say so. The fallback below reads *this checkout*, which is only
    the right answer when the harness and the engine are the same code —
    `fixturecheck` runs homr in its own interpreter, so for it they are.
    `choir-bench` they are not: it can be measuring the installed venv, a
    worktree, or a pod, and letting it fall through here would key its numbers
    to a checkout that did not produce them. Which is this card's own complaint,
    one level down.
    """
    run = {
        "at": now(),
        "harness": harness,
        "tier": tier,
        "homr": homr or os.environ.get("FIXTURECHECK_HOMR") or homr_commit(),
        "references": references,
        "headline": headline(cases),
        "outcomes": outcomes(cases),
        "cases": {case.name: case.to_json() for case in cases},
    }
    if gate is not None:
        run["gate"] = gate
    if extra:
        run.update(extra)
    return append(run, path)
