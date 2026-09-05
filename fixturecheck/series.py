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
               extra: dict | None = None, path: Path = SERIES) -> dict:
    """Write one run of one harness into the series."""
    run = {
        "at": now(),
        "harness": harness,
        "tier": tier,
        "homr": os.environ.get("FIXTURECHECK_HOMR") or homr_commit(),
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
