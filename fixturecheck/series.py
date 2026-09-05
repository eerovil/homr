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

import hashlib
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


def runs(path: Path | None = None) -> list[dict]:
    """Every run recorded, oldest first. A damaged line is skipped, not fatal.

    `SERIES` is read at call time, not bound as a default, so pointing the
    module at another file actually moves every reader. Bound as a default it
    did not, and a test that redirected the series still read the host's real
    one -- which is the sort of thing that passes and means nothing.
    """
    path = path or SERIES
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


def published_gate(runs_of: list[dict], under: tuple | None = None) -> dict | None:
    """The gate over **all** the committed fixtures, not over one run's worth.

    Reading it off the newest run that judged anything was still wrong, and the
    hole is bigger than the zero-fixture one it replaced: `fixturecheck one
    system4` judges one fixture, passes it, and publishes `1/1 perfect` — so a
    standing `FAIL — 3/5` goes green because somebody re-ran a fixture that was
    never the problem. Both of the failing ones are still failing and nothing
    said so.

    A gate is a claim about the whole set, so it is built from the whole set:
    each committed fixture's **own latest standing result**, taken across the
    series. A run can then only ever move the fixtures it actually ran, which is
    the property that was missing. A fixture nobody has judged is not a pass —
    it is counted as unevaluated and holds the gate open, because "we have never
    looked" and "we looked and it was fine" are not the same claim.

    The roster comes out of the runs themselves (`committed`), so the summary
    does not need the fixtures on disk and an old series stays readable.

    **And it is a claim about one homr.** Only results measured under the
    identity being reported count — `(homr, references)`, the newest run's
    unless told otherwise. Aggregating across identities let a 5/5 pass under
    homr A stay a published pass after a single fixture was re-run under homr B,
    with the other four never tested on B at all; a reference change does the
    same. A fixture not yet judged under the current identity is therefore
    `unevaluated` and holds the gate open, exactly like one nobody has ever
    judged, because "it passed on the old homr" is not a claim about this one.
    """
    roster: list[str] = []
    for run in runs_of:
        if run.get("committed"):
            roster = list(run["committed"])
    if not roster:
        return None
    if under is None:
        under = current_identity(runs_of)

    latest: dict[str, tuple[bool, str]] = {}
    for run in runs_of:
        if identity(run) != under:
            continue
        for name in roster:
            case = run.get("cases", {}).get(name)
            if case and case.get("outcome", READ) == READ and "perfect" in case:
                latest[name] = (bool(case["perfect"]), run.get("at", ""))

    failing = sorted(n for n in roster if n in latest and not latest[n][0])
    unevaluated = sorted(n for n in roster if n not in latest)
    return {
        "fixtures": len(roster),
        "perfect": sum(1 for n in roster if latest.get(n, (False,))[0]),
        "failing": failing,
        "unevaluated": unevaluated,
        "passed": not failing and not unevaluated,
        "as_of": max((when for _, when in latest.values()), default=""),
        "homr": under[0],
        "references": under[1],
    }



def origin(path: Path | None = None) -> dict:
    """Which history this is — as an identity, not as a name.

    The series is **per checkout and committed** — deliberately, so the record
    travels with the code that made it — while the report folder is fixed and
    shared, so there is one address to serve. Both are right and neither should
    move, but together they leave a seam: two checkouts render into the same
    folder from two different histories, each page internally consistent, the
    URL alternating between them with nothing saying so.

    **The checkout's directory name does not identify a history**, which the
    first attempt at this got wrong. One checkout switching branches replaces
    `series.jsonl` with a different committed history and keeps its name; two
    checkouts on two hosts can share a basename. Either way the label matches,
    the warning stays silent, and the seam is exactly as open as before.

    So `series_id` is the **resolved path** and a **root fingerprint** — the
    first run ever recorded — hashed together. Both halves earn their place: the
    path separates two folders that share a name, and the root separates two
    histories that share a path. It is stable across appends, which it has to be
    or every ordinary run would cry wolf.

    `checkout`, `runs` and `last_at` come along, the first for display and the
    other two so `continues` can ask whether what a reader saw is still a prefix
    of what is here now.
    """
    path = path or SERIES
    found = runs(path)
    root = ""
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                root = hashlib.sha256(line.strip().encode()).hexdigest()[:16]
                break
    resolved = str(path.resolve())
    return {
        "checkout": resolved.split(os.sep)[-3] if os.sep in resolved else resolved,
        "path": resolved,
        "root": root,
        "series_id": hashlib.sha256(f"{resolved}\0{root}".encode()).hexdigest()[:16],
        "runs": len(found),
        "last_at": found[-1].get("at", "") if found else "",
        # **The whole history shown, hashed** — not its last run. `continues`
        # compares this against the first `runs` runs found next time, which is
        # what makes it a prefix check rather than an endpoint check.
        "prefix": prefix_fingerprint(found),
        # The last run alone, kept only so a marker written by an older version
        # still means something for one render. Nothing new compares it.
        "last_run": run_fingerprint(found[-1]) if found else "",
    }


def run_fingerprint(run: dict) -> str:
    """One run, as a value — so two runs can be told apart by what they are."""
    return hashlib.sha256(
        json.dumps(run, sort_keys=True).encode()).hexdigest()[:16]


def prefix_fingerprint(some: list[dict]) -> str:
    """A run of history, as one value.

    Every run, in order, canonically — so a change anywhere inside it changes
    this, and appending after it does not. Fingerprinting only the last run was
    the defect: `[A, B, C]` and `[A, X, C, D]` share their first run, their path
    and the run at the older endpoint, so an endpoint check calls them the same
    history while the middle of it has been replaced under a reader.
    """
    return hashlib.sha256(
        json.dumps(some, sort_keys=True).encode()).hexdigest()[:16]


def continues(previous: dict | None, path: Path | None = None) -> bool:
    """Is the history here still the one that render saw, carried on?

    An identity alone cannot answer this. Same folder, same first run, and the
    history rewritten after the point the last render reached — a branch that
    forked from a shared start and went its own way — reads as the same series
    by every stable fingerprint, because a fingerprint that noticed would also
    change on every ordinary append.

    So the question is asked as a **prefix**, and it has to be the whole prefix.
    Checking the run at the older endpoint is not the same thing and misses the
    ordinary shape of a rewrite: `[A, B, C]` against `[A, X, C, D]` agrees on
    the path, the first run and the run at that endpoint, and disagrees about
    what happened in between. So every run the last render showed is hashed
    together, and all of them must still be there, in order, unchanged.

    An append leaves them untouched and says nothing, which is the property that
    makes a banner worth reading.
    """
    if not previous:
        return True
    if previous.get("series_id") != origin(path)["series_id"]:
        return False
    seen = int(previous.get("runs", 0) or 0)
    if seen <= 0:
        return True
    here = runs(path)
    if len(here) < seen:
        return False        # shorter than what was already shown: rewritten
    was = previous.get("prefix")
    if was:
        return prefix_fingerprint(here[:seen]) == was
    # Markers written before the whole prefix was recorded. One render of the
    # weaker check rather than a rewrite nobody can evidence; the next marker
    # carries the prefix and this is never reached again.
    if previous.get("last_run"):
        return run_fingerprint(here[seen - 1]) == previous["last_run"]
    return here[seen - 1].get("at", "") == previous.get("last_at", "")


def identity(run: dict) -> tuple:
    """What a measurement is a measurement *of*: the homr, and the references.

    Two numbers taken under different homrs are not two measurements of one
    thing, and neither are two taken against references that have moved. This
    pair is therefore the unit an aggregate may be taken over, and rows outside
    it are history rather than evidence about now.
    """
    return (run.get("homr", ""), run.get("references", ""))


def current_identity(runs_of: list[dict]) -> tuple:
    """The identity a report is describing: the newest run's."""
    return identity(runs_of[-1]) if runs_of else ("", "")


def standing(harness: str, path: Path | None = None,
             under: tuple | None = None) -> dict:
    """Each case as it last stood, with when that was and **under which homr**.

    `previous_cases` answers what a case last measured; this also answers how
    long ago and what it was measured with, which is what makes a list of every
    case readable rather than a row of numbers with no idea what produced them.

    `under` keeps only the cases measured under one identity. That is not a
    filter for tidiness: aggregating across identities is how a 5/5 pass under
    one homr survived a single fixture re-run under the next, with the other
    four never tested — a number that does not say what it describes, which is
    the whole complaint this harness exists to answer. Without `under`,
    everything comes back and the caller is expected to know why it wants that.
    """
    held: dict = {}
    for run in runs(path):
        if run.get("harness") != harness:
            continue
        here = identity(run)
        if under is not None and here != under:
            continue
        for name, case in run.get("cases", {}).items():
            held[name] = (case, run.get("at", ""), here)
    return held


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
