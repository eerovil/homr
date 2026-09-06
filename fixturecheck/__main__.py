"""Run the check over one case, the sample, or everything -- and always report.

    python -m fixturecheck one laulun-aika-s2      ~20s
    python -m fixturecheck pins                    tier 1 -- the pinned failures
    python -m fixturecheck fixtures                tier 2 -- the five
    python -m fixturecheck corpus                  tier 3 -- everything, ~35 min
    python -m fixturecheck ten                     the written-down sample, ~4 min
    python -m fixturecheck status                  instant -- the last run
    python -m fixturecheck pin <name> <case> 3-5 <why>    cut a tier-1 pin
    python -m fixturecheck accept <case> ...       accept what it reads now
    python -m fixturecheck freeze                  fingerprint the references

Every run **records**. `check-report/results.json` used to hold the last run and
only the last run, so a three-case run overwrote a sweep of ninety-eight and the
harness could not answer "is this getting better" at all. Runs now append to
`fixturecheck/series.jsonl`, which is committed, and rewrite `QUALITY.md` from
it -- the answer to "how good is it now", readable without running anything.

**The gate is per case: nothing gets worse.** Each case remembers the
notes-right score it was last accepted at (`references.py`), a run exits
non-zero if any case reads below its own memory, and a case that improves has
its memory raised there and then. Three tiers -- the pinned failures, the five
committed fixtures, the whole corpus of song systems -- and one rule over all of
them, so a change that lifts the fixtures while costing real songs trips it.

That replaces a gate that asked whether each of the five was `perfect`. Two
findings killed that: #152 shaved two pixels off a band and took the verdict
from 7/10 to 3/10 while the notes moved 98.4% to 97.3%, and #147 reserved the
word "perfect" for the operator, judging by eye. **No machine here declares a
parse right.** What the operator reads instead is each case's score and the list
of what is still wrong with it.

The song systems are gated the same way and it is worth saying why that is safe
now, having been refused before: their references are derived from cleaned
scores that are themselves sometimes wrong, so *absolute* judgement of them
would be judging our own transcription. Judging each against its own last
reading is not that. A reference that gets corrected takes its case out of the
gate until somebody freezes it again, which is the same rule seen from the
other side.

The pytest gate is left alone and answers a different question: it says yes or
no, in CI, in seconds. This says how much, and shows the music.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fixturecheck import cases, quality, references, report, series  # noqa: E402
from fixturecheck.compare import compare_output  # noqa: E402

PARSES = cases.CACHE / "parses"


def parse(case: cases.Case, fingerprint: str) -> Path | None:
    """Read the case's picture with this homr, reusing the last read if it stands.

    Keyed on the code: a parse is only stale when homr changes, so editing the
    report or adding a case re-runs nothing. Nothing here shares a filename with
    a reference -- homr writes `<image>.musicxml`, which is exactly where the
    reference lives, and that silently destroyed ninety-three of them once.
    """
    PARSES.mkdir(parents=True, exist_ok=True)
    out = PARSES / f"{case.name}@{fingerprint}.musicxml"
    if out.exists():
        return out
    with tempfile.TemporaryDirectory(prefix="parse-") as tmp:
        copy = Path(tmp) / f"{case.name}.png"
        shutil.copy(case.image, copy)
        run = subprocess.run(
            [sys.executable, "-c", "from homr.main import main; main()",
             str(copy), "--gpu", "no"],
            capture_output=True, text=True, timeout=900,
            cwd=cases.ROOT)
        produced = copy.with_suffix(".musicxml")
        if run.returncode != 0 or not produced.exists():
            return None
        shutil.copy(produced, out)
    return out


def code_fingerprint() -> str:
    """What homr is right now: its own last commit, plus whatever is uncommitted.

    The last commit that touched `homr/`, and not `HEAD`. A parse is only stale
    when homr changes -- that is the promise the cache is worth anything for --
    and keying on `HEAD` broke it: committing to this very directory re-read all
    93 systems with a homr that had not moved a line, three quarters of an hour
    to arrive back at the same parses.
    """
    head = subprocess.run(["git", "log", "-1", "--format=%h", "--", "homr"], cwd=cases.ROOT,
                          capture_output=True, text=True).stdout.strip() or "nogit"
    dirty = subprocess.run(["git", "diff", "--stat", "HEAD", "--", "homr"], cwd=cases.ROOT,
                           capture_output=True, text=True).stdout
    if dirty.strip():
        import hashlib
        head += "+" + hashlib.sha256(dirty.encode()).hexdigest()[:6]
    return head


def gate_over(records: list[series.CaseRecord], adrift: list[str]) -> dict:
    """Every case in this run, against the reading it was last accepted at.

    Not the committed fixtures — **every** case, which is the whole of what
    "three tiers, same rule" comes to in code. A pin, one of the five, and one
    of the ninety-odd song systems are the same object here, so a change that
    lifts the fixtures while costing real songs trips this exactly as loudly as
    one that costs a fixture.

    Only cases with a memory are judged, and a run that judged none has no
    opinion: `None` is how that is said. Returning `passed: True` for a run that
    evaluated nothing was a real bug once — the summary replaced a standing
    failure with a pass nobody had measured.
    """
    read = {r.name: references.marks(r.counts)
            for r in records if r.outcome == series.READ}
    unread = [r.name for r in records if r.outcome == series.UNREADABLE]
    gate = references.judge(read, unread, adrift)
    if not gate["judged"] and not gate["unreadable"]:
        return None
    return gate


def ratchet(records: list[series.CaseRecord], memory: dict,
            adrift: list[str]) -> list[str]:
    """Move a case's memory up to a reading that is not worse than its own.

    **The ratchet only turns one way on its own.** An improvement is recorded
    here, in a committed file, so the next run has to hold on to it; a *fall* is
    never written by a run, because accepting one is a judgement somebody makes
    with the report open, and `accept` is where that judgement is expressed —
    named case by named case.

    A case nobody has accepted yet is recorded as it stands. That is not the
    gate passing it — there was nothing to pass — it is the first sighting, and
    it is what makes adding a case cost one run rather than a hand-edited file.

    A case whose reference has moved is left alone. Its old memory describes
    music that has been edited and its new reading has not been looked at; the
    person who edited the score is the one who should say which it is.
    """
    adrift = set(adrift)
    now: dict[str, dict] = {}
    for record in records:
        if record.outcome != series.READ or record.name in adrift:
            continue
        reading = references.marks(record.counts)
        was = memory.get(record.name)
        if was is None or references.better(reading, was):
            now[record.name] = reading
    return references.remember(now)


def run_cases(names: list[str], tier: str) -> int:
    fingerprint = code_fingerprint()
    committed = {case.name for case in cases.committed_cases()}
    standing = series.previous_cases("fixturecheck")
    memory = references.accepted()
    print(f"homr {fingerprint}: {len(names)} case(s)")

    entries: list[dict] = []
    records: list[series.CaseRecord] = []
    built: list = []

    for name in names:
        found = cases.resolve([name])
        if not found:
            print(f"  {name}: could not be built")
            records.append(series.CaseRecord(name, outcome=series.UNBUILDABLE))
            continue
        case = found[0]
        built.append(case)
        parsed = parse(case, fingerprint)
        if parsed is None:
            print(f"  {case.name}: homr could not read it")
            records.append(series.CaseRecord(case.name, outcome=series.UNREADABLE))
            continue

        result = compare_output(case.reference, parsed, case.name)
        before = standing.get(case.name)
        page = report.case_page(case, parsed, result, before, memory.get(case.name))
        entries.append({"name": case.name, "page": page, "score": result.score,
                        "agree": result.agree, "voice": result.voice,
                        "pitch": result.pitch, "size": result.size,
                        "timing": result.timing, "structure": result.structure,
                        "staves_page": result.staves_page,
                        "staves_homr": result.staves_homr,
                        "at_fault": result.at_fault, "meter": result.meter,
                        "unison": result.unison, "before": before})

        counts = {k: getattr(result, k) for k in series.COUNTS}
        record = series.CaseRecord(case.name, counts=counts,
                                   at_fault=result.at_fault,
                                   faults=series.first_faults(result))
        # The whole table, but only for the five this repository owns -- and only
        # when it differs from the last one recorded, or a gated fixture would
        # append the same clean table forever and a real change would be one
        # identical block among hundreds.
        if case.name in committed:
            table = [series.row_json(row) for row in result.rows]
            was, from_run = series.last_rows(case.name)
            if was == table and from_run:
                record.rows_same_as = from_run
            else:
                record.rows = table
        records.append(record)

        moved = ""
        if before:
            change = (result.voice + result.pitch) - (before.get("voice", 0)
                                                      + before.get("pitch", 0))
            moved = "  (no change)" if change == 0 else f"  ({change:+d} faults)"
        staves = (f", staves {result.staves_page} vs {result.staves_homr}"
                  if result.structure else "")
        # A misread meter is a wrong answer about the bars the notes are read
        # in, so it belongs on the line rather than only on the page.
        meter = f", {result.meter} bar(s) in the wrong meter" if result.meter else ""
        # A unison homr wrote into one voice is not a misreading and is not
        # scored, so the line said nothing at all about it -- and a case
        # carrying two of them read exactly like a case carrying none. It is
        # the part that is missing, not the note. See `Result.warnings`.
        warned = (f", {result.warnings} unison(s) written as one voice"
                  if result.warnings else "")
        print(f"  {case.name}: {result.agree} agree, {result.voice} voice, "
              f"{result.pitch} pitch, {result.size} count, "
              f"{result.timing} beat{meter}{warned}{staves}{moved}")

    moved = references.drift(built)
    # Judged before anything is written back, and against the memory this run
    # started with: a ratchet that raised a case's memory first would then find
    # it standing exactly at its memory and pass everything, every time.
    gate = gate_over(records, moved["changed"])
    raised = ratchet(records, memory, moved["changed"])
    # The roster goes in every run, not just one that judged a fixture: the
    # published gate is built from each committed fixture's own latest result
    # (`quality.published_gate`), so the summary has to know the whole set even
    # when this run touched none of it.
    extra = {"committed": sorted(committed)}
    if moved["changed"]:
        extra["reference_drift"] = moved
    run = series.record_run("fixturecheck", tier, records,
                            references=references.stamp(built), gate=gate,
                            extra=extra)
    quality.write()

    if entries:
        written = report.index_page(entries, tier, run)
        print("\n" + (f"{report.URL.rstrip('/')}/index.html" if report.URL
                      else str(written)))
    head = run["headline"]
    print(f"\n{head['percent']:.1f}% of {head['judged']} judged are right "
          f"(homr {run['homr']}, references {run['references']})")
    lost = (run["outcomes"].get(series.UNREADABLE, 0)
            + run["outcomes"].get(series.UNBUILDABLE, 0))
    if lost:
        print(f"{lost} case(s) were not read at all — recorded, not skipped")
    if moved["changed"]:
        print(f"references have moved since they were frozen: "
              f"{', '.join(moved['changed'])}\n"
              f"  run `python -m fixturecheck freeze` once you have looked at why")
    if raised:
        print(f"{len(raised)} case(s) now remembered higher: {', '.join(raised)}\n"
              f"  {references.MANIFEST.name} has changed — commit it with the change "
              f"that earned it")
    if gate and gate["adrift"]:
        print(f"{len(gate['adrift'])} case(s) held out of the gate, their "
              f"reference having moved: {', '.join(gate['adrift'])}")
    print(f"recorded in {series.SERIES.name}; summary in {quality.QUALITY.name}")

    if gate and not gate["passed"]:
        print("\nGATE FAILED — a case read worse than the reading it was "
              "accepted at:")
        for name, said in gate["below"].items():
            print(f"  {name}: {'; '.join(said)}")
        for name in gate["unreadable"]:
            print(f"  {name}: homr could not read it at all, and it has a "
                  f"remembered reading")
        return 1
    return 0


def accept(names: list[str]) -> int:
    """Record what the named cases read *now*, whichever way that moves them.

    **The escape hatch, and the gate is not honest without it.** A run moves a
    memory up and never down (`ratchet`), which is the right default and cannot
    be the only door: a regression is not always a mistake. An intentional
    trade-off in the model reads worse on some page, and with no way to say "yes,
    I meant that" the gate fails forever and the next person edits
    `references.json` by hand — which is the gate being routed around rather than
    used.

    This was documented as `freeze` and `freeze` cannot do it. That command is
    about the *files*: it keeps a memory whose fingerprint has not moved, and it
    never reads a case, so it has no measurement to write. The two are different
    acts and they are different commands.

    **Cases are named, and never all of them.** The value of this path is that
    somebody chose the case and meant it; an `accept` that took no arguments and
    swallowed the whole run would be a button for making the alarm stop, which is
    the failure mode this project already has a name for.

    It re-reads rather than trusting the last run, because what is being written
    down is a measurement and the last run may have been of other code. The parse
    cache makes that cheap when nothing has moved.
    """
    fingerprint = code_fingerprint()
    memory = references.accepted()
    readings: dict[str, dict] = {}
    for name in names:
        found = cases.resolve([name])
        if not found:
            print(f"{name}: could not be built")
            return 1
        case = found[0]
        parsed = parse(case, fingerprint)
        if parsed is None:
            # Nothing to accept: there is no reading. Recording a zero here
            # would quietly retire the case, since nothing can fall below it.
            print(f"{case.name}: homr could not read it, so there is no reading "
                  f"to accept")
            return 1
        result = compare_output(case.reference, parsed, case.name)
        readings[case.name] = references.marks(
            {k: getattr(result, k) for k in series.COUNTS})

    moved = references.remember(readings)
    for name in sorted(readings):
        now, was = readings[name], memory.get(name)
        if was is None:
            print(f"  {name}: accepted at {now['score']:.2f}% "
                  f"(nothing was remembered before)")
        elif name in moved:
            way = "DOWN" if references.worse(now, was) else "up"
            print(f"  {name}: {way} from {was['score']:.2f}% to "
                  f"{now['score']:.2f}%")
        else:
            print(f"  {name}: unchanged at {now['score']:.2f}%")
    if moved:
        print(f"{references.MANIFEST.name} has changed — commit it with the "
              f"reason you accepted this")
    return 0


def make_pin(argv: list[str]) -> int:
    """`pin <name> <case> <first>-<last> <why>` — tier 1, in one command."""
    if len(argv) < 4:
        print("pin <name> <case> <first>-<last> <why it is being pinned>\n"
              "  e.g. pin hanget-m3-beats hanget-soi 3-3 "
              "'a duration read differently; the six notes land on the wrong beats'")
        return 2
    name, source_name, span, why = argv[0], argv[1], argv[2], " ".join(argv[3:])
    first, _, last = span.partition("-")
    if not first.isdigit() or not (last or first).isdigit():
        print(f"'{span}' is not a range of bars, like 3-5 or 3")
        return 2
    found = cases.resolve([source_name])
    if not found:
        print(f"{source_name}: could not be built, so there is nothing to cut")
        return 1
    try:
        pinned = cases.pin(name, found[0], int(first), int(last or first), why)
    except cases.CannotPin as refused:
        print(f"not pinned: {refused}")
        return 1
    print(f"pinned {pinned.name} from {found[0].name} bars {first}-{last or first}\n"
          f"  {pinned.image}\n  {pinned.reference}\n  registered in "
          f"{cases.PINS.name}\n"
          f"Run it to record what it reads at today — that reading, broken or "
          f"not, becomes the memory nothing may fall below:\n"
          f"  python -m fixturecheck one {pinned.name}")
    return 0


def main() -> int:
    tier = sys.argv[1] if len(sys.argv) > 1 else "ten"

    if tier == "status":
        print(quality.render())
        return 0
    if tier == "pin":
        return make_pin(sys.argv[2:])
    if tier == "accept":
        wanted = sys.argv[2:]
        if not wanted:
            # Deliberately no accept-all: see `accept`.
            print("accept <case> [<case> ...]  — name the cases whose current "
                  "reading you are accepting, including a fall")
            return 2
        return accept(wanted)
    if tier == "freeze":
        wanted = sys.argv[2:] or cases.every()
        manifest, forgotten = references.write(cases.resolve(wanted))
        print(f"froze {len(manifest['cases'])} reference(s), "
              f"digest {manifest['digest']} -> {references.MANIFEST.name}")
        if forgotten:
            print(f"{len(forgotten)} case(s) lost their remembered reading, their "
                  f"files having moved: {', '.join(forgotten)}\n"
                  f"  the next run over them records what they read now")
        return 0

    if tier == "one":
        names = sys.argv[2:]
    elif tier == "pins":
        names = [case.name for case in cases.pinned_cases()]
    elif tier == "fixtures":
        names = [case.name for case in cases.fixture_cases()]
    elif tier == "ten":
        names = cases.sample()
    elif tier in ("all", "corpus"):
        names = cases.every()
    else:
        names = [tier] + sys.argv[2:]
        tier = "one"
    if not names:
        print("nothing to check")
        return 0
    return run_cases(names, tier)


if __name__ == "__main__":
    sys.exit(main())
