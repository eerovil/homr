"""Run the check over one case, the sample, or everything -- and always report.

    python -m fixturecheck one laulun-aika-s2      ~20s
    python -m fixturecheck ten                     ~4 min
    python -m fixturecheck all                     ~35 min
    python -m fixturecheck status                  instant -- the last run
    python -m fixturecheck freeze                  fingerprint the references

Every run **records**. `check-report/results.json` used to hold the last run and
only the last run, so a three-case run overwrote a sweep of ninety-eight and the
harness could not answer "is this getting better" at all. Runs now append to
`fixturecheck/series.jsonl`, which is committed, and rewrite `QUALITY.md` from
it -- the answer to "how good is it now", readable without running anything.

**The five committed fixtures are a gate.** They are small single systems this
repository owns outright and they are expected to be perfect; below 100% the run
exits non-zero. The eighty-eight song systems are not gated: their references
are derived from cleaned scores that are themselves sometimes wrong, and gating
on those would be gating on our own transcription.

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


def gate_over(records: list[series.CaseRecord], committed: set[str]) -> dict:
    """The committed fixtures, which are expected to be perfect.

    Only the fixtures in this run are judged -- a run of one song system has no
    opinion about the five, and reporting a gate it did not evaluate would be
    worse than reporting none.
    """
    judged = [r for r in records if r.name in committed]
    if not judged:
        # No opinion, and `None` is how that is said. Returning `passed: True`
        # here was a real bug: a song-only run would then record a gate nobody
        # evaluated, and the summary would replace a standing FAIL with "all 0
        # committed fixtures are perfect".
        return None
    failing = [r.name for r in judged if not r.counts.get("perfect")]
    return {"fixtures": len(judged), "perfect": len(judged) - len(failing),
            "failing": sorted(failing), "passed": not failing}


def run_cases(names: list[str], tier: str) -> int:
    fingerprint = code_fingerprint()
    committed = {case.name for case in cases.committed_cases()}
    standing = series.previous_cases("fixturecheck")
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
        page = report.case_page(case, parsed, result, before)
        entries.append({"name": case.name, "page": page, "score": result.score,
                        "agree": result.agree, "voice": result.voice,
                        "pitch": result.pitch, "size": result.size,
                        "timing": result.timing, "structure": result.structure,
                        "staves_page": result.staves_page,
                        "staves_homr": result.staves_homr,
                        "at_fault": result.at_fault, "meter": result.meter,
                        "unison": result.unison, "before": before})

        counts = {k: getattr(result, k) for k in series.COUNTS}
        counts["perfect"] = result.perfect
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
        print(f"  {case.name}: {result.agree} agree, {result.voice} voice, "
              f"{result.pitch} pitch, {result.size} count, "
              f"{result.timing} beat{meter}{staves}{moved}")

    gate = gate_over(records, committed)
    moved = references.drift(built)
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
    print(f"recorded in {series.SERIES.name}; summary in {quality.QUALITY.name}")

    if gate and not gate["passed"]:
        print(f"\nGATE FAILED: {', '.join(gate['failing'])} "
              f"— the committed fixtures are expected to be perfect")
        return 1
    return 0


def main() -> int:
    tier = sys.argv[1] if len(sys.argv) > 1 else "ten"

    if tier == "status":
        print(quality.render())
        return 0
    if tier == "freeze":
        wanted = sys.argv[2:] or cases.every()
        manifest = references.write(cases.resolve(wanted))
        print(f"froze {len(manifest['cases'])} reference(s), "
              f"digest {manifest['digest']} -> {references.MANIFEST.name}")
        return 0

    if tier == "one":
        names = sys.argv[2:]
    elif tier == "ten":
        names = cases.sample()
    elif tier == "all":
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
