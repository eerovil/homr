#!/usr/bin/env python3
"""Run a homr worktree against the choir project's fixtures and say what changed.

The question this loop exists to answer is **accuracy on choral scores**: does a
change to homr read more of the staves, bars and notes a printed page actually
has? So every number here is a comparison against something written down by a
person — the printed staff count and bar ranges in ``fixtures/omr-benchmark/
pages.json``, and for B1a/B1b the note-level ground truth transcribed by hand.

Three targets, all optional (``--all`` runs the lot):

``--pytest``     the fork's own unit tests, in the worktree. Catches breakage;
                 says nothing about accuracy.
``--benchmark``  the three public-domain benchmark pages, read the way the app
                 reads them: one printed system at a time, bands padded, through
                 ``omr_systems``. Reports staves and bars per system against what
                 the page prints, and notes per staff against the truth table.
``--scan``       the whole Virta venhettä vie fixture (15 systems, 2 pages) end
                 to end, including ``assemble``. The slow one; it is what the app
                 does, seams included.

Baselines are the point. Write one from the installed homr, then compare:

    scripts/choir-bench.py --benchmark -o baseline.json          # no --tree
    scripts/choir-bench.py --tree ~/homr-trees/slurs --benchmark \\
        -o slurs.json --compare baseline.json

**It runs under the choir app's interpreter, not homr's.** The reading is done by
``src/song_app`` (poppler crops, the per-system flattening, the slur repair), and
only the binary under test comes from the worktree — via ``HOMR_BIN``, exactly
the way the app reaches it. The script re-execs itself under
``$CHOIR_REPO/.venv/bin/python``, so it can be run with anything.

Needs poppler (``pdftoppm``) for the crops. A page is ~10-20s per system, so
``--benchmark`` is about 4 minutes and ``--scan`` about 4 more.
"""

from __future__ import annotations

import os
import sys

CHOIR_REPO = os.environ.get(
    "CHOIR_REPO", os.path.expanduser("~/musescore-choir-plugins")
)

# Re-exec under the choir app's venv before importing anything of its own.
if os.environ.get("_CHOIR_BENCH_REEXEC") != "1":
    _py = os.path.join(CHOIR_REPO, ".venv", "bin", "python")
    if not os.path.exists(_py):
        sys.exit(
            f"No choir app venv at {_py}. Set CHOIR_REPO to the checkout "
            "(default ~/musescore-choir-plugins)."
        )
    os.environ["_CHOIR_BENCH_REEXEC"] = "1"
    os.execv(_py, [_py, os.path.abspath(__file__), *sys.argv[1:]])

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from typing import Dict, List, Optional  # noqa: E402

sys.path.insert(0, CHOIR_REPO)
os.chdir(CHOIR_REPO)

from src.song_app import omr_systems, pdf_systems, scan  # noqa: E402
from src.song_app.tests import benchmark  # noqa: E402

FIXTURE_DIR = os.path.join(CHOIR_REPO, "fixtures", "virta-venhetta-vie")
FIXTURE_PDF = os.path.join(FIXTURE_DIR, "00-registered", "Virta venhettä vie.pdf")
FIXTURE_BOUNDS_DIR = os.path.join(FIXTURE_DIR, "10-cleaned")


def log(message: str) -> None:
    print(message, flush=True)


# --- counting what came back ---------------------------------------------


def notes_per_measure(staff: omr_systems.Staff) -> List[int]:
    """Note events per bar of one staff — noteheads, rests excluded.

    Chord noteheads count one each, and both voices of a divisi staff are
    included, because that is what the truth table's ``notes`` column counts.
    """
    counts = []
    for measure in staff.measures:
        counts.append(
            sum(1 for n in measure.findall("note") if n.find("rest") is None)
        )
    return counts


def truth_notes(page: benchmark.BenchmarkPage) -> Dict[int, Dict[int, int]]:
    """``{staff: {measure: notes}}`` from the hand transcription, if there is one."""
    out: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for row in page.truth():
        out[row.staff][row.measure] += row.notes
    return {s: dict(m) for s, m in out.items()}


# --- the benchmark pages -------------------------------------------------


def run_benchmark(out_root: str, only: Optional[List[str]] = None) -> dict:
    result = {}
    for page in benchmark.pages():
        if only and page.id not in only:
            continue
        log(f"\n=== {page.id}: {page.title}")
        log(f"    page prints {page.staves} staves, {page.bars} bars "
            f"in {len(page.systems)} systems")
        out_dir = os.path.join(out_root, page.id)
        os.makedirs(out_dir, exist_ok=True)
        started = time.time()

        bands = [scan.padded(b) for b in page.systems]
        images = pdf_systems.crop_systems(page.pdf, bands, out_dir, dpi=omr_systems.SCAN_DPI)

        truth = truth_notes(page)
        systems = []
        for image, band in zip(images, page.systems):
            entry = {
                "index": image.index,
                "expected_staves": page.staves,
                "expected_bars": band.measure_end - band.measure_start + 1,
                "measure_start": band.measure_start,
            }
            try:
                # queue=False: this loop is one job the operator is watching, and
                # taking a heavy slot per system would make it queue behind itself
                # on a busy host. The app's own scan does take them.
                read = omr_systems.read_system(image, out_dir, log=lambda _m: None,
                                               queue=False)
            except Exception as exc:                      # noqa: BLE001
                entry["error"] = f"{type(exc).__name__}: {exc}"
                log(f"    system {image.index}: FAILED — {entry['error']}")
            else:
                entry["staves"] = read.width
                entry["bars"] = read.bars
                entry["notes"] = [notes_per_measure(s) for s in read.staves]
                flag = "" if read.width == page.staves else "  <-- staff count"
                log(f"    system {image.index}: {read.width} staves, "
                    f"{read.bars} bars{flag}")
                if truth:
                    _report_truth(entry, band, truth)
            systems.append(entry)

        result[page.id] = {
            "title": page.title,
            "expected_staves": page.staves,
            "expected_bars": page.bars,
            "seconds": round(time.time() - started, 1),
            "systems": systems,
            "score": _score(page, systems, truth),
        }
        log(f"    {result[page.id]['score']}  ({result[page.id]['seconds']}s)")
    return result


def _report_truth(entry: dict, band: pdf_systems.SystemBounds,
                  truth: Dict[int, Dict[int, int]]) -> None:
    """Attach and print notes-found vs notes-printed, bar by bar."""
    rows = []
    for staff_no, counts in enumerate(entry["notes"], start=1):
        want = truth.get(staff_no, {})
        for offset, found in enumerate(counts):
            measure = band.measure_start + offset
            rows.append({
                "staff": staff_no,
                "measure": measure,
                "found": found,
                "truth": want.get(measure),
            })
    entry["truth_rows"] = rows
    scored = [r for r in rows if r["truth"] is not None]
    if scored:
        exact = sum(1 for r in scored if r["found"] == r["truth"])
        found = sum(r["found"] for r in scored)
        want = sum(r["truth"] for r in scored)
        log(f"        notes {found}/{want}, {exact}/{len(scored)} bars exact")


def _score(page: benchmark.BenchmarkPage, systems: List[dict],
           truth: Dict[int, Dict[int, int]]) -> str:
    read = [s for s in systems if "staves" in s]
    right_staves = sum(1 for s in read if s["staves"] == s["expected_staves"])
    right_bars = sum(1 for s in read if s["bars"] == s["expected_bars"])
    parts = [
        f"{len(read)}/{len(systems)} systems read",
        f"{right_staves} with the printed staff count",
        f"{right_bars} with the printed bar count",
    ]
    if truth:
        rows = [r for s in read for r in s.get("truth_rows", []) if r["truth"] is not None]
        exact = sum(1 for r in rows if r["found"] == r["truth"])
        parts.append(f"{exact}/{len(rows)} bars note-exact")
    return "; ".join(parts)


def record_benchmark(pages: dict, tree: Optional[str], in_pod: bool) -> None:
    """Put this run in the series, beside `fixturecheck`'s and not mixed into it.

    The two harnesses answer different questions -- this one scores staves, bars
    and a little note-level truth across three public-domain pages, the other
    scores notes across the printed systems of the whole repertoire -- so they
    are recorded under separate harness names and never averaged. A baseline
    written to a file under an invented name in `/var/tmp`, which is how this was
    kept before, is not a series and cannot be asked a question about months.

    A system homr could not read is recorded as unread rather than as a zero,
    for the same reason it is in `fixturecheck`: a crash and a bad parse are
    different failures.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fixturecheck import quality, series

    records = []
    for page_id, page in pages.items():
        for system in page.get("systems", []):
            name = f"{page_id}-s{system['index']}"
            if "staves" not in system:
                records.append(series.CaseRecord(name, outcome=series.UNREADABLE))
                continue
            # Two checks the page itself settles, plus one per bar of hand
            # transcribed truth where there is any.
            checks = [system["staves"] == system["expected_staves"],
                      system["bars"] == system["expected_bars"]]
            checks += [row["found"] == row["truth"]
                       for row in system.get("truth_rows", [])
                       if row.get("truth") is not None]
            records.append(series.CaseRecord(name, counts={
                "agree": sum(1 for ok in checks if ok),
                "judged": len(checks),
                "staves_page": system["expected_staves"],
                "staves_homr": system["staves"],
                "perfect": all(checks),
            }))

    where = "pod" if in_pod else (os.path.basename(tree) if tree else "installed")
    # This harness's reference is the benchmark manifest, which is committed in
    # the choir repository -- so unlike `fixturecheck`'s it is already frozen,
    # and its content hash is enough to key a run by.
    manifest = os.path.join(CHOIR_REPO, "fixtures", "omr-benchmark", "pages.json")
    stamp = "missing"
    if os.path.exists(manifest):
        import hashlib
        with open(manifest, "rb") as reading:
            stamp = hashlib.sha256(reading.read()).hexdigest()[:16]
    run = series.record_run("choir-bench", "benchmark", records,
                            references=stamp,
                            extra={"engine": where})
    quality.write()
    head = run["headline"]
    log(f"\nrecorded: {head['percent']:.1f}% of {head['judged']} checks "
        f"(homr {run['homr']}, {where}) -> {series.SERIES.name}, {quality.QUALITY.name}")


# --- the fixture, end to end ---------------------------------------------


def run_scan(out_root: str) -> dict:
    log("\n=== Virta venhettä vie (the song fixture), whole score")
    bounds = pdf_systems.load_bounds(FIXTURE_BOUNDS_DIR)
    log(f"    {len(bounds)} printed systems")
    out_dir = os.path.join(out_root, "fixture")
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()

    bands = [scan.padded(b) for b in bounds]
    images = pdf_systems.crop_systems(FIXTURE_PDF, bands, out_dir,
                                      dpi=omr_systems.SCAN_DPI)
    scans, systems = [], []
    for image in images:
        try:
            read = omr_systems.read_system(image, out_dir, log=lambda _m: None,
                                           queue=False)
        except Exception as exc:                          # noqa: BLE001
            systems.append({"index": image.index, "error": f"{type(exc).__name__}: {exc}"})
            log(f"    system {image.index}: FAILED — {exc}")
            continue
        scans.append(read)
        systems.append({"index": image.index, "staves": read.width, "bars": read.bars})
        log(f"    system {image.index}: {read.width} staves, {read.bars} bars")

    assembled = None
    if scans and len(scans) == len(images):
        assembled = os.path.join(out_dir, "scanned.musicxml")
        omr_systems.assemble(scans, assembled)
        log(f"    assembled -> {assembled}")

    holes = [s for s in systems if "error" in s]
    result = {
        "systems": systems,
        "holes": len(holes),
        "bars": sum(s.get("bars", 0) for s in systems),
        "assembled": assembled,
        "seconds": round(time.time() - started, 1),
    }
    log(f"    {len(systems) - len(holes)}/{len(systems)} systems, "
        f"{result['bars']} bars, {result['seconds']}s")
    return result


# --- the fork's own tests ------------------------------------------------


def run_pytest(tree: str) -> dict:
    log("\n=== homr unit tests")
    py = os.path.join(tree, ".venv", "bin", "python")
    proc = subprocess.run(
        [py, "-m", "pytest", "tests", "-q", "--no-header",
         # This one asserts the *checkout* has pre-commit hooks installed. A
         # worktree keeps its hooks in the main repo, so it fails in every tree
         # whatever the change under test does — noise, not a signal.
         "--deselect", "tests/test_poetry_config.py::test_isset_precommit_hooks"],
        cwd=tree, capture_output=True, text=True,
    )
    output = proc.stdout + proc.stderr
    tail = output.strip().splitlines()[-15:]
    for line in tail:
        log(f"    {line}")
    # Six of the fork's test modules import torch, which the worktree script
    # leaves out by default. Say so rather than leaving six collection errors
    # looking like a regression the change caused.
    if "No module named 'torch'" in output:
        log("    ^ these need torch. Rebuild the tree with HOMR_TEST_DEPS=1 to run them.")
    return {"returncode": proc.returncode, "tail": tail,
            "torch_missing": "No module named 'torch'" in output}


# --- comparing two runs --------------------------------------------------


def compare(new: dict, old: dict) -> None:
    log("\n=== against the baseline")
    for page_id, now in (new.get("benchmark") or {}).items():
        before = (old.get("benchmark") or {}).get(page_id)
        if not before:
            log(f"    {page_id}: not in the baseline")
            continue
        log(f"    {page_id}")
        log(f"        was  {before['score']}")
        log(f"        now  {now['score']}")
        for a, b in zip(before["systems"], now["systems"]):
            if a.get("staves") != b.get("staves") or a.get("bars") != b.get("bars"):
                log(f"        system {b['index']}: "
                    f"{a.get('staves')} staves/{a.get('bars')} bars -> "
                    f"{b.get('staves')}/{b.get('bars')}")
    if "scan" in new and "scan" in old:
        log(f"    fixture: {old['scan']['holes']} holes, {old['scan']['bars']} bars "
            f"-> {new['scan']['holes']} holes, {new['scan']['bars']} bars")


def _kubernetes_shim(tree: str | None) -> str:
    """Prepare the pod and return an executable that reads a page in it.

    Three steps, all in ``choir-k8s.sh``: build the pod and its venv if this is
    the first run, put the worktree's source in front of that venv, and write a
    shim honouring homr's own CLI. The shim is what goes in ``HOMR_BIN``, so
    nothing downstream knows the difference.

    The tree is shipped on **every** run rather than only when it changes: it is
    a few hundred KB of Python and a second of wall clock, and a sweep measuring
    source the pod does not actually have is the one failure worth spending a
    second to make impossible.
    """
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "choir-k8s.sh")
    run = [script, "up"]
    if subprocess.run(run).returncode:
        sys.exit("choir-k8s.sh up failed — is kubectl pointed at a cluster?")
    if tree and subprocess.run([script, "ship", tree]).returncode:
        sys.exit(f"could not ship {tree} to the pod")
    shim = os.path.join(tempfile.mkdtemp(prefix="choir-k8s-"), "homr")
    if subprocess.run([script, "shim", shim]).returncode:
        sys.exit("could not write the pod shim")
    return shim


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", help="worktree to test; omit to test the installed homr")
    ap.add_argument("--kubernetes", action="store_true",
                    help="run homr in a pod instead of on this host "
                         "(see scripts/choir-k8s.sh)")
    ap.add_argument("--all", action="store_true", help="every target")
    ap.add_argument("--pytest", action="store_true", help="the fork's unit tests")
    ap.add_argument("--benchmark", action="store_true", help="the benchmark pages")
    ap.add_argument("--scan", action="store_true", help="the fixture, end to end")
    ap.add_argument("--pages", help="comma-separated benchmark page ids (default: all)")
    ap.add_argument("--dpi", type=int, help=f"crop dpi (default {omr_systems.SCAN_DPI})")
    ap.add_argument("--keep", help="keep the crops and MusicXML in this directory")
    ap.add_argument("-o", "--out", help="write the numbers here as JSON")
    ap.add_argument("--compare", help="a previous --out file to diff against")
    args = ap.parse_args()

    targets = {
        "pytest": args.pytest or args.all,
        "benchmark": args.benchmark or args.all,
        "scan": args.scan or args.all,
    }
    if not any(targets.values()):
        ap.error("pick at least one of --pytest / --benchmark / --scan / --all")

    if args.dpi:
        omr_systems.SCAN_DPI = args.dpi

    if args.tree:
        tree = os.path.abspath(args.tree)
    else:
        tree = None

    if args.kubernetes:
        # The pod is reached the same way a worktree is: as an executable in
        # HOMR_BIN. Everything else here — the crops, the flattening, the
        # scoring — still runs on this host, because that is the choir app's
        # code and it is not what the sweep spends its cores on.
        binary = _kubernetes_shim(tree)
        os.environ["HOMR_BIN"] = binary
        log(f"homr under test: pod {os.environ.get('CHOIR_K8S_POD', 'homr-bench')}"
            + (f" running {os.path.basename(tree)}" if tree else " running its own install"))
    elif tree:
        binary = os.path.join(tree, ".venv", "bin", "homr")
        if not os.path.exists(binary):
            sys.exit(f"No homr in {tree} — run scripts/choir-worktree.sh first.")
        os.environ["HOMR_BIN"] = binary
        log(f"homr under test: {binary}")
    else:
        log(f"homr under test: {os.environ.get('HOMR_BIN', 'the installed venv')}")
    log(f"crops at {omr_systems.SCAN_DPI} dpi")

    if targets["pytest"] and not tree:
        sys.exit("--pytest needs --tree (the installed venv has no test suite).")
    if targets["pytest"] and args.kubernetes:
        # --kubernetes moves the page reading, which is what the sweep's cores
        # go on. The unit tests are the tree's own pytest and stay here; saying
        # so is better than appearing to have run them somewhere else.
        log("note: --pytest runs on this host; --kubernetes moves only the page reading")

    results: dict = {"homr": os.environ.get("HOMR_BIN"), "dpi": omr_systems.SCAN_DPI}
    scratch = args.keep or tempfile.mkdtemp(prefix="choir-bench-")
    os.makedirs(scratch, exist_ok=True)
    log(f"working in {scratch}")

    if targets["pytest"]:
        results["pytest"] = run_pytest(tree)
    if targets["benchmark"]:
        only = args.pages.split(",") if args.pages else None
        results["benchmark"] = run_benchmark(scratch, only)
    if targets["scan"]:
        results["scan"] = run_scan(scratch)

    if targets["benchmark"]:
        record_benchmark(results["benchmark"], tree, args.kubernetes)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        log(f"\nwrote {args.out}")
    if args.compare:
        with open(args.compare, encoding="utf-8") as f:
            compare(results, json.load(f))
    return 0


if __name__ == "__main__":
    sys.exit(main())
