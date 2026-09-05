"""The answer to "how good is it now", written down so nobody has to run it.

The card this comes from is called *make the quality of the current homr
readable without running anything*, and the reason it could not be was that the
only artefact was HTML on one host, overwritten by whatever ran last. Answering
took an environment, thirty-five minutes, and an ssh session.

So every run rewrites **QUALITY.md** from the series, and Markdown rather than a
page to host: GitHub renders it on a phone over Tailscale and on a desktop, at
no cost and with no service to keep alive. The HTML report is still written and
is still where you go to look at the music; this is the part you can read from
a chairlift.

It names **which homr** it is describing. That was the other half of the
complaint: "the current homr" meant the fork's `main` to one reader and the venv
the choir actually sings from to another, and no number anywhere said which.
"""

from __future__ import annotations

from pathlib import Path

from fixturecheck import report, series

QUALITY = Path(__file__).resolve().parent.parent / "QUALITY.md"

#: Said in full, every time, because the number above it invites exactly this
#: conclusion and the conclusion is wrong. Three things get called quality here
#: and this file measures one of them.
NOT_MEASURED = """\
## What this does not measure

**Whether the choir gets a correct practice track.** That is the question that
matters and nothing here answers it. What is measured is whether homr's output
matches a reference for the same printed system — one stage earlier than the
score anybody sings from, and several stages earlier than a practice video.
Everything `clean_score` does afterwards is unmeasured, and so is every repair a
person made by hand.

**Detection.** Noteheads and stems found in the picture are a different layer
from the MusicXML homr writes, and they disagree in both directions: a missed
head still reaches the output at the right pitch. There is no ground truth for
detection anywhere in this repository, so there is no number for it — only
`detection_diff.py`, which compares two runs to each other. Three reports were
filed in one day claiming homr had misread music when what had been compared was
the detector; keeping the layers apart is deliberate.
"""


def _percent(run: dict) -> str:
    head = run.get("headline", {})
    return f"{head.get('percent', 0):.1f}%"


def _trend(harness: str, path: Path) -> list[dict]:
    return [run for run in series.runs(path) if run.get("harness") == harness]


def _last_evaluated_gate(runs_of: list[dict]) -> tuple[dict | None, dict | None]:
    """The most recent run that actually judged a committed fixture.

    **Not simply the most recent run.** Most runs judge no fixture at all — a
    song system, a retry of one case — and treating "this run had nothing to say
    about the gate" as "the gate is fine" is how a standing FAIL gets quietly
    replaced by `all 0 committed fixtures are perfect`. A gate is a claim about
    the last time anybody looked, so that is the run it is read from, and the
    summary says which run that was.
    """
    for run in reversed(runs_of):
        gate = run.get("gate")
        if gate and gate.get("fixtures"):
            return gate, run
    return None, None


def _gate_line(runs_of: list[dict]) -> str:
    gate, from_run = _last_evaluated_gate(runs_of)
    if not gate:
        return ("_not evaluated_ — no recorded run has judged a committed fixture.")
    when = from_run.get("at", "")
    newest = runs_of[-1] if runs_of else from_run
    stale = ("" if newest is from_run
             else f" Carried forward: the run of {newest.get('at', '')} judged none.")
    if gate.get("passed"):
        return (f"**pass** — all {gate['fixtures']} committed fixtures were perfect "
                f"when last judged, {when}.{stale}")
    failing = ", ".join(f"`{name}`" for name in gate.get("failing", [])) or "unknown"
    return (f"**FAIL** — {gate.get('perfect', 0)}/{gate['fixtures']} perfect as of "
            f"{when}; below 100%: {failing}.{stale}")


def render(path: Path = series.SERIES) -> str:
    """The whole summary, from the series and nothing else."""
    fixture_runs = _trend("fixturecheck", path)
    bench_runs = _trend("choir-bench", path)

    lines = ["# How good is the scanning right now?", ""]
    if not fixture_runs and not bench_runs:
        lines += ["Nothing has been recorded yet. Run `python -m fixturecheck ten`.", ""]
        return "\n".join(lines) + "\n" + NOT_MEASURED

    lines += [
        "Regenerated from `fixturecheck/series.jsonl` on every run — do not edit "
        "by hand. Each number names the homr it measured and the state of the "
        "references it measured against, because those move independently and a "
        "score that improved because a reference was corrected is not homr "
        "improving.",
        "",
        "## Now",
        "",
        "| harness | measured | homr | references | of everything judged, right | cases |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for label, runs_of in (("fixturecheck", fixture_runs), ("choir-bench", bench_runs)):
        if not runs_of:
            lines.append(f"| `{label}` | _never_ | — | — | — | — |")
            continue
        run = runs_of[-1]
        tally = run.get("outcomes", {})
        read = tally.get(series.READ, 0)
        lost = tally.get(series.UNREADABLE, 0) + tally.get(series.UNBUILDABLE, 0)
        cases = f"{read} read" + (f", **{lost} not read**" if lost else "")
        lines.append(
            f"| `{label}` | {run.get('at', '')} ({run.get('tier', '')}) "
            f"| `{run.get('homr', '?')}` | `{run.get('references', '?')}` "
            f"| **{_percent(run)}** | {cases} |")

    if report.URL:
        lines += [
            "",
            f"**[Look at the music]({report.URL.rstrip('/')}/index.html)** — the "
            "printed band, homr's engraving and the reference, system by system. "
            "A count can say a system agrees on staves, bars and noteheads and "
            "still not say whether the parse is the music.",
        ]

    lines += [
        "",
        "The two are **not averaged**. `fixturecheck` scores notes across the "
        "printed systems of the repertoire; `choir-bench` scores staves and bars "
        "across the public-domain benchmark pages. They answer different "
        "questions and a single figure over both would mean nothing.",
        "",
        "## The gate",
        "",
        _gate_line(fixture_runs),
        "",
        "The five committed fixtures are small single systems this repository "
        "owns outright, and they are expected to be **perfect**. Anything less "
        "is a failure of the run, not a row in a table.",
        "",
    ]

    if len(fixture_runs) > 1:
        lines += ["## Over time", "",
                  "| when | harness | tier | homr | references | right |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for run in (fixture_runs + bench_runs)[-12:]:
            lines.append(
                f"| {run.get('at', '')} | `{run.get('harness', '')}` "
                f"| {run.get('tier', '')} | `{run.get('homr', '?')}` "
                f"| `{run.get('references', '?')}` | {_percent(run)} |")
        lines += ["",
                  "A tier is not a sample of the one above it — a ten-case run "
                  "and a full sweep are different populations, so read a "
                  "percentage against runs of the same tier.",
                  ""]

    return "\n".join(lines) + "\n" + NOT_MEASURED


def write(path: Path = series.SERIES, into: Path = QUALITY) -> Path:
    into.write_text(render(path))
    return into
