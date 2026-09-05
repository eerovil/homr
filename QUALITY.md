# How good is the scanning right now?

Regenerated from `fixturecheck/series.jsonl` on every run — do not edit by hand. Each number names the homr it measured and the state of the references it measured against, because those move independently and a score that improved because a reference was corrected is not homr improving.

## Now

| harness | measured | homr | references | of everything judged, right | cases |
| --- | --- | --- | --- | --- | --- |
| `fixturecheck` | 2026-09-05T04:29:52+00:00 (one) | `71e6819` | `ff33f48a1bcaac77` | **95.8%** | 5 read |
| `choir-bench` | _never_ | — | — | — | — |

The two are **not averaged**. `fixturecheck` scores notes across the printed systems of the repertoire; `choir-bench` scores staves and bars across the public-domain benchmark pages. They answer different questions and a single figure over both would mean nothing.

## The gate

**FAIL** — 3/5 perfect; below 100%: `hanget-soi`, `sammon-ryosto`

The five committed fixtures are small single systems this repository owns outright, and they are expected to be **perfect**. Anything less is a failure of the run, not a row in a table.

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
