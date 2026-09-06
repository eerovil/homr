# How good is the scanning right now?

Regenerated from `fixturecheck/series.jsonl` on every run — do not edit by hand. Each number names the homr it measured and the state of the references it measured against, because those move independently and a score that improved because a reference was corrected is not homr improving.

## Now

| harness | measured | homr | references | of everything judged, right | cases |
| --- | --- | --- | --- | --- | --- |
| `fixturecheck` | 2026-09-06T12:04:47+00:00 (fixtures) | `4e981e7` | `87e0fbd0ec898895` | **98.4%** | 5 read |
| `choir-bench` | _never_ | — | — | — | — |

The two are **not averaged**. `fixturecheck` scores notes across the printed systems of the repertoire; `choir-bench` scores staves and bars across the public-domain benchmark pages. They answer different questions and a single figure over both would mean nothing.

## The gate

**pass** — all 5 committed cases stand at or above the reading they were accepted at, under homr `4e981e7`, latest as of 2026-09-06T12:04:47+00:00.

**Nothing gets worse, per case.** Each case remembers the notes-right score it was last accepted at; a run fails if any case reads below its own memory, and a case that improves has its memory raised. Not a total — a win on one page must not pay for a loss on another. Three tiers under one rule: the pinned failures, the five committed fixtures, and every song system on the host that owns the songs.

**No number here says a parse is right.** That word belongs to the operator, reading the case pages against the printed music; what is published is how far each case has moved and what is still wrong with it.

## Over time

| when | harness | tier | homr | references | right |
| --- | --- | --- | --- | --- | --- |
| 2026-09-05T14:10:54+00:00 | `fixturecheck` | one | `8faa87b` | `ff33f48a1bcaac77` | 97.5% |
| 2026-09-05T14:33:10+00:00 | `fixturecheck` | one | `94b20bd` | `ff33f48a1bcaac77` | 97.5% |
| 2026-09-05T17:09:23+00:00 | `fixturecheck` | all | `94b20bd` | `ff33f48a1bcaac77+new93` | 88.5% |
| 2026-09-06T07:14:58+00:00 | `fixturecheck` | one | `86d0f2a` | `ff33f48a1bcaac77` | 97.6% |
| 2026-09-06T07:51:47+00:00 | `fixturecheck` | all | `94b20bd` | `87e0fbd0ec898895` | 92.0% |
| 2026-09-06T09:13:53+00:00 | `fixturecheck` | all | `86d0f2a` | `87e0fbd0ec898895` | 92.5% |
| 2026-09-06T09:41:24+00:00 | `fixturecheck` | one | `86d0f2a` | `ff33f48a1bcaac77` | 85.3% |
| 2026-09-06T09:49:00+00:00 | `fixturecheck` | one | `86d0f2a+dirty` | `ff33f48a1bcaac77` | 98.4% |
| 2026-09-06T10:02:22+00:00 | `fixturecheck` | one | `86d0f2a+dirty` | `ff33f48a1bcaac77` | 98.4% |
| 2026-09-06T10:55:01+00:00 | `fixturecheck` | one | `ffa2327+dirty` | `ff33f48a1bcaac77` | 98.4% |
| 2026-09-06T12:04:14+00:00 | `fixturecheck` | fixtures | `4e981e7` | `87e0fbd0ec898895` | 98.4% |
| 2026-09-06T12:04:47+00:00 | `fixturecheck` | fixtures | `4e981e7` | `87e0fbd0ec898895` | 98.4% |

A tier is not a sample of the one above it — a ten-case run and a full sweep are different populations, so read a percentage against runs of the same tier.

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
