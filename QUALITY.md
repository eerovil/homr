# How good is the scanning right now?

Regenerated from `fixturecheck/series.jsonl` on every run — do not edit by hand. Each number names the homr it measured and the state of the references it measured against, because those move independently and a score that improved because a reference was corrected is not homr improving.

## Now

| harness | measured | homr | references | of everything judged, right | cases |
| --- | --- | --- | --- | --- | --- |
| `fixturecheck` | 2026-09-07T04:20:54+00:00 (corpus) | `d1f2c85~pod` | `87e0fbd0ec898895+drift93` | **96.3%** | 98 read |
| `choir-bench` | _never_ | — | — | — | — |

The two are **not averaged**. `fixturecheck` scores notes across the printed systems of the repertoire; `choir-bench` scores staves and bars across the public-domain benchmark pages. They answer different questions and a single figure over both would mean nothing.

## The gate

**pass** — all 5 committed cases stand at or above the reading they were accepted at, under homr `d1f2c85~pod`, latest as of 2026-09-07T04:20:54+00:00.

**Nothing gets worse, per case.** Each case remembers the notes-right score it was last accepted at; a run fails if any case reads below its own memory, and a case that improves has its memory raised. Not a total — a win on one page must not pay for a loss on another. Three tiers under one rule: the pinned failures, the five committed fixtures, and every song system on the host that owns the songs.

**No number here says a parse is right.** That word belongs to the operator, reading the case pages against the printed music; what is published is how far each case has moved and what is still wrong with it.

## Over time

| when | harness | tier | homr | references | right |
| --- | --- | --- | --- | --- | --- |
| 2026-09-06T12:37:57+00:00 | `fixturecheck` | fixtures | `a820bb4` | `87e0fbd0ec898895` | 98.4% |
| 2026-09-06T12:38:33+00:00 | `fixturecheck` | fixtures | `a820bb4` | `87e0fbd0ec898895` | 98.4% |
| 2026-09-06T13:43:54+00:00 | `fixturecheck` | fixtures | `a820bb4` | `87e0fbd0ec898895` | 98.4% |
| 2026-09-06T15:26:42+00:00 | `fixturecheck` | fixtures | `a820bb4` | `87e0fbd0ec898895` | 98.4% |
| 2026-09-06T18:34:23+00:00 | `fixturecheck` | fixtures | `8e4610c` | `87e0fbd0ec898895` | 98.8% |
| 2026-09-06T19:04:07+00:00 | `fixturecheck` | fixtures | `8e4610c+dirty` | `87e0fbd0ec898895` | 98.4% |
| 2026-09-06T19:13:45+00:00 | `fixturecheck` | fixtures | `8e4610c+dirty` | `87e0fbd0ec898895` | 98.8% |
| 2026-09-06T20:24:48+00:00 | `fixturecheck` | corpus | `1d47f81` | `87e0fbd0ec898895` | 96.0% |
| 2026-09-07T03:07:53+00:00 | `fixturecheck` | one | `0997326` | `87e0fbd0ec898895` | 94.6% |
| 2026-09-07T03:17:03+00:00 | `fixturecheck` | one | `0997326+dirty` | `87e0fbd0ec898895` | 97.3% |
| 2026-09-07T03:47:43+00:00 | `fixturecheck` | one | `d1f2c85~pod` | `87e0fbd0ec898895` | 100.0% |
| 2026-09-07T04:20:54+00:00 | `fixturecheck` | corpus | `d1f2c85~pod` | `87e0fbd0ec898895+drift93` | 96.3% |

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
