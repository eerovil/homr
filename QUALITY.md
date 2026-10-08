# How good is the scanning right now?

Regenerated from `fixturecheck/series.jsonl` on every run — do not edit by hand. Each number names the homr it measured and the state of the references it measured against, because those move independently and a score that improved because a reference was corrected is not homr improving.

## Now

| harness | measured | homr | references | of everything judged, right | cases |
| --- | --- | --- | --- | --- | --- |
| `fixturecheck` | 2026-10-08T16:02:51+00:00 (fixtures) | `2c0d3b9~pod` | `f26eeff6c43ce42b` | **99.2%** | 7 read |
| `choir-bench` | _never_ | — | — | — | — |

The two are **not averaged**. `fixturecheck` scores notes across the printed systems of the repertoire; `choir-bench` scores staves and bars across the public-domain benchmark pages. They answer different questions and a single figure over both would mean nothing.

## The gate

**pass** — all 7 committed cases stand at or above the reading they were accepted at, under homr `2c0d3b9~pod`, latest as of 2026-10-08T16:02:51+00:00.

**Nothing gets worse, per case.** Each case remembers the notes-right score it was last accepted at; a run fails if any case reads below its own memory, and a case that improves has its memory raised. Not a total — a win on one page must not pay for a loss on another. Three tiers under one rule: the pinned failures, the five committed fixtures, and every song system on the host that owns the songs.

**No number here says a parse is right.** That word belongs to the operator, reading the case pages against the printed music; what is published is how far each case has moved and what is still wrong with it.

## Over time

| when | harness | tier | homr | references | right |
| --- | --- | --- | --- | --- | --- |
| 2026-10-07T18:48:53+00:00 | `fixturecheck` | private | `3c4310c~pod` | `f26eeff6c43ce42b+new71` | 99.9% |
| 2026-10-07T18:52:10+00:00 | `fixturecheck` | fixtures | `3c4310c~pod` | `f26eeff6c43ce42b` | 99.2% |
| 2026-10-07T18:36:41+00:00 | `fixturecheck` | private | `365b061~pod` | `f26eeff6c43ce42b+new71` | 99.9% |
| 2026-10-07T18:39:06+00:00 | `fixturecheck` | fixtures | `365b061~pod` | `f26eeff6c43ce42b` | 99.2% |
| 2026-10-07T19:21:10+00:00 | `fixturecheck` | private | `365b061~pod` | `f26eeff6c43ce42b+new71` | 99.9% |
| 2026-10-07T19:21:40+00:00 | `fixturecheck` | fixtures | `365b061~pod` | `f26eeff6c43ce42b` | 99.2% |
| 2026-10-07T20:03:03+00:00 | `fixturecheck` | private | `48884c4~pod` | `f26eeff6c43ce42b+new71` | 99.9% |
| 2026-10-07T20:04:15+00:00 | `fixturecheck` | fixtures | `48884c4~pod` | `f26eeff6c43ce42b` | 99.2% |
| 2026-10-08T14:57:11+00:00 | `fixturecheck` | private | `55bcdeb~pod` | `f26eeff6c43ce42b+new71` | 99.9% |
| 2026-10-08T14:57:43+00:00 | `fixturecheck` | fixtures | `55bcdeb~pod` | `f26eeff6c43ce42b` | 99.2% |
| 2026-10-08T16:01:32+00:00 | `fixturecheck` | private | `2c0d3b9~pod` | `f26eeff6c43ce42b+new71` | 99.9% |
| 2026-10-08T16:02:51+00:00 | `fixturecheck` | fixtures | `2c0d3b9~pod` | `f26eeff6c43ce42b` | 99.2% |

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
