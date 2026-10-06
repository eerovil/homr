# How good is the scanning right now?

Regenerated from `fixturecheck/series.jsonl` on every run — do not edit by hand. Each number names the homr it measured and the state of the references it measured against, because those move independently and a score that improved because a reference was corrected is not homr improving.

## Now

| harness | measured | homr | references | of everything judged, right | cases |
| --- | --- | --- | --- | --- | --- |
| `fixturecheck` | 2026-10-05T19:01:08+00:00 (private) | `c7b0bac+2d6093~pod` | `f26eeff6c43ce42b+new4` | **98.8%** | 4 read |
| `choir-bench` | _never_ | — | — | — | — |

The two are **not averaged**. `fixturecheck` scores notes across the printed systems of the repertoire; `choir-bench` scores staves and bars across the public-domain benchmark pages. They answer different questions and a single figure over both would mean nothing.

## The gate

**FAIL** — 0/7 committed cases stand under homr `c7b0bac+2d6093~pod`, latest as of . Not judged under this homr, so the gate cannot pass: `hanget-soi`, `kolme-kakea`, `laulun-aika-s2`, `sammon-ryosto`, `system4`, `talviuni-s1`, `talviuni-s2`. A pass under an earlier homr is not a claim about this one.

Each case counts by its own latest result under that homr, so re-running one cannot speak for the others.

**Nothing gets worse, per case.** Each case remembers the notes-right score it was last accepted at; a run fails if any case reads below its own memory, and a case that improves has its memory raised. Not a total — a win on one page must not pay for a loss on another. Three tiers under one rule: the pinned failures, the five committed fixtures, and every song system on the host that owns the songs.

**No number here says a parse is right.** That word belongs to the operator, reading the case pages against the printed music; what is published is how far each case has moved and what is still wrong with it.

## Over time

| when | harness | tier | homr | references | right |
| --- | --- | --- | --- | --- | --- |
| 2026-10-04T04:05:35+00:00 | `fixturecheck` | one | `c343b30` | `5c1c6d63604b9876` | 97.8% |
| 2026-10-04T04:09:37+00:00 | `fixturecheck` | fixtures | `c343b30` | `5c1c6d63604b9876` | 99.0% |
| 2026-10-04T04:48:53+00:00 | `fixturecheck` | fixtures | `c343b30+9f25c5` | `f26eeff6c43ce42b` | 99.2% |
| 2026-10-04T04:49:13+00:00 | `fixturecheck` | one | `c343b30+9f25c5` | `f26eeff6c43ce42b` | 98.7% |
| 2026-10-04T04:49:15+00:00 | `fixturecheck` | one | `c343b30+9f25c5` | `f26eeff6c43ce42b` | 100.0% |
| 2026-10-04T04:49:23+00:00 | `fixturecheck` | one | `c343b30+9f25c5` | `f26eeff6c43ce42b` | 92.6% |
| 2026-10-04T07:09:12+00:00 | `fixturecheck` | fixtures | `cb5546d+b49cef~pod` | `f26eeff6c43ce42b` | 99.2% |
| 2026-10-04T07:24:37+00:00 | `fixturecheck` | fixtures | `3c47efd+7d0176~pod` | `f26eeff6c43ce42b` | 99.2% |
| 2026-10-04T07:25:08+00:00 | `fixturecheck` | one | `3c47efd+678fa6~pod` | `f26eeff6c43ce42b+drift1` | 100.0% |
| 2026-10-04T07:26:47+00:00 | `fixturecheck` | ten | `3c47efd+678fa6~pod` | `f26eeff6c43ce42b+drift5` | 99.4% |
| 2026-10-05T18:58:29+00:00 | `fixturecheck` | fixtures | `c7b0bac+2d6093~pod` | `f26eeff6c43ce42b` | 99.2% |
| 2026-10-05T19:01:08+00:00 | `fixturecheck` | private | `c7b0bac+2d6093~pod` | `f26eeff6c43ce42b+new4` | 98.8% |

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
