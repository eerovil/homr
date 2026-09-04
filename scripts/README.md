# Prototyping homr against the choir project

Two scripts, one loop. The question they exist to answer is **accuracy on choral
scores**: does a change here read more of the staves, bars and notes a printed
page actually has?

The measuring is not done in this repo. It is done by
[musescore-choir-plugins](https://github.com/eerovil/musescore-choir-plugins),
which owns the fixtures (three public-domain scanned pages with printed staff and
bar counts, one of them with note-level ground truth transcribed by hand) and the
per-system reading path the app really uses. These scripts drive that code with
whichever homr you point them at.

## The loop

```bash
# 1. a worktree that can run: source, its own venv, weights shared
scripts/choir-worktree.sh slurs

# 2. what the host runs today, for something to compare against
scripts/choir-bench.py --benchmark -o /tmp/baseline.json

# 3. edit /var/home/eero/homr-trees/slurs, then measure it
scripts/choir-bench.py --tree ~/homr-trees/slurs --benchmark \
    -o /tmp/slurs.json --compare /tmp/baseline.json
```

The edit needs no reinstall — the venv is an editable install of the tree.

## choir-worktree.sh

`scripts/choir-worktree.sh <name> [start-point]` makes
`../homr-trees/<name>` on branch `prototype/<name>`, off `origin/main` by
default (which is what `install-homr.sh` in the choir repo installs, so it is the
baseline worth measuring against).

Each tree gets **its own venv**, because the app reaches homr as a subprocess
through `HOMR_BIN` — a tree is testable exactly when it has its own executable.
One shared venv would have to be reinstalled on every switch, and it is the venv
the live app uses.

The **model weights are symlinked**, not downloaded. homr keeps ~150 MB of
`.onnx` inside its own package directory, and an editable install makes that
directory the worktree's `homr/` folder; every tree points at the one copy this
host already has. They are gitignored, so the tree stays clean. A new tree costs
about five seconds and no download.

`HOMR_TREES`, `HOMR_PYTHON`, `HOMR_WEIGHTS_FROM` move the pieces.
`HOMR_TEST_DEPS=1` also installs torch, which six of the test modules import.

Throw a tree away with `git worktree remove ../homr-trees/<name> --force`.

## choir-bench.py

Runs under the **choir app's** interpreter (it re-execs itself there), because
the reading is that project's code — poppler crops, per-system flattening, the
slur repair. Only the binary under test comes from the worktree. Set
`CHOIR_REPO` if the checkout is not at `~/musescore-choir-plugins`.

| flag | what it runs |
| --- | --- |
| `--benchmark` | the three benchmark pages, one printed system at a time, bands padded the way the scan stage pads them. Staves and bars per system against what the page prints; notes per bar against the truth table where there is one. ~80s a page. |
| `--scan` | the whole Virta venhettä vie fixture, 15 systems, including `assemble`. What the app does, seams included. |
| `--pytest` | the fork's unit tests in the worktree. Catches breakage; says nothing about accuracy. |
| `--all` | all three. |

Useful extras: `--pages B1a,B1b` narrows the benchmark, `--dpi` overrides the
200 dpi the app scans at (a measured default — at 300 dpi B4 came back short of
staves), `--keep DIR` keeps the crops and the MusicXML to look at, and
`--compare` diffs against a previous `-o` file.

## --kubernetes

`--kubernetes` runs the homr half in a pod instead of on this host.

**Not for speed, on an idle host.** Measured page for page with nothing else
running, the pod reads one in 15–17 s and this host in 16–19 s — a wash. What it
buys is whose cores are spent: a page through the shim costs **18.6 s of wall
clock and 0.29 s of this host's CPU**.

**On a busy host it is also much faster, which is the real reason.** This is a
four-core machine that serves the live app, runs its own deploy and hosts other
agents' test suites; the first `--kubernetes` run happened to land at a load
average of 22.8, and B1a took **31 s in the pod against 181 s here** — same
three systems, same staff and bar counts, the same 4/14 bars note-exact. Nearly
6x, none of it from the cluster being quick. A sweep nobody is waiting for is
exactly the work that should be somewhere else.

Everything except the reading still runs here: the crops, the flattening, the
scoring, and `--pytest`. Those are the choir app's code and the fork's own tests,
and they are not where the sweep spends itself.

```bash
scripts/choir-k8s.sh up                                   # once, ~2 min
scripts/choir-bench.py --kubernetes --benchmark -o pod.json
scripts/choir-bench.py --kubernetes --tree ~/homr-trees/slurs --benchmark \
    -o slurs.json --compare pod.json
```

`--tree` still means what it means: the tree's `homr/` package is copied into the
pod on every run and put in front of the pod's venv on `PYTHONPATH` — the same
arrangement `omr.py`'s engines use locally, so editing a tree and re-running
costs one copy and no reinstall. Without `--tree`, the pod runs its own install.

The pod is stock `python:3.12` and builds its venv onto a persistent volume the
first time; there is no image to build, which is deliberate (an arm64 image from
an x86 host needs emulation or a second machine, and this is a benchmarking tool,
not something the app depends on). The cost is that the volume is not
reproducible from this repository — `scripts/choir-k8s.sh purge` throws it away
and `up` builds it again from `HOMR_SOURCE`.

**One trap when comparing pod numbers with host numbers.** The pod's venv is
installed from `HOMR_SOURCE` at the moment `up` ran, and the host's install came
from whenever `install-homr.sh` was last run — they can be different commits, so
a pod-vs-host diff can be measuring that rather than your change. Compare
pod against pod, or pass `--tree` to both so the code under test is the same
source in either place. `scripts/choir-k8s.sh status` says which commit the pod
has.

`KUBECTL`, `CHOIR_K8S_NAMESPACE`, `CHOIR_K8S_POD`, `CHOIR_K8S_IMAGE` and
`CHOIR_K8S_VOLUME_SIZE` move the pieces. `down` deletes the pod and keeps the
volume; `purge` deletes both.

Needs poppler for the crops. Nothing here is fast: homr is ~10–20s a system.
