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

Needs poppler for the crops. Nothing here is fast: homr is ~10–20s a system.
