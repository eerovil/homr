#!/usr/bin/env bash
# Make a homr worktree you can actually run: source tree, its own venv, and the
# model weights shared with the venv that is already installed on this host.
#
#   scripts/choir-worktree.sh <name> [start-point]
#   scripts/choir-worktree.sh slurs            # branch prototype/slurs off origin/main
#   scripts/choir-worktree.sh slurs origin/choir-0.7.0
#
# The default start-point is origin/main because that is what the app installs:
# scripts/install-homr.sh in the choir repo pulls the fork's main. Prototyping
# off anything else means the baseline you measure against is not the homr the
# host is running.
#
# Why a venv per tree rather than one shared one: the choir app calls homr as a
# subprocess through HOMR_BIN, so a tree is testable exactly when it has its own
# executable. Sharing one venv would mean reinstalling it every time you switch
# trees, and that venv is the one the live app uses.
#
# Why the weights are symlinked: homr keeps ~150 MB of .onnx inside its own
# package directory, and an editable install makes that directory the worktree's
# own homr/ folder. Downloading them per tree would cost 150 MB and several
# minutes each time, so every tree points at the one copy this host already has.
# They are gitignored (*.onnx), so the worktree stays clean.
#
# Environment:
#   HOMR_TREES          where worktrees go (default: <repo>/../homr-trees)
#   HOMR_PYTHON         interpreter version (default: 3.12, what the benchmarks used)
#   HOMR_WEIGHTS_FROM   package dir to borrow weights from
#                       (default: the homr-venv scripts/install-homr.sh builds)
#   HOMR_TEST_DEPS=1    also install torch, so the fork's unit tests can run.
#                       Off by default: six of the test modules import torch and
#                       it is a gigabyte-scale download, which is a poor trade
#                       for a loop whose question is accuracy on real pages.

set -euo pipefail

name="${1:-}"
start="${2:-origin/main}"
if [ -z "$name" ]; then
    echo "usage: scripts/choir-worktree.sh <name> [start-point]" >&2
    exit 2
fi

repo="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
trees="${HOMR_TREES:-$(dirname "$repo")/homr-trees}"
tree="$trees/$name"
branch="prototype/$name"
python_version="${HOMR_PYTHON:-3.12}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. See https://docs.astral.sh/uv/getting-started/" >&2
    exit 1
fi

# --- the worktree --------------------------------------------------------

if [ -d "$tree" ]; then
    echo "Worktree already at $tree — reusing it"
else
    mkdir -p "$trees"
    # A stale origin/main is the one way this quietly does the wrong thing: the
    # tree would branch off an old baseline and every number below it would be
    # measured against homr the host is no longer running.
    git -C "$repo" fetch --quiet origin || echo "  (could not fetch; using what is here)"
    if git -C "$repo" show-ref --verify --quiet "refs/heads/$branch"; then
        echo "Checking out existing branch $branch"
        git -C "$repo" worktree add "$tree" "$branch"
    else
        echo "Creating branch $branch from $start"
        git -C "$repo" worktree add "$tree" -b "$branch" "$start"
    fi
fi

# --- its environment -----------------------------------------------------

echo "Creating $tree/.venv (python $python_version)"
uv venv --allow-existing --python "$python_version" "$tree/.venv"

# Editable, so an edit in the tree is live without reinstalling. pytest comes
# too: the fork's dev group pulls torch and the whole training stack, which is
# gigabytes this loop never touches, so the unit tests are run with pytest alone
# and anything needing the training deps reports as a collection error rather
# than being silently skipped.
# The [cpu] extra is where main keeps onnxruntime — without it homr installs and
# then dies on "No module named 'onnxruntime'". It does not exist on 0.7.0 and
# older, where uv warns and ignores it and the runtime comes in as a plain
# dependency, so asking for it is right on both. CPU deliberately: this host's
# GTX 970 is below onnxruntime's compute floor (choir issue #93).
echo "Installing the worktree (editable, [cpu]) + pytest"
VIRTUAL_ENV="$tree/.venv" uv pip install --quiet -e "$tree[cpu]" pytest

if [ "${HOMR_TEST_DEPS:-}" = "1" ]; then
    # What the test modules import, and not the whole dev group: that group does
    # not install here at all — onnx-simplifier builds from source and wants
    # cmake — and the rest of it is the training stack, which this loop never
    # touches. torch is the big one; the others are small.
    echo "Installing test dependencies (HOMR_TEST_DEPS=1) — torch is a big download"
    VIRTUAL_ENV="$tree/.venv" uv pip install --quiet \
        torch editdistance music21 musicdiff datasets
fi

# --- the weights ---------------------------------------------------------

default_weights="$HOME/.local/share/musescore-choir-plugins/homr-venv/lib/python3.12/site-packages/homr"
weights_from="${HOMR_WEIGHTS_FROM:-$default_weights}"
# Where weights this host has downloaded but the installed venv does not have
# are kept, so the second tree on a given model costs nothing. The fork's main
# is on a newer model than the app's venv, so this is not hypothetical: without
# it every tree off main downloads the same 81 MB again.
cache="${HOMR_WEIGHTS_CACHE:-$HOME/.cache/homr-weights}"

# Link anything $1 has that the tree has not, except the code itself. That is
# the weights today; it stays right if a future model ships a companion file
# that is not .onnx.
link_weights() {
    local from="$1" linked=0 src rel dst
    [ -d "$from" ] || return 0
    while IFS= read -r -d '' src; do
        rel="${src#"$from"/}"
        case "$rel" in
            *__pycache__*|*.py) continue ;;
        esac
        dst="$tree/homr/$rel"
        [ -e "$dst" ] && continue
        mkdir -p "$(dirname "$dst")"
        ln -s "$src" "$dst"
        linked=$((linked + 1))
    done < <(find "$from" -type f -print0)
    echo "  $linked file(s) from $from"
}

echo "Sharing model weights"
link_weights "$weights_from"
link_weights "$cache"

# Idempotent, and only fetches what is genuinely missing: with the symlinks in
# place a matching model costs nothing, and a model the fork moved to is pulled
# once and then cached below.
echo "Checking the models are complete"
"$tree/.venv/bin/homr" --init

# Whatever --init just downloaded is a real file in the tree rather than a
# symlink. Move it to the cache and link it back, so it survives this tree being
# thrown away and the next tree on the same model does not fetch it again.
cached=0
while IFS= read -r -d '' src; do
    rel="${src#"$tree/homr"/}"
    case "$rel" in
        *__pycache__*|*.py) continue ;;
    esac
    # Only files the checkout does not track — never touch source control. That
    # is the guard, not the extension: a tokenizer .json is tracked, a model is not.
    git -C "$tree" ls-files --error-unmatch "homr/$rel" >/dev/null 2>&1 && continue
    mkdir -p "$(dirname "$cache/$rel")"
    mv "$src" "$cache/$rel"
    ln -s "$cache/$rel" "$src"
    cached=$((cached + 1))
done < <(find "$tree/homr" -type f -print0)
[ "$cached" -gt 0 ] && echo "Cached $cached downloaded file(s) in $cache"

cat <<EOF

Ready.

  tree    $tree
  branch  $branch
  homr    $tree/.venv/bin/homr

Test it against the choir fixtures:

  scripts/choir-bench.py --tree "$tree" --all
EOF
