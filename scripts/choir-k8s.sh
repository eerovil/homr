#!/usr/bin/env bash
# Run the homr half of choir-bench.py inside a Kubernetes pod.
#
# WHY THIS EXISTS, since it is not for speed. Measured on one benchmark page:
# a pod on the cluster reads it in 15-17s and this host reads it in 16-19s, so
# the cluster is a wash. What it buys is that a benchmark sweep is minutes of
# every core on a four-core host that also runs the live app, its deploy and
# other agents' test suites — and nobody is waiting for the sweep. Moving it off
# is worth more than the 10% it costs.
#
# WHAT RUNS IN THE POD is the worktree under test, not a homr baked into an
# image. That is the whole point of the harness: the question is always whether
# *this branch* reads the repertoire better. So the pod's venv provides only the
# dependencies and the model weights — the same arrangement omr.py's "engines"
# use locally — and `ship` copies the tree's own `homr/` package in front of it
# on PYTHONPATH. Editing the tree and re-running costs one `ship`, no reinstall.
#
# There is deliberately NO IMAGE TO BUILD. The pod is stock `python:3.12` and
# installs homr's dependencies once onto a persistent volume, because building
# an arm64 image from an x86 host needs either emulation or a second machine in
# the loop, and this is a benchmarking tool rather than something the app
# depends on. The cost is honest and worth naming: the volume's contents are not
# reproducible from this repository. `purge` throws it away and `up` builds it
# again from HOMR_SOURCE.
#
#   scripts/choir-k8s.sh up                  # pod + venv + weights (once, ~2 min)
#   scripts/choir-k8s.sh ship ~/homr-trees/slurs
#   scripts/choir-k8s.sh shim /tmp/homr-k8s  # an executable homr's callers accept
#   scripts/choir-k8s.sh status
#   scripts/choir-k8s.sh down                # delete the pod, keep the volume
#   scripts/choir-k8s.sh purge               # delete the volume too
#
# choir-bench.py --kubernetes does up/ship/shim for you.
set -euo pipefail

KUBECTL="${KUBECTL:-kubectl}"
NAMESPACE="${CHOIR_K8S_NAMESPACE:-default}"
POD="${CHOIR_K8S_POD:-homr-bench}"
IMAGE="${CHOIR_K8S_IMAGE:-python:3.12}"
VOLUME_SIZE="${CHOIR_K8S_VOLUME_SIZE:-8Gi}"
# The same source install-homr.sh uses, so the pod's dependencies match the
# host's rather than being a second thing to keep in step.
HOMR_SOURCE="${HOMR_SOURCE:-homr[cpu] @ git+https://github.com/eerovil/homr.git@main}"

VENV=/work/venv
SRC=/work/src

kube() { "$KUBECTL" -n "$NAMESPACE" "$@"; }
say() { printf '[choir-k8s] %s\n' "$*" >&2; }

pod_running() {
    [ "$(kube get pod "$POD" -o jsonpath='{.status.phase}' 2>/dev/null || true)" = Running ]
}

apply_pod() {
    cat <<EOF | kube apply -f - >/dev/null
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${POD}-work
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: ${VOLUME_SIZE}
---
apiVersion: v1
kind: Pod
metadata:
  name: ${POD}
  labels: {app: ${POD}}
spec:
  restartPolicy: Always
  containers:
  - name: homr
    image: ${IMAGE}
    command: ["sleep", "infinity"]
    volumeMounts:
    - {name: work, mountPath: /work}
    resources:
      requests: {cpu: "1", memory: "2Gi"}
      limits: {memory: "16Gi"}
  volumes:
  - name: work
    persistentVolumeClaim:
      claimName: ${POD}-work
EOF
}

cmd_up() {
    if ! pod_running; then
        say "creating pod $POD in $NAMESPACE"
        apply_pod
        kube wait --for=condition=Ready "pod/$POD" --timeout=180s >/dev/null
    fi

    if kube exec "$POD" -- test -x "$VENV/bin/python" 2>/dev/null; then
        say "venv already built"
    else
        say "building the venv — once, a few minutes"
        # libgl/libglib are opencv's, and are not in the slim base.
        kube exec "$POD" -- bash -lc "
            set -e
            apt-get update -qq
            apt-get install -y -qq git libgl1 libglib2.0-0 >/dev/null
            python -m venv $VENV
            $VENV/bin/pip install --quiet --no-cache-dir '$HOMR_SOURCE'
            $VENV/bin/homr --init
        " >&2
    fi
    # The weights live beside homr's own source, so a shipped tree needs them
    # where *it* is. Symlinked, never copied: the names carry a content hash, so
    # a branch wanting different weights asks for a different name.
    # homr resolves a weight file relative to its own module, so the shipped
    # tree needs them at the same relative paths — and they are in
    # segmentation/ and transformer/, not at the top.
    kube exec "$POD" -- bash -lc "
        pkg=\$(ls -d $VENV/lib/python*/site-packages/homr)
        mkdir -p $SRC/homr
        n=0
        while read -r f; do
            rel=\${f#\$pkg/}
            mkdir -p $SRC/homr/\$(dirname \"\$rel\")
            ln -sf \"\$f\" $SRC/homr/\$rel
            n=\$((n+1))
        done < <(find \$pkg -name '*.onnx')
        echo \"[choir-k8s] linked \$n weight files\" >&2
    " >&2
    say "ready"
}

# Put a worktree's homr package in front of the venv's copy. Source only — the
# weights are already linked in by `up` and are not sent over the wire.
cmd_ship() {
    local tree="${1:?usage: choir-k8s.sh ship <worktree>}"
    [ -d "$tree/homr" ] || { say "no homr package in $tree"; exit 1; }
    say "shipping $(basename "$tree")/homr"
    tar -C "$tree" -cf - --exclude='*.onnx' --exclude='__pycache__' homr \
        | kube exec -i "$POD" -- bash -lc "
            rm -rf $SRC/homr.py $SRC/homr-new
            mkdir -p $SRC/homr-new && tar -C $SRC/homr-new -xf -
            # keep the linked weights, replace everything else
            find $SRC/homr -type f -not -name '*.onnx' -delete 2>/dev/null || true
            cp -r $SRC/homr-new/homr/. $SRC/homr/
            rm -rf $SRC/homr-new
        "
    kube exec "$POD" -- bash -lc "cd $SRC && $VENV/bin/python -c 'import homr, os; print(os.path.dirname(homr.__file__))'" >&2
}

# An executable that behaves like `homr`: takes the image as its last argument,
# writes <base>.musicxml beside it, and writes nothing at all when homr failed —
# omr.read_page treats a zero exit with no file as a failure, so a stub output
# file here would turn a broken parse into a silently empty one.
cmd_shim() {
    local path="${1:?usage: choir-k8s.sh shim <path>}"
    cat > "$path" <<EOF
#!/usr/bin/env bash
# Generated by scripts/choir-k8s.sh — runs homr in pod $POD.
set -euo pipefail
KUBECTL="\${KUBECTL:-$KUBECTL}"
kube() { "\$KUBECTL" -n "$NAMESPACE" "\$@"; }

image=""
for arg in "\$@"; do image="\$arg"; done      # homr takes the image last
[ -n "\$image" ] || { echo "no image argument" >&2; exit 2; }
name="\$(basename "\$image")"
base="\${name%.*}"
remote="/work/run/\$\$"

cleanup() { kube exec "$POD" -- rm -rf "\$remote" >/dev/null 2>&1 || true; }
trap cleanup EXIT

kube exec -i "$POD" -- bash -lc "mkdir -p \$remote && cat > \$remote/\$name" < "\$image"

status=0
kube exec "$POD" -- bash -lc "
    cd \$remote && PYTHONPATH=$SRC $VENV/bin/python -c 'from homr.main import main; main()' --gpu no \$remote/\$name
" || status=\$?

# Only bring the answer back if there is one. Base64 so nothing in the pipe can
# reinterpret the bytes.
if kube exec "$POD" -- test -f "\$remote/\$base.musicxml" 2>/dev/null; then
    kube exec "$POD" -- base64 "\$remote/\$base.musicxml" \\
        | base64 -d > "\$(dirname "\$image")/\$base.musicxml"
fi
exit \$status
EOF
    chmod +x "$path"
    say "shim written to $path"
}

cmd_status() {
    kube get pod "$POD" -o wide 2>&1 || true
    kube exec "$POD" -- bash -lc "
        echo -n 'venv: '; [ -x $VENV/bin/python ] && $VENV/bin/pip show homr 2>/dev/null | head -2 | tr '\n' ' ' || echo missing
        echo; echo -n 'shipped source: '; ls -d $SRC/homr 2>/dev/null || echo none
        echo -n 'weights: '; find $SRC/homr -name '*.onnx' 2>/dev/null | wc -l
    " 2>/dev/null || true
}

cmd_down() { kube delete pod "$POD" --ignore-not-found; }

cmd_purge() {
    kube delete pod "$POD" --ignore-not-found
    kube delete pvc "${POD}-work" --ignore-not-found
}

case "${1:-}" in
    up)     shift; cmd_up "$@" ;;
    ship)   shift; cmd_ship "$@" ;;
    shim)   shift; cmd_shim "$@" ;;
    status) shift; cmd_status "$@" ;;
    down)   shift; cmd_down "$@" ;;
    purge)  shift; cmd_purge "$@" ;;
    *) sed -n '1,40p' "$0" >&2; exit 2 ;;
esac
