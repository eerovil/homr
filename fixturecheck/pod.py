"""Read pages in the cluster's pod when it answers, on this host when it does not.

Not for speed -- ``scripts/choir-k8s.sh``'s own header measured the pod as a
wash per page.  What it buys is that a corpus sweep stops being minutes of
every core on a four-core host that also runs the live choir app, its deploy
and other agents' test suites.  So the pod is the *default* and this host is
the fallback, taken out loud rather than silently:

    CHOIR_K8S=prefer   the default -- pod when the cluster answers, else here
    CHOIR_K8S=off      stay on this host, ask the cluster nothing
    CHOIR_K8S=require  refuse to read locally at all

The plumbing is ``choir-k8s.sh`` unchanged -- ``up`` (pod + venv + weights),
``ship`` (this tree's ``homr/`` in front of the pod's venv, so the code under
test is the code that reads), ``shim`` (an executable honouring homr's own
CLI).  The same three steps ``choir-bench.py --kubernetes`` has always made.

**A pod parse is cached under its own name** (the ``~pod`` tag on the
fingerprint).  The pod is arm64 and this host is not, and onnx inference is
not promised to be bit-identical across them -- so a parse says where it was
read, or two hosts' readings would blur into one cache key and a comparison
could mix them with nothing saying so.  The first pod sweep therefore re-reads
everything once; that is the point of having a pod to read it on.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from fixturecheck import cases

SCRIPT = cases.ROOT / "scripts" / "choir-k8s.sh"

#: The tag a pod-read parse carries in its cache filename and in the run record.
TAG = "~pod"

_shim: str | None = None
_resolved = False
_lost = False


def mode() -> str:
    return os.environ.get("CHOIR_K8S", "prefer")


def _pod_name() -> str:
    return os.environ.get("CHOIR_K8S_POD", "homr-bench")


def _kubectl() -> list[str]:
    return [os.environ.get("KUBECTL", "kubectl"),
            "-n", os.environ.get("CHOIR_K8S_NAMESPACE", "default")]


def _say(message: str) -> None:
    print(f"[pod] {message}")


def _fall_back(reason: str) -> None:
    """Local is the answer -- loudly, or not at all under ``require``."""
    if mode() == "require":
        sys.exit(f"[pod] {reason} — and CHOIR_K8S=require refuses to read "
                 f"pages on this host")
    _say(f"{reason} — reading pages on this host instead "
         f"(CHOIR_K8S=require to refuse)")
    return None


def reachable() -> bool:
    """One cheap question with a short deadline, so a sleeping Mac costs
    seconds rather than the 180s ``choir-k8s.sh up`` is prepared to wait."""
    try:
        probe = subprocess.run(
            _kubectl() + ["--request-timeout=5s", "get", "--raw", "/readyz"],
            capture_output=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def alive() -> bool:
    """Is the pod still answering?  Asked after a pod read failed, to tell a
    page homr could not read from a cluster that has gone away under us."""
    try:
        probe = subprocess.run(
            _kubectl() + ["exec", _pod_name(), "--", "true"],
            capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def lose() -> None:
    """The cluster went away mid-run.  The rest of the run reads here -- those
    parses are cached under their local names, and only this run's record still
    carries the ``~pod`` tag it started with."""
    global _lost
    if mode() == "require":
        sys.exit("[pod] the pod stopped answering mid-run — and "
                 "CHOIR_K8S=require refuses to read pages on this host")
    _lost = True
    _say("the pod stopped answering mid-run — the remaining cases are read on "
         "this host, under their local cache names")


def shim() -> str | None:
    """The pod as an executable homr's callers accept, or None for local.

    Resolved once per process: the pod is built if this is the first run,
    this tree's ``homr/`` is shipped so the code under test is the code that
    reads, and the shim is written.  Every failure on the way is a fallback,
    not an error -- nobody's sweep should die because a Mac went to sleep.
    """
    global _shim, _resolved
    if _resolved:
        return None if _lost else _shim
    _resolved = True
    if mode() == "off":
        return None
    if not SCRIPT.exists():
        return _fall_back(f"{SCRIPT.name} is not in this tree")
    if not reachable():
        return _fall_back("the cluster did not answer")
    for step, deadline in ((["up"], 600),
                           (["ship", str(cases.ROOT)], 120)):
        try:
            run = subprocess.run([str(SCRIPT)] + step, capture_output=True,
                                 text=True, timeout=deadline)
        except subprocess.TimeoutExpired:
            return _fall_back(f"choir-k8s.sh {step[0]} took over {deadline}s")
        if run.returncode:
            tail = (run.stderr or run.stdout or "").strip().splitlines()
            return _fall_back(f"choir-k8s.sh {step[0]} failed"
                              + (f": {tail[-1]}" if tail else ""))
    path = Path(tempfile.mkdtemp(prefix="fixturecheck-pod-")) / "homr"
    if subprocess.run([str(SCRIPT), "shim", str(path)],
                      capture_output=True).returncode:
        return _fall_back("could not write the pod shim")
    _shim = str(path)
    _say(f"reading pages in pod {_pod_name()} — CHOIR_K8S=off stays on this host")
    return _shim


def tag() -> str:
    """What the current reader stamps on a parse: ``~pod`` or nothing."""
    return TAG if shim() else ""
