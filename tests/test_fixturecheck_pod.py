"""The pod preference's own rules: local is the fallback, never the surprise.

The sweep reads pages in the Mac cluster's pod by default (`fixturecheck.pod`),
because a corpus run is minutes of every core on a host that also runs the live
choir app. What these pin is the shape of that preference: `off` really asks
the cluster nothing, an unanswered cluster falls back to this host out loud,
`require` refuses to fall back at all, a pod parse is cached under its own
`~pod` name so two architectures' readings cannot share a cache key, a pod
that dies mid-run costs one local re-read rather than a run of fake
"unreadable" cases, and a page homr genuinely could not read is still a
failure on either side of the wire.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import fixturecheck.__main__ as check
from fixturecheck import cases, pod


@pytest.fixture(autouse=True)
def fresh_pod(monkeypatch):
    """Every test resolves the pod from scratch, and none touches a cluster."""
    monkeypatch.setattr(pod, "_shim", None)
    monkeypatch.setattr(pod, "_resolved", False)
    monkeypatch.setattr(pod, "_lost", False)
    monkeypatch.setattr(pod, "reachable", lambda: pytest.fail(
        "the test did not say whether the cluster answers"))


@pytest.fixture
def case(tmp_path, monkeypatch):
    """A case whose picture exists, parsed into a private cache."""
    monkeypatch.setattr(check, "PARSES", tmp_path / "parses")
    image = tmp_path / "little.png"
    image.write_bytes(b"not really a png")
    return cases.Case(name="little", image=image,
                      reference=tmp_path / "little.ref.musicxml", origin="song")


def reader(tmp_path, script: str) -> str:
    """An executable standing in for the pod shim (the image comes last)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "shim"
    path.write_text("#!/usr/bin/env bash\n" + script)
    path.chmod(0o755)
    return str(path)


WRITES = 'for a in "$@"; do img="$a"; done\necho "<x/>" > "${img%.*}.musicxml"\n'
FAILS = "exit 1\n"


def test_off_asks_the_cluster_nothing(monkeypatch):
    monkeypatch.setenv("CHOIR_K8S", "off")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail(
        "CHOIR_K8S=off ran a subprocess on the way to saying no"))
    assert pod.shim() is None
    assert pod.tag() == ""


def test_an_unanswered_cluster_falls_back_out_loud(monkeypatch, capsys):
    monkeypatch.delenv("CHOIR_K8S", raising=False)
    monkeypatch.setattr(pod, "reachable", lambda: False)
    assert pod.shim() is None
    said = capsys.readouterr().out
    assert "reading pages on this host" in said
    # Resolved once: asking again is free and quiet.
    assert pod.shim() is None
    assert capsys.readouterr().out == ""


def test_require_refuses_to_read_here(monkeypatch):
    monkeypatch.setenv("CHOIR_K8S", "require")
    monkeypatch.setattr(pod, "reachable", lambda: False)
    with pytest.raises(SystemExit):
        pod.shim()


def test_a_pod_parse_is_cached_under_its_own_name(tmp_path, case, monkeypatch):
    shim = reader(tmp_path, WRITES)
    monkeypatch.setattr(pod, "shim", lambda: shim)
    parsed = check.parse(case, "abc123")
    assert parsed is not None
    assert parsed.name == f"little@abc123{pod.TAG}.musicxml"
    # And the local read of the same code is a different file: the pod is a
    # different architecture, and a parse says where it was read.
    local_reader = reader(tmp_path / "local", WRITES)
    monkeypatch.setattr(pod, "shim", lambda: None)
    monkeypatch.setattr(check, "_local_read",
                        lambda image: [local_reader, str(image)])
    local = check.parse(case, "abc123")
    assert local is not None
    assert local.name == "little@abc123.musicxml"
    assert local != parsed


def test_losing_the_pod_midrun_reads_the_case_here(tmp_path, case, monkeypatch,
                                                   capsys):
    # The pod resolved fine at the start of the run and dies under this case:
    # the failing shim, the real `shim()` bookkeeping, and the real `lose()`.
    monkeypatch.setattr(pod, "_resolved", True)
    monkeypatch.setattr(pod, "_shim", reader(tmp_path, FAILS))
    monkeypatch.setattr(pod, "alive", lambda: False)
    monkeypatch.delenv("CHOIR_K8S", raising=False)
    local_reader = reader(tmp_path / "local", WRITES)
    monkeypatch.setattr(check, "_local_read",
                        lambda image: [local_reader, str(image)])
    parsed = check.parse(case, "abc123")
    assert parsed is not None
    assert parsed.name == "little@abc123.musicxml"  # the local name, no ~pod
    assert pod.shim() is None                        # and the pod stays out
    assert "stopped answering" in capsys.readouterr().out


def test_a_page_homr_cannot_read_is_still_none(tmp_path, case, monkeypatch):
    monkeypatch.setattr(pod, "shim", lambda: reader(tmp_path, FAILS))
    monkeypatch.setattr(pod, "alive", lambda: True)  # the pod is fine; the page is not
    assert check.parse(case, "abc123") is None


def test_losing_the_pod_under_require_stops_the_run(monkeypatch):
    monkeypatch.setenv("CHOIR_K8S", "require")
    with pytest.raises(SystemExit):
        pod.lose()
