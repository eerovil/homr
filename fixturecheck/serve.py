"""The report, served — and able to start a run, so a phone is enough.

The report answers "how good is it now" without running anything. What it could
not do was *start* the run that refreshes it: that still meant ssh, a checkout,
the right `MUSESCORE_CLI_PATH`, and remembering the incantation. So a static
folder is not quite enough after all, and this is the smallest thing that is —
the same directory, plus two routes.

    POST /run     {"tier": "ten"}            queue a run
    GET  /queue   what is running, what is waiting, what finished last

**One run at a time, and presses queue rather than collide.** A run is minutes
of every core the machine has and homr is the heavy part of it; two at once is
two slow runs rather than either finishing sooner. A worker thread takes one
request off the queue at a time, and the page shows what is running and what is
behind it, because a button that appears to do nothing for four minutes is a
button people press again.

**Nothing is interpolated into a shell.** The runs are `python -m fixturecheck
<tier> [names...]` as an argument list, and a name has to be one the harness
already knows. Not because the tailnet is dangerous — `tailscale serve`
authenticates and this binds to loopback — but because building the argument
list out of request text is the kind of thing that is only ever wrong.

Run it the way the unit does, under the homr venv, or a run cannot import homr:

    <homr venv>/bin/python -m fixturecheck.serve
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fixturecheck import cases, report  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("FIXTURECHECK_PORT", "8125"))

#: The tiers `python -m fixturecheck` understands. `one` also needs names.
TIERS = ("one", "ten", "all")

_wanted: queue.Queue = queue.Queue()
_state_lock = threading.Lock()
_state: dict = {"running": None, "waiting": [], "last": None}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _label(job: dict) -> str:
    names = job.get("names") or []
    return f"{job['tier']}: {', '.join(names)}" if names else job["tier"]


def _worker() -> None:
    """One run at a time, for as long as anything is waiting."""
    while True:
        job = _wanted.get()
        with _state_lock:
            _state["running"] = dict(job, started=_now())
            _state["waiting"] = [w for w in _state["waiting"] if w["id"] != job["id"]]
        argv = [sys.executable, "-m", "fixturecheck", job["tier"], *job["names"]]
        try:
            done = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                                  timeout=4 * 60 * 60)
            # A non-zero exit is the gate failing, which is a result and not a
            # crash: the report is written either way, so it is reported as an
            # outcome rather than an error.
            tail = (done.stdout or done.stderr or "").strip().splitlines()
            outcome = {"exit": done.returncode,
                       "said": tail[-1] if tail else ""}
        except Exception as exc:                             # noqa: BLE001
            outcome = {"exit": -1, "said": f"{type(exc).__name__}: {exc}"}
        with _state_lock:
            _state["last"] = dict(job, finished=_now(), **outcome)
            _state["running"] = None
        _wanted.task_done()


def enqueue(tier: str, names: list[str]) -> dict:
    """Accept a run, or say why not. Never builds a command out of free text."""
    if tier not in TIERS:
        return {"error": f"no such tier: {tier}"}
    known = set(cases.every())
    unknown = [n for n in names if n not in known]
    if unknown:
        return {"error": f"not a case here: {', '.join(unknown)}"}
    if tier == "one" and not names:
        return {"error": "a single-case run needs a case"}
    if tier != "one":
        names = []
    with _state_lock:
        job = {"id": f"{_now()}-{len(_state['waiting'])}", "tier": tier,
               "names": names, "queued": _now()}
        _state["waiting"].append(job)
    _wanted.put(job)
    return {"queued": _label(job)}


def snapshot() -> dict:
    with _state_lock:
        return {
            "running": _label(_state["running"]) if _state["running"] else None,
            "waiting": [_label(w) for w in _state["waiting"]],
            "last": (dict(_state["last"], label=_label(_state["last"]))
                     if _state["last"] else None),
        }


class Handler(SimpleHTTPRequestHandler):
    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:                                # noqa: N802
        if self.path.split("?")[0] == "/queue":
            return self._json(snapshot())
        return super().do_GET()

    def do_POST(self) -> None:                               # noqa: N802
        if self.path.split("?")[0] != "/run":
            return self._json({"error": "no such route"}, 404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            asked = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError):
            return self._json({"error": "unreadable request"}, 400)
        answer = enqueue(str(asked.get("tier", "")),
                         [str(n) for n in (asked.get("names") or [])])
        return self._json(answer, 400 if "error" in answer else 202)

    def end_headers(self) -> None:
        # The report is rewritten by every run, and a browser told nothing
        # invents its own freshness. The same reasoning as the choir app's.
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))


def main() -> None:
    report.OUT.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_worker, daemon=True).start()
    handler = partial(Handler, directory=str(report.OUT))
    with ThreadingHTTPServer(("127.0.0.1", PORT), handler) as httpd:
        sys.stderr.write(f"serving {report.OUT} on 127.0.0.1:{PORT}\n")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
