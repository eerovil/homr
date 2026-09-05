"""Starting a run from the page: one at a time, and only runs that exist.

The report answered "how good is it now" without running anything, and could not
start the run that refreshes it without ssh and a checkout. These pin the two
things that makes true -- a queue rather than a collision, and an argument list
built from what the harness knows rather than from request text.
"""

from __future__ import annotations

import pytest

serve = pytest.importorskip("fixturecheck.serve")


@pytest.fixture(autouse=True)
def _empty_queue(monkeypatch):
    """Each test gets its own queue and its own idea of what cases exist."""
    import queue as queue_module

    monkeypatch.setattr(serve, "_wanted", queue_module.Queue())
    monkeypatch.setattr(serve, "_state",
                        {"running": None, "waiting": [], "last": None})
    monkeypatch.setattr(serve.cases, "every",
                        lambda: ["system4", "hanget-soi", "song-s3"])


def test_a_tier_that_is_not_a_tier_is_refused():
    """Nothing here is ever handed to a shell, and nothing is invented either."""
    assert "error" in serve.enqueue("rm -rf /", [])
    assert "error" in serve.enqueue("", [])
    assert serve.snapshot()["waiting"] == []


def test_a_case_this_host_does_not_have_is_refused():
    assert "error" in serve.enqueue("one", ["../../etc/passwd"])
    assert "error" in serve.enqueue("one", ["not-a-case"])
    assert serve.snapshot()["waiting"] == []


def test_a_single_case_run_needs_a_case():
    assert "error" in serve.enqueue("one", [])


def test_a_bulk_tier_ignores_any_names_sent_with_it():
    """`ten` and `all` are their own case lists; a name alongside means nothing."""
    assert "queued" in serve.enqueue("ten", ["system4"])
    assert serve.snapshot()["waiting"] == ["ten"]


def test_presses_queue_rather_than_collide():
    """A run is minutes of every core; two at once is two slow runs."""
    for name in ("system4", "hanget-soi"):
        assert "queued" in serve.enqueue("one", [name])
    assert "queued" in serve.enqueue("all", [])

    waiting = serve.snapshot()["waiting"]
    assert waiting == ["one: system4", "one: hanget-soi", "all"]
    # ...and they come off in the order they went on.
    assert [serve._wanted.get()["tier"] for _ in range(3)] == ["one", "one", "all"]


def test_what_is_running_and_what_finished_are_both_said():
    """A button that appears to do nothing for four minutes gets pressed again."""
    serve.enqueue("one", ["system4"])
    job = serve._wanted.get()

    with serve._state_lock:
        serve._state["running"] = job
        serve._state["waiting"] = []
    assert serve.snapshot()["running"] == "one: system4"

    with serve._state_lock:
        serve._state["running"] = None
        serve._state["last"] = dict(job, exit=1, said="GATE FAILED: hanget-soi")
    done = serve.snapshot()
    assert done["running"] is None
    assert done["last"]["label"] == "one: system4"
    # A failing gate is a result, not a crash: the report was written anyway.
    assert done["last"]["exit"] == 1
    assert "GATE FAILED" in done["last"]["said"]


def test_the_page_carries_the_controls_and_hides_them_without_a_runner():
    """The same HTML is right served or opened off disk."""
    from fixturecheck import report

    markup = report._controls(["system4", "hanget-soi"])
    assert 'id="runbar" hidden' in markup          # hidden until /queue answers
    assert 'data-tier="ten"' in markup and 'data-tier="all"' in markup
    assert "<option>hanget-soi</option>" in markup
    # `all` says what it costs before it is pressed.
    assert "35 minutes" in markup
