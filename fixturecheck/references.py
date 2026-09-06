"""What each case was made from, and how well it last read.

Two things are frozen here and they are frozen together on purpose: the
**fingerprint** of a case's two files, and the **memory** of what homr scored on
it when somebody last accepted that reading. A memory is a claim about a
particular picture and a particular reference, so it is worth nothing once
either of them moves — keeping both in one manifest, written by one command, is
what stops a number outliving the music it was about.

A reference here is a song's cleaned score imploded back to the shape of the
print. Those scores are edited — nineteen of the ninety-three were wrong about
their own staves on the day this was written — so a reference is not a fixed
thing, and a series built against a moving reference partly measures the
reference. A number can improve because somebody corrected a score.

The obvious fix is to commit the references. **This repository is public and the
music is not ours.** The ninety-three systems are Fazer, Sulasol, Breitkopf and
Fennica Gehrman; the five committed fixtures were a deliberate handful, and
ninety-three cropped systems with their transcriptions is a different thing
entirely.

So what is committed is a **fingerprint per case** — the picture and the
reference, each as a hash — and the music stays on the host that owns the songs.
That buys the property the freezing was for: a reference changing is a line in a
diff that somebody had to commit, not a silent drift under a number. What it
does not buy, and this is worth saying plainly rather than discovering later, is
reproducibility from a clone: a fresh checkout has the five fixtures and no
songs, and cannot rebuild the other eighty-eight to check any figure in the
series against them.

    python -m fixturecheck freeze      # write the manifest from what is here now

**The alarm rule is per case: nothing gets worse.** Each case remembers the
notes-right score it was last accepted at; a run fails the gate if any case
reads below its own memory, and an improvement moves that memory up. Not a
total across cases — a win on one page must not pay for a loss on another, which
is the whole reason the memory is per case rather than one figure for the run.

**A run only ever moves a memory up. A person can move one down**, by naming
the case:

    python -m fixturecheck accept sammon-ryosto    # yes, I meant that

That is the escape hatch, and it has to exist. A regression is not always a
mistake — an intentional trade-off in the model is a regression on some page —
and a gate with no way to say "yes, I meant that" is one people route around.
It is a command of its own rather than a flag on a run, and it takes the cases
by name rather than accepting everything, because the whole value of the path
is that somebody had to say which case and mean it.

**A pass/fail per case is what this replaces**, and #155 settled why. The old
gate asked whether each of the five committed fixtures was `perfect` — every
note right, no argument about the staves or the meter. #152 shaved two pixels
off the top of a band and watched that verdict go from 7/10 to 3/10 while the
notes-right figure moved 98.4% to 97.3%: a two-pixel reframe moves about one
note in ninety, and a zero-fault boolean amplified that one note into a flipped
verdict on five cases. The measurement under it was never the brittle part. So
what is remembered is the score, which moves by about as much as the reading
really moved.

**Structure and meter are remembered too, and must not rise.** They are counted
as cases rather than as notes — one wrong answer about the staves, one bar in
the wrong meter — so neither is in the notes-right percentage and a memory made
only of that percentage would have quietly dropped the meter gate #175 built.
"Nothing gets worse" is the rule; these are two more ways of getting worse.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent / "references.json"


def _hash(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def fingerprint(case) -> dict:
    """One case's two files, as hashes."""
    return {"image": _hash(case.image), "reference": _hash(case.reference)}


def digest(entries: dict) -> str:
    """One stamp for the whole manifest, which is what a run is keyed by.

    **The memories are deliberately not in it.** The digest says which music was
    measured; a case reading better than it used to is not different music. Were
    the scores hashed here, every improvement would start a fresh identity and
    every case would go back to `unevaluated` under it — the ratchet would erase
    the record it is meant to keep.

    That has to hold for the *name* as well as the hashes, hence the skip: a run
    can record a memory for a case nobody has frozen, and if that entry entered
    the digest the first sighting of an unfrozen case would move the identity
    just as surely as hashing its score would.
    """
    sponge = hashlib.sha256()
    for name in sorted(entries):
        entry = entries[name]
        if not frozen(entry):
            continue
        sponge.update(name.encode())
        sponge.update(entry.get("image", "").encode())
        sponge.update(entry.get("reference", "").encode())
    return sponge.hexdigest()[:16]


def frozen(entry: dict) -> bool:
    """Whether this entry says anything about the case's *files*.

    An entry can hold a memory and no fingerprint — a case that has been read
    but never frozen. Read as a fingerprint of nothing it looks like a reference
    that has been emptied, so the case reports as drifted on every later run and
    is held out of the gate for good.
    """
    return bool(entry.get("image") or entry.get("reference"))


# --- how well a case last read -------------------------------------------

#: What a case is remembered by. `score` may not fall; the other two may not
#: rise. Named here rather than spelled out at each use, so adding a fourth way
#: of getting worse is one line.
MEMORY = ("score", "structure", "meter")

#: How much a score may fall and still count as the same reading. Zero, and
#: that is the point of the card: a real regression shows up as notes, and
#: giving the alarm a tolerance would be re-inventing by the back door the
#: slack that a per-case pass/fail was criticised for not having. The rounding
#: to two places is what absorbs float noise.
SLACK = 0.0


def marks(counts: dict) -> dict:
    """The three numbers a case is remembered by, out of one run's counts.

    Takes the counts as the series records them rather than a `Result`, because
    the gate has to be able to judge a case out of the record as well as out of
    a live run, and two spellings of the same score is how they come to
    disagree. `test_fixturecheck_memory` pins this against `Result.score`.
    """
    right = counts.get("agree", 0)
    judged = sum(counts.get(k, 0) for k in ("agree", "voice", "pitch", "size", "timing"))
    return {
        "score": round(100.0 * right / judged, 2) if judged else 0.0,
        "structure": int(counts.get("staves_page", 0) != counts.get("staves_homr", 0)),
        "meter": int(counts.get("meter", 0)),
    }


def worse(now: dict, was: dict) -> list[str]:
    """Which of the remembered numbers got worse, said in words.

    A list rather than a boolean, because the report has to name what moved:
    "below its memory" with no number in it is the kind of alarm people learn to
    click through.
    """
    said: list[str] = []
    if now.get("score", 0.0) < was.get("score", 0.0) - SLACK:
        said.append(f"{now.get('score', 0.0):.2f}% of notes right, "
                    f"against {was.get('score', 0.0):.2f}% accepted")
    for field in ("structure", "meter"):
        if now.get(field, 0) > was.get(field, 0):
            said.append(f"{field} {now.get(field, 0)}, against "
                        f"{was.get(field, 0)} accepted")
    return said


def better(now: dict, was: dict) -> bool:
    """Whether this reading is one the memory should move up to."""
    return not worse(now, was) and any(now.get(f) != was.get(f) for f in MEMORY)


def accepted(path: Path | None = None) -> dict:
    """Every case's memory, by name. Absent means nobody has accepted a reading."""
    held = load(path)["cases"]
    return {name: {f: entry[f] for f in MEMORY}
            for name, entry in held.items() if all(f in entry for f in MEMORY)}


def remember(readings: dict, path: Path | None = None) -> list[str]:
    """Move the named cases' memories to whatever is passed in, up or down.

    **This writes in either direction and does not judge**, which is why the
    one-way rule lives in its callers rather than here. An ordinary run offers
    it only readings that are not worse (`__main__.ratchet`); `accept` offers it
    whatever the case reads now, because that is a person saying so out loud.

    Both paths were needed and only the first existed. The escape hatch was
    documented as `freeze`, and `freeze` cannot do it: `write` is about the
    *files*, and it deliberately keeps a memory whose fingerprint has not
    moved. So a case that regressed on purpose — an intentional model
    trade-off — could never be accepted, and the gate failed forever unless
    somebody hand-edited this file. A gate with no way to say "yes, I meant
    that" is one people route around.
    """
    path = path or MANIFEST
    manifest = load(path)
    held = manifest.setdefault("cases", {})
    moved: list[str] = []
    for name, reading in readings.items():
        entry = held.setdefault(name, {})
        if all(entry.get(f) == reading.get(f) for f in MEMORY):
            continue
        entry.update({f: reading[f] for f in MEMORY})
        moved.append(name)
    if moved:
        manifest.setdefault("why", _WHY)
        manifest["digest"] = digest(held)
        path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return sorted(moved)


def load(path: Path | None = None) -> dict:
    """Read the manifest.

    `MANIFEST` is resolved here rather than bound as a default, so pointing the
    module at another file actually moves every reader *and every writer*. Bound
    as a default it did not, and the tests around this file had to stub the
    writer out — which meant the escape hatch was covered by a reimplementation
    of itself and a sabotage of the real one passed. `series.runs` learned the
    same lesson first and says so in its own docstring.
    """
    path = path or MANIFEST
    if not path.exists():
        return {"cases": {}}
    try:
        return json.loads(path.read_text())
    except ValueError:
        return {"cases": {}}


_WHY = ("A fingerprint per case and the score it was last accepted at — not the "
        "music: this repository is public and the songs are not ours. A "
        "reference changing, or a case being accepted lower than it stood, is "
        "then a deliberate commit rather than a silent drift under a number. "
        "See fixturecheck/references.py.")


def write(cases: list, path: Path | None = None) -> tuple[dict, list[str]]:
    """Freeze what is on this host now, and say whose memory that cost.

    Merged rather than replaced: a run of ten cases must not drop the other
    eighty-three out of the manifest, which would read in the diff as
    eighty-three references having been deleted.

    **A case whose files moved loses its memory here.** The score was about the
    picture and the reference that have just been replaced, so carrying it over
    would gate a new case against an old case's number — and it would do it
    silently, since freezing is exactly the moment nobody is looking at scores.
    The next run records what the new files read, as a first sighting.

    **And a case whose files did not move keeps its memory**, which is right for
    what this does and is why it is *not* the way to accept a regression. This
    is about the files; it never reads a case and has no measurement to write.
    `accept` is the path that does — see the module docstring.
    """
    path = path or MANIFEST
    held = load(path)["cases"]
    forgotten: list[str] = []
    for case in cases:
        now = fingerprint(case)
        entry = held.get(case.name, {})
        kept = {f: entry[f] for f in MEMORY if f in entry}
        if kept and any(entry.get(k) != now[k] for k in ("image", "reference")):
            forgotten.append(case.name)
            kept = {}
        held[case.name] = {**now, **kept}
    manifest = {"why": _WHY, "digest": digest(held), "cases": held}
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest, sorted(forgotten)


def drift(cases: list, path: Path | None = None) -> dict:
    """Which of these cases no longer match what was frozen.

    Three answers and they are not the same: `changed` is a reference that moved
    under a recorded number, `unfrozen` is one nobody has frozen yet, and a case
    absent from this run is not mentioned at all — it was not looked at.

    Only the two hashes are compared. A manifest entry also carries the case's
    memory, and a case reading better than it did is not its picture moving.
    """
    held = load(path)["cases"]
    changed, unfrozen = [], []
    for case in cases:
        was = held.get(case.name)
        if was is None or not frozen(was):
            unfrozen.append(case.name)
        elif {k: was.get(k) for k in ("image", "reference")} != fingerprint(case):
            changed.append(case.name)
    return {"changed": sorted(changed), "unfrozen": sorted(unfrozen)}


def judge(readings: dict, unread: list[str], adrift: list[str] | None = None,
          path: Path | None = None) -> dict:
    """Has anything in this run got worse than the reading it was accepted at?

    `readings` is `marks()` per case that homr read; `unread` is the cases homr
    ran on and produced nothing usable for. The three tiers are judged by this
    one function and nothing in it knows which tier a case belongs to — a pin, a
    committed fixture and a song system are the same object with the same rule,
    which is what stops a win on the fixtures paying for a loss on real songs.

    **A case homr could no longer read at all is the loudest regression there
    is**, so a remembered case that comes back unreadable fails. A case that
    could not be *built* does not appear here: that is this host missing a song,
    not homr misreading one.

    **A case whose reference has moved is held rather than judged.** Its memory
    is a number about music that has since been edited, and failing a run for
    that would blame homr for somebody correcting a score. `freeze` is how such
    a case rejoins the gate.
    """
    was = accepted(path)
    adrift = set(adrift or [])
    below: dict[str, list[str]] = {}
    for name, reading in readings.items():
        if name in adrift or name not in was:
            continue
        said = worse(reading, was[name])
        if said:
            below[name] = said
    lost = sorted(n for n in unread if n in was and n not in adrift)
    judged = sorted(n for n in readings if n in was and n not in adrift)
    return {
        "judged": judged,
        "below": {n: below[n] for n in sorted(below)},
        "unreadable": lost,
        "unremembered": sorted(n for n in list(readings) + list(unread)
                               if n not in was and n not in adrift),
        "adrift": sorted(n for n in list(readings) + list(unread) if n in adrift),
        "passed": not below and not lost,
    }


def stamp(cases: list, path: Path | None = None) -> str:
    """What to key a run by: the frozen digest, marked when this run drifts from it.

    A run measured against references that have moved is not a run against the
    manifest, and saying so in the key is the whole reason the key exists.
    """
    manifest = load(path)
    if not manifest.get("cases"):
        return "unfrozen"
    base = manifest.get("digest") or "unfrozen"
    moved = drift(cases, path)
    mark = ""
    if moved["changed"]:
        mark += f"+drift{len(moved['changed'])}"
    if moved["unfrozen"]:
        # Not the same as drift and must not be silent: these cases were
        # measured against a reference nobody has frozen, so the manifest's
        # digest does not describe what this run was compared with.
        mark += f"+new{len(moved['unfrozen'])}"
    return base + mark
