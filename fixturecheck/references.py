"""What each case was made from, frozen as a fingerprint rather than as music.

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
    """One stamp for the whole manifest, which is what a run is keyed by."""
    sponge = hashlib.sha256()
    for name in sorted(entries):
        sponge.update(name.encode())
        sponge.update(entries[name].get("image", "").encode())
        sponge.update(entries[name].get("reference", "").encode())
    return sponge.hexdigest()[:16]


def load(path: Path = MANIFEST) -> dict:
    if not path.exists():
        return {"cases": {}}
    try:
        return json.loads(path.read_text())
    except ValueError:
        return {"cases": {}}


def write(cases: list, path: Path = MANIFEST) -> dict:
    """Freeze what is on this host now.

    Merged rather than replaced: a run of ten cases must not drop the other
    eighty-three out of the manifest, which would read in the diff as
    eighty-three references having been deleted.
    """
    held = load(path)["cases"]
    held.update({case.name: fingerprint(case) for case in cases})
    manifest = {
        "why": "A fingerprint per case, not the music: this repository is public "
               "and the songs are not ours. A reference changing is then a "
               "deliberate commit rather than a silent drift under a number. "
               "See fixturecheck/references.py.",
        "digest": digest(held),
        "cases": held,
    }
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def drift(cases: list, path: Path = MANIFEST) -> dict:
    """Which of these cases no longer match what was frozen.

    Three answers and they are not the same: `changed` is a reference that moved
    under a recorded number, `unfrozen` is one nobody has frozen yet, and a case
    absent from this run is not mentioned at all — it was not looked at.
    """
    held = load(path)["cases"]
    changed, unfrozen = [], []
    for case in cases:
        was = held.get(case.name)
        if was is None:
            unfrozen.append(case.name)
        elif was != fingerprint(case):
            changed.append(case.name)
    return {"changed": sorted(changed), "unfrozen": sorted(unfrozen)}


def stamp(cases: list, path: Path = MANIFEST) -> str:
    """What to key a run by: the frozen digest, marked when this run drifts from it.

    A run measured against references that have moved is not a run against the
    manifest, and saying so in the key is the whole reason the key exists.
    """
    manifest = load(path)
    if not manifest.get("cases"):
        return "unfrozen"
    base = manifest.get("digest") or "unfrozen"
    moved = drift(cases, path)
    marks = ""
    if moved["changed"]:
        marks += f"+drift{len(moved['changed'])}"
    if moved["unfrozen"]:
        # Not the same as drift and must not be silent: these cases were
        # measured against a reference nobody has frozen, so the manifest's
        # digest does not describe what this run was compared with.
        marks += f"+new{len(moved['unfrozen'])}"
    return base + marks
