"""A case is one printed system, and where its three files come from.

The five fixtures in `fixtures/` and the ninety-odd printed systems of the songs
on this host are the same object: a picture of one system, a reference saying
what that system holds, and a parse to be judged against it.  Keeping them one
type is the whole point -- anything that can judge a fixture can judge every
system of every song we sing, which is a corpus three orders of magnitude larger
and made of ordinary music rather than of cases chosen because something was
wrong with them.

A committed fixture carries its own picture and reference because its song
cannot be committed.  A song case is built on demand from the choir app's own
files, by the choir app's own code -- the band a person drew, cropped from the
PDF, and that song's cleaned score imploded back to the printed shape and
trimmed to those bars.

**What is cached is decided by what each file depends on.** The crop and the
reference change when the PDF, the bounds or the cleaned score change, and
nothing else; a parse changes when homr changes.  So an ordinary edit to homr
re-runs homr and nothing else, and the poppler and MuseScore work -- half the
wall clock -- is already on disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
CACHE = ROOT / ".check-cache"
#: The choir app's checkout, which owns the songs. Configurable because it is
#: host state: a fresh clone of this fork has fixtures and no songs, and must
#: still be able to run the committed ones.
CHOIR = Path(os.environ.get("CHOIR_REPO", Path.home() / "musescore-choir-plugins"))


@dataclass(frozen=True)
class Case:
    """One printed system: a picture, a reference, and where they came from."""

    name: str
    image: Path
    reference: Path
    origin: str

    @property
    def committed(self) -> bool:
        return self.origin == "fixture"


def _digest(*paths: Path) -> str:
    """A stamp of the files a built case was made from."""
    sponge = hashlib.sha256()
    for path in paths:
        sponge.update(path.name.encode())
        sponge.update(path.read_bytes() if path.exists() else b"")
    return sponge.hexdigest()[:16]


def committed_cases() -> list[Case]:
    """The fixtures in this repository."""
    listed = json.loads((FIXTURES / "stem-direction-fixtures.json").read_text())
    return [
        Case(
            name=name,
            image=FIXTURES / entry["image"],
            reference=FIXTURES / entry["reference"],
            origin="fixture",
        )
        for name, entry in sorted(listed["fixtures"].items())
    ]


def song_systems() -> list[tuple[str, int]]:
    """Every printed system of every song the choir app's manifest keeps."""
    manifest = CHOIR / "fixtures" / "omr-songs.json"
    if not manifest.exists():
        return []
    found: list[tuple[str, int]] = []
    for slug, entry in json.loads(manifest.read_text())["songs"].items():
        if entry.get("review", {}).get("status") == "excluded":
            continue
        bounds = CHOIR / "songs" / slug / ".systems.json"
        if not bounds.exists():
            continue
        count = len(json.loads(bounds.read_text()).get("systems", []))
        found.extend((slug, index) for index in range(1, count + 1))
    return found


def song_case(slug: str, index: int) -> Case | None:
    """Build -- or reuse -- the picture and reference for one printed system.

    The building is the choir app's `make_stem_fixture`, run as a subprocess
    rather than imported: songs are that repository's business, it already knows
    how to trim a cleaned score to a band's bars, and reaching across two
    checkouts with `sys.path` to borrow it would tie this to its internals.

    Its two outputs share a stem, and so does homr's -- `<image>.musicxml` --
    which is how ninety-three references came to be silently overwritten by the
    parses of the same systems. Here they are moved into separate folders on the
    way in, so no two files this harness writes can ever collide.
    """
    name = f"{slug}-s{index}"
    pages, refs = CACHE / "pages", CACHE / "refs"
    pages.mkdir(parents=True, exist_ok=True)
    refs.mkdir(parents=True, exist_ok=True)
    image, reference = pages / f"{name}.png", refs / f"{name}.musicxml"
    stamp_file = CACHE / "built" / f"{name}.stamp"
    stamp_file.parent.mkdir(parents=True, exist_ok=True)

    song = CHOIR / "songs" / slug
    sources = [song / ".systems.json"]
    sources += sorted(song.glob("*_cleaned.mscx"))
    stamp = _digest(*sources)
    if image.exists() and reference.exists() and stamp_file.exists() \
            and stamp_file.read_text() == stamp:
        return Case(name, image, reference, f"{slug} system {index}")

    with tempfile.TemporaryDirectory(prefix="case-") as tmp:
        built = subprocess.run(
            [str(CHOIR / ".venv/bin/python"), "scripts/make_stem_fixture.py",
             slug, str(index), "--name", name],
            cwd=CHOIR, capture_output=True, text=True, timeout=900,
            env={**os.environ, "HOMR_FIXTURES": tmp},
        )
        made_image, made_reference = Path(tmp) / f"{name}.png", Path(tmp) / f"{name}.musicxml"
        if built.returncode != 0 or not made_image.exists() or not made_reference.exists():
            return None
        shutil.copy(made_image, image)
        shutil.copy(made_reference, reference)
    stamp_file.write_text(stamp)
    return Case(name, image, reference, f"{slug} system {index}")


def sample() -> list[str]:
    """The ten, as a written-down list rather than a slice of whatever is first.

    A sample that changes between runs cannot be compared between runs, and a
    sample taken with `head` is one engraver's hand: the first attempt at this
    took eighteen crops and got seven systems of a single song.
    """
    listed = json.loads((Path(__file__).parent / "sample.json").read_text())
    return list(listed["cases"])


def resolve(names: list[str]) -> list[Case]:
    """Turn case names into cases, building song systems where needed."""
    committed = {case.name: case for case in committed_cases()}
    found: list[Case] = []
    for name in names:
        if name in committed:
            found.append(committed[name])
            continue
        slug, _, index = name.rpartition("-s")
        if not index.isdigit():
            continue
        case = song_case(slug, int(index))
        if case is not None:
            found.append(case)
    return found


def every() -> list[str]:
    """Every case this host can offer: the fixtures, then the songs."""
    return [case.name for case in committed_cases()] + [
        f"{slug}-s{index}" for slug, index in song_systems()
    ]
