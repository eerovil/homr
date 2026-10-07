"""Slurs against an answer key read off the page -- found, missed and invented.

The note score never looked at slurs, and the references could not be used to:
their slurs came from homr itself and nobody checked them. So the truth lives in
its own file, `homr-fixtures/slurs.json` in the private songs repository, made
by reading every slur off the printed system and checked by the owner
(eerovil/musescore-choir-plugins#318). A case with no entry there is not judged.

A slur is its staff and its two ends, each end a bar and an onset in quarter
notes -- the same keys `compare.read_score` places notes on, so a slur is found
when homr put both ends on the notes the page joins. An end that is `None` runs
off the edge of the system: a slur arriving from the system before has no start
here, one continuing into the next has no stop.

Kept out of the note score and out of the gate on purpose. The note percentage
has a history every figure in QUALITY.md is quoted against, and a slur is not a
note: missing one costs a singer a syllable, not a pitch.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from fixturecheck import cases

KEY = cases.PRIVATE / "slurs.json"

#: Where a slur end sits: (bar, onset in quarters), or None off the system edge.
End = tuple[str, float] | None
Slur = tuple[int, End, End]


def _where(bar: object, onset: object) -> tuple[str, float]:
    return str(bar), round(float(onset), 3)


def read_slurs(path: Path) -> list[Slur]:
    """Every slur in a MusicXML file, as (staff, start, stop).

    Starts and stops are paired per printed staff and slur `number`, in the
    order they are written -- what MuseScore does when it opens the file. A stop
    with nothing open is a slur arriving from before this system; a start left
    open at the end is one continuing past it.
    """
    found: list[Slur] = []
    printed = 0
    for part in ET.parse(path).getroot().findall("part"):
        staves = max((int(n.text or 1) for n in part.iter("staves")), default=1)
        base, printed = printed, printed + staves
        divisions = 1.0
        open_slurs: dict[tuple[int, str], list[tuple[str, float]]] = {}
        for measure in part.findall("measure"):
            for attributes in measure.findall("attributes"):
                declared = attributes.findtext("divisions")
                if declared:
                    divisions = float(declared) or 1.0
            at = previous = 0.0
            for node in measure:
                if node.tag == "backup":
                    at -= float(node.findtext("duration", "0")) / divisions
                    continue
                if node.tag == "forward":
                    at += float(node.findtext("duration", "0")) / divisions
                    continue
                if node.tag != "note":
                    continue
                grace = node.find("grace") is not None
                length = 0.0 if grace else float(node.findtext("duration", "0")) / divisions
                chord = node.find("chord") is not None
                onset = previous if chord else at
                if not chord:
                    previous, at = at, at + length
                staff = base + int(node.findtext("staff", "1"))
                here = _where(measure.get("number", "?"), onset)
                for slur in node.iter("slur"):
                    key = (staff, slur.get("number", "1"))
                    if slur.get("type") == "start":
                        open_slurs.setdefault(key, []).append(here)
                    elif slur.get("type") == "stop":
                        waiting = open_slurs.get(key)
                        start = waiting.pop(0) if waiting else None
                        found.append((staff, start, here))
        for (staff, _), waiting in open_slurs.items():
            found.extend((staff, start, None) for start in waiting)
    return found


def answer_key(path: Path = KEY) -> dict[str, list[dict]]:
    """The checked slurs per case, `{}` when the key is absent."""
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("cases", {})


def key_slurs(entries: list[dict]) -> list[Slur]:
    def end(e: dict | None) -> End:
        return None if e is None else _where(e["bar"], e["onset"])
    return [(int(e["staff"]), end(e.get("from")), end(e.get("to"))) for e in entries]


@dataclass
class SlurResult:
    found: int = 0
    missed: int = 0
    invented: int = 0
    #: Of the page's slurs, how many cross the system edge, and of those found.
    edge: int = 0
    edge_found: int = 0
    missed_list: list[Slur] = field(default_factory=list)
    invented_list: list[Slur] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"found": self.found, "missed": self.missed, "invented": self.invented,
                "edge": self.edge, "edge_found": self.edge_found}


def compare_slurs(want: list[Slur], got: list[Slur]) -> SlurResult:
    """Match the page's slurs to homr's, both ends exactly, each used once."""
    result = SlurResult()
    left = list(got)
    for slur in want:
        crosses = slur[1] is None or slur[2] is None
        result.edge += crosses
        if slur in left:
            left.remove(slur)
            result.found += 1
            result.edge_found += crosses
        else:
            result.missed += 1
            result.missed_list.append(slur)
    result.invented = len(left)
    result.invented_list = left
    return result


def judge(case_name: str, parsed: Path, key: dict | None = None) -> SlurResult | None:
    """The case's slurs against the key, or None when the key has no entry."""
    key = answer_key() if key is None else key
    if case_name not in key:
        return None
    return compare_slurs(key_slurs(key[case_name]), read_slurs(parsed))


def total(results: list[SlurResult]) -> dict:
    out = SlurResult()
    for r in results:
        out.found += r.found
        out.missed += r.missed
        out.invented += r.invented
        out.edge += r.edge
        out.edge_found += r.edge_found
    return out.to_json()


def describe(slur: Slur) -> str:
    staff, start, stop = slur
    a = "from the system before" if start is None else f"bar {start[0]} beat {start[1]:g}"
    b = "into the next system" if stop is None else f"bar {stop[0]} beat {stop[1]:g}"
    return f"staff {staff}: {a} -> {b}"
