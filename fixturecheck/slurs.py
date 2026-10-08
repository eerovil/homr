"""Slurs and ties against an answer key read off the page -- found, missed, invented.

The note score never looked at the arcs, and the references could not be used to:
their slurs and ties came from homr itself and nobody checked them. So the truth
lives in its own file, `homr-fixtures/slurs.json` in the private songs repository,
read off the printed system and checked by the owner, arc by arc
(eerovil/musescore-choir-plugins#318). A case with no entry there is not judged; a
case listed with no arcs is judged, and catches an invented one.

Slurs and ties are counted apart. Both take a syllable away from the note they
reach, so a singer meets either one missing, but they are different marks on the
page and homr reads them with one model head: a tie read as a slur is a fault.

An arc is its kind, its staff and its two ends, each end a bar and an onset in
quarter notes -- the keys `compare.read_score` places notes on -- and, for a tie,
the pitch (letter and accidental: a male-choir reference writes an octave away
from homr), so the two tied notes of a chord are two ties. An end that is `None`
runs over the edge of the system. **homr cannot tell a slur from a tie there**:
it turns a pair into a tie only once it sees both ends, and over a line break it
sees one. So an edge end in the key is found by an edge end of either kind at the
same place, and the choir app, which sees both systems, decides which it is.

A missed arc is **marked** when homr put a red ``⚠ slur?`` or ``⚠ tie?`` in the bar
of either of its ends on that staff (eerovil/musescore-choir-plugins#328): it is
not in the file, but a person reading the score is sent to it. Marks are written
only under ``--mark-doubt``, so a run without it reports none.

Kept out of the note score and the gate on purpose. The note percentage has a
history every figure in QUALITY.md is quoted against, and an arc is not a note.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from fixturecheck import cases
from fixturecheck.compare import _voice_rank, collapse_unisons, read_score

KEY = cases.PRIVATE / "slurs.json"
KINDS = ("slur", "tie")

#: Where an arc end sits: (bar, onset in quarters), or None off the system edge.
End = tuple[str, float] | None
#: (kind, staff, voice, start, stop, pitch) -- pitch only for a tie, "" for a
#: slur. The voice is its rank on the staff (1 = the higher line), not the file's
#: own voice number: the reference and homr number voices differently, so they
#: are ranked the way the note score ranks them (`compare._voice_rank`).
Arc = tuple[str, int, int, End, End, str]


def _where(bar: object, onset: object) -> tuple[str, float]:
    return str(bar), round(float(onset), 3)


def _pitch(note: ET.Element) -> str:
    pitch = note.find("pitch")
    if pitch is None:
        return ""
    alter = {"1": "#", "-1": "b", "2": "##", "-2": "bb"}.get(pitch.findtext("alter") or "", "")
    # Without the octave: a male-choir reference writes an octave from homr.
    return f"{pitch.findtext('step')}{alter}"


def voice_ranks(path: Path) -> dict[int, dict[str, int]]:
    """Each staff's voices ranked as the note score ranks them."""
    return _voice_rank(collapse_unisons(read_score(path)))


def _rank(ranks: dict[int, dict[str, int]], staff: int, voice: str) -> int:
    return ranks.get(staff, {}).get(voice, 1)


def read_arcs(path: Path) -> list[Arc]:
    """Every slur and tie in a MusicXML file.

    Slurs pair per printed staff and `number`, ties per staff and pitch, both in
    the order they are written -- what MuseScore does when it opens the file. A
    stop with nothing open arrives from before this system; a start left open
    continues past it.
    """
    found: list[Arc] = []
    ranks = voice_ranks(path)
    printed = 0
    for part in ET.parse(path).getroot().findall("part"):
        staves = max((int(n.text or 1) for n in part.iter("staves")), default=1)
        base, printed = printed, printed + staves
        divisions = 1.0
        open_: dict[tuple, list[tuple[tuple[str, float], str]]] = {}
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
                voice = node.findtext("voice", "1")
                marks = [("slur", s.get("number", "1"), s.get("type")) for s in node.iter("slur")]
                marks += [("tie", _pitch(node), t.get("type")) for t in node.iter("tied")]
                # A note both ending one arc and starting the next: the stop first.
                marks.sort(key=lambda m: m[2] != "stop")
                for kind, label, what in marks:
                    key = (kind, staff, label)
                    if what == "start":
                        open_.setdefault(key, []).append((here, voice))
                    elif what == "stop":
                        waiting = open_.get(key)
                        start, by = waiting.pop(0) if waiting else (None, voice)
                        found.append(
                            (
                                kind,
                                staff,
                                _rank(ranks, staff, by),
                                start,
                                here,
                                label if kind == "tie" else "",
                            )
                        )
        for (kind, staff, label), waiting in open_.items():
            found.extend(
                (kind, staff, _rank(ranks, staff, by), start, None, label if kind == "tie" else "")
                for start, by in waiting
            )
    return found


def answer_key(path: Path = KEY) -> dict[str, list[dict]]:
    """The checked arcs per case, `{}` when the key is absent."""
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("cases", {})


def key_arcs(entries: list[dict], ranks: dict[int, dict[str, int]] | None = None) -> list[Arc]:
    """The key's arcs; ``ranks`` are the reference's voice ranks, which its
    voice numbers are written in."""
    ranks = ranks or {}

    def end(e: dict | None) -> End:
        return None if e is None else _where(e["bar"], e["onset"])

    arcs = []
    for e in entries:
        kind = e.get("kind", "slur")
        pitch = ""
        if kind == "tie":
            pitch = (e.get("from") or e.get("to") or {}).get("pitch", "").rstrip("0123456789")
        staff = int(e["staff"])
        voice = _rank(ranks, staff, str(e.get("voice", "1")))
        arcs.append((kind, staff, voice, end(e.get("from")), end(e.get("to")), pitch))
    return arcs


@dataclass
class KindResult:
    found: int = 0
    missed: int = 0
    invented: int = 0
    #: Of the missed, how many have a ⚠ slur?/tie? mark in an end's bar.
    marked: int = 0
    missed_list: list[Arc] = field(default_factory=list)
    invented_list: list[Arc] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "found": self.found,
            "missed": self.missed,
            "invented": self.invented,
            "marked": self.marked,
        }


@dataclass
class SlurResult:
    """One case: slurs and ties, and the ends at the system edge apart."""

    slur: KindResult = field(default_factory=KindResult)
    tie: KindResult = field(default_factory=KindResult)
    #: The page's arc ends at a system edge, and how many homr left an end at.
    edge: int = 0
    edge_found: int = 0

    def to_json(self) -> dict:
        return {
            "slur": self.slur.to_json(),
            "tie": self.tie.to_json(),
            "edge": self.edge,
            "edge_found": self.edge_found,
        }


def _edge(arc: Arc) -> bool:
    return arc[3] is None or arc[4] is None


def _edge_place(arc: Arc) -> tuple:
    return (arc[1], arc[2], arc[3] is None, arc[4] if arc[3] is None else arc[3])


#: (printed staff from 1, bar) carrying a ⚠ mark about an arc
Marks = set[tuple[int, str]]


def read_marks(path: Path) -> Marks:
    """Where homr marked a slur or tie it was unsure of (``--mark-doubt``)."""
    found: Marks = set()
    printed = 0
    for part in ET.parse(path).getroot().findall("part"):  # noqa: S314
        staves = max((int(n.text or 1) for n in part.iter("staves")), default=1)
        base, printed = printed, printed + staves
        for measure in part.findall("measure"):
            for direction in measure.findall("direction"):
                words = "".join(w.text or "" for w in direction.iter("words"))
                if words.startswith("⚠") and ("slur?" in words or "tie?" in words):
                    found.add(
                        (base + int(direction.findtext("staff", "1")), measure.get("number", "?"))
                    )
    return found


def compare_arcs(want: list[Arc], got: list[Arc], marks: Marks | None = None) -> SlurResult:
    """Match the page's arcs to homr's, both ends exactly, each used once.

    Staff and voice must agree, so a slur read on the wrong voice is missed and
    invented, not found. Inside the system a slur is matched by a slur and a tie
    by a tie at the same pitch. At the edge the kind is not asked (see the module
    docstring)."""
    result = SlurResult()
    left = list(got)
    for arc in want:
        kind = result.slur if arc[0] == "slur" else result.tie
        if _edge(arc):
            result.edge += 1
            hit = next((g for g in left if _edge(g) and _edge_place(g) == _edge_place(arc)), None)
            result.edge_found += hit is not None
        else:
            hit = arc if arc in left else None
        if hit is not None:
            left.remove(hit)
            kind.found += 1
        else:
            kind.missed += 1
            kind.missed_list.append(arc)
            ends = [end for end in (arc[3], arc[4]) if end is not None]
            kind.marked += any((arc[1], end[0]) in (marks or set()) for end in ends)
    for arc in left:
        kind = result.slur if arc[0] == "slur" else result.tie
        kind.invented += 1
        kind.invented_list.append(arc)
    return result


def judge(
    case_name: str, parsed: Path, reference: Path, key: dict | None = None
) -> SlurResult | None:
    """The case's arcs against the key, or None when the key has no entry.

    ``reference`` is the case's reference score: the key names voices by its
    numbers, and they are ranked off it."""
    key = answer_key() if key is None else key
    if case_name not in key:
        return None
    return compare_arcs(
        key_arcs(key[case_name], voice_ranks(reference)), read_arcs(parsed), read_marks(parsed)
    )


def total(results: list[SlurResult]) -> dict:
    out = SlurResult()
    for r in results:
        for kind in KINDS:
            mine, theirs = getattr(out, kind), getattr(r, kind)
            mine.found += theirs.found
            mine.missed += theirs.missed
            mine.invented += theirs.invented
            mine.marked += theirs.marked
        out.edge += r.edge
        out.edge_found += r.edge_found
    return out.to_json()


def describe(arc: Arc) -> str:
    kind, staff, voice, start, stop, pitch = arc
    a = "from the system before" if start is None else f"bar {start[0]} beat {start[1]:g}"
    b = "into the next system" if stop is None else f"bar {stop[0]} beat {stop[1]:g}"
    return f"{kind} staff {staff} voice {voice}{' ' + pitch if pitch else ''}: {a} -> {b}"
