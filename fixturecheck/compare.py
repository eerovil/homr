"""What homr wrote against what the page holds, and separately what it saw.

Two layers, kept apart on purpose. **Detection** is noteheads and stems found in
the picture; that is what the fixture tests pin. **Output** is the MusicXML homr
writes, which is what anything downstream sings from. They are not the same and
they can disagree in both directions: a note the detector misses still reaches
the output (in the wrong voice, having no stem to place it), and a head detected
a step off the page is still written at the right pitch, because the pitch is
read from the image by the transformer rather than taken from that geometry.

Three reports were filed in one day claiming homr had misread music when what
had been compared was the detector. So the two are separate functions here, they
return separate results, and the report names which is which.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

STEPS = "CDEFGAB"
CLEF_REFERENCE = {"G": ("G", 4), "F": ("F", 3), "C": ("C", 4)}


def _diatonic(step: str, octave: int) -> int:
    return 7 * octave + STEPS.index(step)


def _clefs(measure: ET.Element) -> dict[int, tuple[str, int, int]]:
    found = {}
    for attributes in measure.findall("attributes"):
        for clef in attributes.findall("clef"):
            found[int(clef.get("number", "1"))] = (
                clef.findtext("sign", "G"),
                int(clef.findtext("line", "2")),
                int(clef.findtext("clef-octave-change", "0")),
            )
    return found


def staff_position(pitch: ET.Element, clef: tuple[str, int, int]) -> int:
    """Bottom line 1, one step per line or space.

    Position and not pitch name, because a male-choir score is written an octave
    above where it sounds: the reference says A3 where homr says A4 for the same
    printed note, and comparing names would report every note in the piece.
    """
    sign, line, octave_change = clef
    reference_step, reference_octave = CLEF_REFERENCE[sign]
    written = _diatonic(pitch.findtext("step", "C"), int(pitch.findtext("octave", "4")))
    return 2 * line - 1 + written - 7 * octave_change - _diatonic(reference_step, reference_octave)


def read_score(path: Path) -> dict[tuple[str, int, float], list[dict]]:
    """Every sounding note, keyed by when and where it sounds.

    Notes are placed by walking the measure with the duration cursor -- a note
    advances it, a chord tone does not, a backup winds it back -- and not by
    document order: the reference writes one voice out and backs up for the next
    while homr interleaves them, so document order pairs a tenor with a bass. It
    did, and produced twenty confident nonsense rows.

    The printed staff is counted across the whole file rather than read from
    `<staff>`, because both sides write their staves as separate parts that each
    call themselves staff 1, and keying on that folds the bass into the treble.
    """
    found: dict[tuple[str, int, float], list[dict]] = {}
    printed = 0
    for part in ET.parse(path).getroot().findall("part"):
        staves = max((int(n.text or 1) for n in part.iter("staves")), default=1)
        base, printed = printed, printed + staves
        clefs: dict[int, tuple[str, int, int]] = {}
        for measure in part.findall("measure"):
            clefs.update(_clefs(measure))
            at = previous = 0.0
            for node in measure:
                if node.tag == "backup":
                    at -= float(node.findtext("duration", "0"))
                    continue
                if node.tag == "forward":
                    at += float(node.findtext("duration", "0"))
                    continue
                if node.tag != "note":
                    continue
                length = float(node.findtext("duration", "0"))
                onset = previous if node.find("chord") is not None else at
                if node.find("chord") is None:
                    previous, at = at, at + length
                if node.find("rest") is not None:
                    continue
                pitch = node.find("pitch")
                if pitch is None:
                    continue
                inner = int(node.findtext("staff", "1"))
                found.setdefault((measure.get("number", "?"), base + inner, round(onset, 3)),
                                 []).append({
                    "voice": node.findtext("voice", "1"),
                    "name": f"{pitch.findtext('step')}{pitch.findtext('octave')}",
                    "position": staff_position(pitch, clefs.get(inner, ("G", 2, 0))),
                    "stem": node.findtext("stem", ""),
                    "chord": node.find("chord") is not None,
                })
    return found


def collapse_unisons(found: dict) -> dict:
    """Two voices in unison print ONE notehead, so count it once.

    Older choral engraving writes a unison as a single head serving both voices,
    and a reference with one staff per part has to write it twice. Counting those
    two against homr's one reported a lost note on ten moments of Sammon ryosto,
    every one of them a unison and not one of them wrong.
    """
    merged: dict = {}
    for key, group in found.items():
        seen: dict[int, dict] = {}
        for note in group:
            at = seen.get(note["position"])
            if at is None:
                seen[note["position"]] = dict(note, unison=False)
            else:
                at["unison"] = True
        merged[key] = list(seen.values())
    return merged


def _voice_rank(found: dict) -> dict[int, dict[str, int]]:
    """The staff's voices as first, second, ... counted from the top line down.

    Compared as a rank and not as a number: the two files number voices
    differently and neither numbering means anything on its own.
    """
    order: dict[int, list[str]] = {}
    for (_, staff, _), group in sorted(found.items()):
        for note in sorted(group, key=lambda n: -n["position"]):
            order.setdefault(staff, [])
            if note["voice"] not in order[staff]:
                order[staff].append(note["voice"])
    return {staff: {voice: rank + 1 for rank, voice in enumerate(voices)}
            for staff, voices in order.items()}


@dataclass
class Row:
    where: str
    page: str
    homr: str
    verdict: str
    kind: str          # agree | voice | pitch | size | unison


@dataclass
class Result:
    agree: int = 0
    voice: int = 0
    pitch: int = 0
    size: int = 0
    unison: int = 0
    rows: list[Row] = field(default_factory=list)

    @property
    def judged(self) -> int:
        return self.agree + self.voice + self.pitch

    @property
    def score(self) -> float:
        return 100.0 * self.agree / self.judged if self.judged else 0.0

    @property
    def faults(self) -> int:
        return self.voice + self.pitch + self.size


def compare_output(reference: Path, parsed: Path) -> Result:
    """The page against homr's MusicXML, note by note."""
    want = collapse_unisons(read_score(reference))
    got = read_score(parsed)
    here, there = _voice_rank(want), _voice_rank(got)
    result = Result()
    for key in sorted(want, key=lambda k: (int(re.sub(r"\D", "", k[0]) or 0), k[1], k[2])):
        bar, staff, onset = key
        where = f"bar {bar}, staff {staff}, beat {onset:g}"
        mine = sorted(want[key], key=lambda n: -n["position"])
        theirs = sorted(got.get(key, []), key=lambda n: -n["position"])
        if len(mine) != len(theirs):
            result.size += 1
            result.rows.append(Row(where, f"{len(mine)} notehead(s)",
                                   f"{len(theirs)} notehead(s)",
                                   "a different number of notes here", "size"))
            continue
        for a, b in zip(mine, theirs):
            if a["position"] != b["position"]:
                result.pitch += 1
                gap = abs(a["position"] - b["position"])
                result.rows.append(Row(
                    where, f"{a['name']} · position {a['position']}",
                    f"{b['name']} · position {b['position']}",
                    f"a different note, {gap} position(s) "
                    f"{'higher' if b['position'] > a['position'] else 'lower'}", "pitch"))
                continue
            if a.get("unison"):
                result.unison += 1
                result.rows.append(Row(where, f"{a['name']} · both voices", b["name"],
                                       "a unison — one head, both parts", "unison"))
                continue
            mine_rank, their_rank = here[staff][a["voice"]], there[staff][b["voice"]]
            if mine_rank != their_rank:
                result.voice += 1
                result.rows.append(Row(
                    where, f"{a['name']} · voice {mine_rank}",
                    f"{b['name']} · voice {their_rank}"
                    + ("" if b["stem"] else " (no stem)"),
                    "folded into the other voice as a chord member" if b["chord"]
                    else "written in the other voice", "voice"))
                continue
            result.agree += 1
            result.rows.append(Row(where, f"{a['name']} · voice {mine_rank}",
                                   f"{b['name']} · voice {their_rank}",
                                   "the voice the page prints", "agree"))
    return result
