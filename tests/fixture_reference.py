"""Read a fixture's reference MusicXML as printed noteheads per staff.

The reference files carry MuseScore's own engraving coordinates: every measure
has a ``width`` and every note a ``default-x`` relative to its measure, both in
tenths.  A fixture is one printed system, so a note's horizontal place in the
system is the cumulative measure width plus its ``default-x`` -- no rendering
and no reflow.  ``default-y`` is measured from the staff's top line, which is
staff position 9 in homr's numbering (bottom line 1, one unit per line/space).
"""

import xml.etree.ElementTree as ET
from pathlib import Path

STEPS = "CDEFGAB"
CLEF_REFERENCE = {"G": ("G", 4), "F": ("F", 3), "C": ("C", 4)}


def _diatonic(step: str, octave: int) -> int:
    return 7 * octave + STEPS.index(step)


def _staff_count(part: ET.Element) -> int:
    counts = [int(node.text or "1") for node in part.iter("staves")]
    return max(counts) if counts else 1


def _clefs(measure: ET.Element) -> dict[int, tuple[str, int, int]]:
    """The clefs this measure declares, by staff number."""
    found = {}
    for attributes in measure.findall("attributes"):
        for clef in attributes.findall("clef"):
            number = int(clef.get("number", "1"))
            sign = clef.findtext("sign", "G")
            line = int(clef.findtext("line", "2"))
            octave_change = int(clef.findtext("clef-octave-change", "0"))
            found[number] = (sign, line, octave_change)
    return found


def _staff_position(pitch: ET.Element, clef: tuple[str, int, int]) -> int:
    """Where a pitch sits on the staff: bottom line 1, one step per line or space.

    This is printed geometry derived from the music itself, so it does not
    depend on the reference file's page layout the way ``default-y`` does.
    """
    sign, line, octave_change = clef
    reference_step, reference_octave = CLEF_REFERENCE[sign]
    step = pitch.findtext("step", "C")
    octave = int(pitch.findtext("octave", "4"))
    written = _diatonic(step, octave) - 7 * octave_change
    return 2 * line - 1 + written - _diatonic(reference_step, reference_octave)


def reference_staffs(path: Path) -> list[dict]:
    """Return one entry per printed staff, top to bottom, with its notes."""
    root = ET.parse(path).getroot()
    staffs: list[dict] = []
    for part in root.findall("part"):
        by_staff: dict[int, list[dict]] = {
            index: [] for index in range(1, _staff_count(part) + 1)
        }
        origin = 0.0
        divisions = 1.0
        clefs: dict[int, tuple[str, int, int]] = {}
        for index, measure in enumerate(part.findall("measure")):
            clefs.update(_clefs(measure))
            divisions = float(measure.findtext("attributes/divisions") or divisions)
            # Follow the measure's cursor so every note knows when it sounds:
            # a note advances it, a chord note shares the moment before it, and
            # backup winds it back for the next voice.
            cursor = 0.0
            previous = 0.0
            for element in measure:
                if element.tag == "backup":
                    cursor -= float(element.findtext("duration", "0"))
                    continue
                if element.tag == "forward":
                    cursor += float(element.findtext("duration", "0"))
                    continue
                if element.tag != "note":
                    continue
                duration = float(element.findtext("duration", "0"))
                if element.find("chord") is not None:
                    onset = previous
                else:
                    onset, previous = cursor, cursor
                    cursor += duration
                pitch = element.find("pitch")
                default_x = element.get("default-x")
                if pitch is None or default_x is None or element.find("grace") is not None:
                    continue
                staff = int(element.findtext("staff", "1"))
                by_staff.setdefault(staff, []).append(
                    {
                        "x": round(origin + float(default_x), 2),
                        "moment": (index, round(onset / divisions, 4)),
                        "position": _staff_position(pitch, clefs.get(staff, ("G", 2, 0))),
                        "stem": element.findtext("stem"),
                        "voice": element.findtext("voice", "1"),
                        "measure": measure.get("number"),
                        "step": pitch.findtext("step"),
                        "octave": pitch.findtext("octave"),
                    }
                )
            origin += float(measure.get("width", "0"))
        for staff in sorted(by_staff):
            notes = sorted(by_staff[staff], key=lambda note: (note["x"], -note["position"]))
            staffs.append({"part": part.get("id"), "staff": staff, "notes": notes})
    return staffs
