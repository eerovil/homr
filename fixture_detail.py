#!/usr/bin/env python3
"""One fixture in full: every printed notehead, and what homr made of it.

``stem_failure_report.py`` draws the failures.  This draws everything, which is
the other half of trusting a fixture: a report that shows only what went wrong
cannot tell you whether the 56 notes it stayed quiet about were read correctly
or never looked at.  Every printed moment gets a row, both staves, matched or
not.

    .venv/bin/python fixture_detail.py laulun-aika-s2

Writes ``fixture-detail/<name>.html``.  Run ``fixture_probe.py <name>`` first.
"""

import html
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import xml.etree.ElementTree as ET

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from homr.main import load_and_preprocess_predictions, predict_symbols  # noqa: E402
from stem_blame import Quiet, verdict  # noqa: E402
from stem_failure_report import CODES, crop, findings  # noqa: E402
from tests.fixture_matching import (  # noqa: E402
    align_columns,
    detected_columns,
    reference_columns,
    _heads_in_column,
)
from tests.fixture_reference import (  # noqa: E402
    STEPS,
    _clefs,
    _staff_position,
    reference_staffs,
)

#: The homr this host installs, borrowed for its dependencies while the code
#: comes from this working copy -- the same way the choir app runs a checkout.
HOMR_VENV = Path(os.environ.get(
    "HOMR_VENV", Path.home() / ".local/share/musescore-choir-plugins/homr-venv"))
ROOT = Path(__file__).parent
FIXTURES = ROOT / "fixtures"
OUT = ROOT / "fixture-detail"

STYLE = """
body { font: 14px/1.55 system-ui, sans-serif; margin: 0 auto; max-width: 1150px;
       padding: 24px; color: #1a1a1a; background: #fafafa; }
h1 { font-size: 22px; margin-bottom: 2px; }
h2 { font-size: 17px; margin: 30px 0 6px; }
p.lead { color: #555; margin-top: 0; }
img.page { width: 100%; border: 1px solid #e2e2e2; border-radius: 6px;
           background: #fff; margin-bottom: 8px; }
table { border-collapse: collapse; width: 100%; background: #fff;
        border: 1px solid #e2e2e2; border-radius: 6px; overflow: hidden; }
th, td { padding: 5px 9px; text-align: left; border-bottom: 1px solid #f0f0f0;
         font-variant-numeric: tabular-nums; }
th { background: #f4f4f4; font-size: 12px; text-transform: uppercase;
     letter-spacing: .04em; color: #555; }
tr.bad td { background: #fdecec; }
tr.gap td { background: #fff8e1; }
tr.warn td { background: #eef4ff; }
td.ok { color: #1c5c2c; }
td.no { color: #8a1f1f; font-weight: 600; }
.mono { font-family: ui-monospace, monospace; font-size: 12px; }
figure { margin: 0 0 14px; background: #fff; border: 1px solid #e2e2e2;
         border-radius: 8px; padding: 10px; }
figure img { width: 49%; border: 1px solid #eee; }
figcaption { font-size: 13px; margin-top: 6px; }
.tag { border-radius: 4px; padding: 1px 7px; font-size: 12px; font-weight: 600; }
.wrong { background: #fdecec; color: #8a1f1f; }
.missing { background: #fff3cd; color: #7a5b00; }
.extra { background: #e7edfb; color: #1b3a7a; }
.id { font-family: ui-monospace, monospace; color: #555; }
.sum { display: flex; gap: 22px; flex-wrap: wrap; margin: 6px 0 14px; }
.sum b { font-size: 20px; display: block; }
"""


def bottom_line(notes: list[dict]) -> int | None:
    """What pitch the staff's bottom line is, read off the reference itself.

    Deriving it from the notes rather than the clef keeps this honest about
    transposition: the reference already knows where each of its own pitches
    sits, so the offset is a subtraction, and a staff whose clef the file writes
    unusually cannot be spelled wrongly here.
    """
    for note in notes:
        diatonic = 7 * int(note["octave"]) + STEPS.index(note["step"])
        return diatonic - int(round(note["position"]))
    return None


def spell(position: float, bottom: int | None) -> str:
    """A staff position said as a pitch, so a row can be read as music.

    The tables are position numbers otherwise, which is the level the detector
    works at and nobody reads a score at.
    """
    if bottom is None:
        return ""
    note = bottom + int(round(position))
    return f"{STEPS[note % 7]}{note // 7}"


def homr_output(image: Path, into: Path) -> str:
    """Engrave what homr actually writes for this image, MusicXML and all.

    The rest of this page is homr's *detection*: noteheads and stems found in
    the picture. That is what the fixture tests, but it is not what comes out of
    homr, and a reader who cannot see the music has no way to tell a fixture
    failure from a parse that is wrong in some way the fixture never asked
    about.
    """
    cli = (os.environ.get("MUSESCORE_CLI_PATH") or "musescore3").strip().strip('"')
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / image.name
        shutil.copy(image, copy)
        run = subprocess.run(
            [str(HOMR_VENV / "bin/python"), "-c",
             "from homr.main import main; main()", str(copy), "--gpu", "no"],
            capture_output=True, text=True, timeout=900,
            env={**os.environ, "PYTHONPATH": str(ROOT)})
        parsed = copy.with_suffix(".musicxml")
        if run.returncode != 0 or not parsed.exists():
            return ""
        shutil.copy(parsed, into.parent / f"{into.stem}.musicxml")
        engraved = subprocess.run(
            [cli, "-T", "10", "-r", "200", str(parsed), "-o", str(into)],
            capture_output=True, text=True, timeout=600)
        numbered = into.with_name(f"{into.stem}-1.png")
        if numbered.exists():
            numbered.replace(into)
        return into.name if engraved.returncode == 0 and into.exists() else ""


def voice_lines(reference: Path, parsed: Path) -> tuple[list[str], dict]:
    """Which voice each notehead ended up in, the page against homr's output.

    This is the half the fixture cannot see. A fixture failure is about the
    picture -- a notehead the detector missed, a stem it found no ink for -- and
    says nothing about what became of the note. It is easy to read "not detected"
    as "not in the output", and on this fixture that would be wrong: homr writes
    the note and puts it in the wrong voice, because a notehead with no stem has
    nothing to say which line it belongs to.

    A wrong voice is the defect that matters here. Two singers share a staff, and
    a note in the wrong voice is a singer given the wrong line.

    Notes are lined up by when they sound, not by the order they are written in:
    the reference writes one voice out and backs up for the next, while homr
    interleaves them, so document order pairs a tenor with a bass. Within a
    moment they are ordered top to bottom, and what is compared is which of the
    staff's voices a note is in rather than the voice's number -- the two files
    number them differently, and the reference is a male-choir score written an
    octave above where it sounds, so absolute pitch cannot be compared either.
    """
    def read(path: Path) -> dict[tuple[str, int, float], list[dict]]:
        found: dict[tuple[str, int, float], list[dict]] = {}
        printed = 0
        for part in ET.parse(path).getroot().findall("part"):
            staves = max((int(n.text or 1) for n in part.iter("staves")), default=1)
            base, printed = printed, printed + staves
            clefs: dict[int, tuple[str, int, int]] = {}
            for measure in part.findall("measure"):
                clefs.update(_clefs(measure))
                at, previous = 0.0, 0.0
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
                    # A chord member sounds with the note before it, not after.
                    onset = previous if node.find("chord") is not None else at
                    if node.find("chord") is None:
                        previous, at = at, at + length
                    if node.find("rest") is not None:
                        continue
                    pitch = node.find("pitch")
                    if pitch is None:
                        continue
                    staff = base + int(node.findtext("staff", "1"))
                    key = (measure.get("number", "?"), staff, round(onset, 3))
                    found.setdefault(key, []).append({
                        "voice": node.findtext("voice", "1"),
                        "name": f"{pitch.findtext('step')}{pitch.findtext('octave')}",
                        # Where the note sits on the staff, which is what the two
                        # files can be compared on: the reference is a male-choir
                        # score written an octave above where it sounds, so the
                        # pitch names differ by an octave while the printed note
                        # is the same one.
                        "position": _staff_position(
                            pitch,
                            clefs.get(int(node.findtext("staff", "1")), ("G", 2, 0)),
                        ),
                        "height": 7 * int(pitch.findtext("octave", "4"))
                        + STEPS.index(pitch.findtext("step", "C")),
                        "stem": node.findtext("stem", ""),
                        "chord": node.find("chord") is not None,
                    })
        return found

    def ranked(notes: dict[tuple[str, int, float], list[dict]]) -> dict[int, dict[str, int]]:
        """The staff's voices as first, second, ... top line down."""
        order: dict[int, list[str]] = {}
        for (_, staff, _), group in sorted(notes.items()):
            for note in sorted(group, key=lambda n: -n["height"]):
                order.setdefault(staff, [])
                if note["voice"] not in order[staff]:
                    order[staff].append(note["voice"])
        return {staff: {voice: index + 1 for index, voice in enumerate(voices)}
                for staff, voices in order.items()}

    want, got = read(reference), read(parsed)
    here, there = ranked(want), ranked(got)
    rows, wrong = [], {}
    for key in sorted(want, key=lambda k: (int(k[0]), k[1], k[2])):
        bar, staff, onset = key
        mine = sorted(want[key], key=lambda n: -n["height"])
        theirs = sorted(got.get(key, []), key=lambda n: -n["height"])
        if len(mine) != len(theirs):
            rows.append(
                f"<tr class='gap'><td>bar {bar}, staff {staff}, beat "
                f"{onset:g}</td><td colspan='3'>{len(mine)} notehead(s) on the "
                f"page, {len(theirs)} in homr's output here</td></tr>")
            continue
        for a, b in zip(mine, theirs):
            mine_voice = here.get(staff, {}).get(a["voice"], 0)
            their_voice = there.get(staff, {}).get(b["voice"], 0)
            if a["position"] != b["position"]:
                wrong[(bar, staff, a["name"])] = (
                    f"a different note &mdash; {b['name']}, "
                    f"{abs(a['position'] - b['position'])} staff position(s) "
                    f"{'higher' if b['position'] > a['position'] else 'lower'}")
                rows.append(
                    f"<tr class='bad'><td>bar {bar}, staff {staff}, beat {onset:g}</td>"
                    f"<td>{a['name']} &middot; position {a['position']}</td>"
                    f"<td>{b['name']} &middot; position {b['position']}</td>"
                    f"<td class='no'>a different note</td></tr>")
                continue
            same = mine_voice == their_voice
            note = "the voice the page prints" if same else (
                "folded into the other voice as a chord member" if b["chord"]
                else "written in the other voice")
            if not same:
                wrong[(bar, staff, a["name"])] = (
                    f"voice {their_voice} instead of {mine_voice} &mdash; {note}")
            rows.append(
                f"<tr class='{'' if same else 'bad'}'>"
                f"<td>bar {bar}, staff {staff}, beat {onset:g}</td>"
                f"<td>{a['name']} &middot; voice {mine_voice}</td>"
                f"<td>{b['name']} &middot; voice {their_voice}"
                f"{'' if b['stem'] else " <span class='mono'>no stem</span>"}</td>"
                f"<td class='{'ok' if same else 'no'}'>{note}</td></tr>")
    return rows, wrong


def show(head) -> str:
    """One notehead as position and stems, or a dash when there is none."""
    if head is None:
        return "&mdash;"
    stems = "+".join(sorted(head.stems)) or "no stem"
    voice = f" v{'/'.join(sorted(head.voices))}" if head.voices else ""
    return f"{head.position:g} <span class='mono'>{stems}</span>{voice}"


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "laulun-aika-s2"
    manifest = json.loads((FIXTURES / "stem-direction-fixtures.json").read_text())
    entry = manifest["fixtures"][name]
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob(f"{name}-*.png"):
        old.unlink()

    image_path = FIXTURES / entry["image"]
    (OUT / f"{name}-page.png").write_bytes(image_path.read_bytes())
    reference = reference_staffs(FIXTURES / entry["reference"])
    probe = json.loads((FIXTURES / f"{name}.fixture-probe.json").read_text())

    tables, totals = [], {"heads": 0, "agreed": 0, "stem": 0, "missing": 0,
                                "extra": 0, "pitch": 0}
    for index in range(max(len(reference), len(probe["detected"]))):
        notes = reference[index]["notes"] if index < len(reference) else []
        bottom = bottom_line(notes)
        detected = probe["detected"][index] if index < len(probe["detected"]) else []
        left, right = reference_columns(notes), detected_columns(detected)
        rows = []
        for number, (a, b) in enumerate(align_columns(left, right), 1):
            here = left[a] if a is not None else None
            there = right[b] if b is not None else None
            if here is not None and there is not None:
                pairs = _heads_in_column(here, there)
            elif here is not None:
                pairs = [(head, None) for head in here.heads]
            else:
                pairs = [(None, other) for other in there.heads]
            for head, other in pairs:
                totals["heads"] += 1
                heard = spell(other.position, bottom) if other else ""
                if head is None:
                    state, css = "homr saw a note the page does not print", "gap"
                    totals["extra"] += 1
                elif other is None:
                    state, css = "homr did not find this notehead", "gap"
                    totals["missing"] += 1
                elif head.stems != other.stems:
                    state, css = "stem read differently", "bad"
                    totals["stem"] += 1
                elif heard and heard != head.label.split(" ")[-1]:
                    # The DETECTOR's staff position, not the pitch homr writes.
                    # Pitch is the transformer's answer and it reads the image
                    # itself, so it can be right while the detected head sits a
                    # step off -- on this fixture it is. Calling this row "pitch
                    # a step out" said homr had misread a note when homr's
                    # output agreed with the page exactly, which is the same
                    # confusion of layers as calling a note "not detected at
                    # all" when it reaches the score in the wrong voice.
                    state, css = "detected a step off the page", "warn"
                    totals["pitch"] += 1
                else:
                    state, css = "agrees", ""
                    totals["agreed"] += 1
                rows.append(
                    f"<tr class='{css}'><td>{number}</td>"
                    f"<td>{html.escape(head.label) if head else '&mdash;'}</td>"
                    f"<td>{show(head)}</td>"
                    f"<td>{heard or '&mdash;'} &middot; {show(other)}</td>"
                    f"<td class='{'ok' if not css else 'no'}'>{state}</td></tr>")
        tables.append(
            f"<h2>Printed staff {index + 1}</h2>"
            f"<p class='lead'>{len(notes)} noteheads on the page, "
            f"{len(detected)} found by homr.</p>"
            "<table><tr><th>moment</th><th>the page</th><th>page: position, stem</th>"
            "<th>homr: pitch, position, stem</th><th></th></tr>"
            + "".join(rows) + "</table>")

    # The failures again, this time as pixels: the scan around each one, beside
    # the mask the model produced there. A count says a stem is missing; only
    # the mask says whether there was ever ink for it to find.
    engraved = homr_output(image_path, OUT / f"{name}-homr.png")
    parsed = OUT / f"{name}-homr.musicxml"
    rows, wrong = voice_lines(FIXTURES / entry["reference"], parsed) \
        if parsed.exists() else ([], {})

    predictions, _ = load_and_preprocess_predictions(str(image_path), False, False, False)
    symbols = predict_symbols(Quiet(), predictions)
    unit = float(np.median([head.size[1] for head in symbols.noteheads]))
    cards = []
    for finding in findings(name):
        whose, why, how = verdict(finding, predictions, unit)
        picture, mask = f"{name}-{finding['id']}.png", f"{name}-{finding['id']}-mask.png"
        crop(predictions.preprocessed, finding, OUT / picture)
        crop(255 - getattr(predictions, how["mask"]) * 255, finding, OUT / mask,
             marks=False)
        # What became of this note in homr's output -- the thing the finding
        # itself cannot know, and the thing a reader assumes it means.
        label = finding["title"].split(":")[0]
        bar, _, pitch = label.partition(" ")
        became = wrong.get((bar.lstrip("m"), finding["staff"], pitch))
        outcome = (f"<br>in homr's output: <b>{became}</b>" if became else
                   "<br>in homr's output: in the voice the page prints")
        cards.append(
            f"<figure><img src='{picture}' alt='the scan'>"
            f"<img src='{mask}' alt='what the model segmented'>"
            f"<figcaption><span class='id'>{finding['id']}</span> "
            f"<span class='tag {finding['kind']}'>{finding['kind']}</span> "
            f"&nbsp;staff {finding['staff']} &middot; "
            f"<b>{html.escape(finding['title'])}</b><br>"
            f"page: <b>{html.escape(finding['expected'])}</b> &nbsp;&middot;&nbsp; "
            f"homr: <b>{html.escape(finding['detected'])}</b><br>"
            f"<b>{whose}</b> &mdash; {html.escape(why)}{outcome}</figcaption></figure>")

    output = (f"<h2>What homr writes</h2><p class='lead'>The parse itself, "
              f"engraved &mdash; not the detection the tables below check. "
              f"The MusicXML is beside this page as "
              f"<span class='id'>{name}-homr.musicxml</span>.</p>"
              f"<img class='page' src='{engraved}' alt=\"homr's parse\">"
              if engraved else
              "<h2>What homr writes</h2><p class='lead'>(could not be produced)</p>")

    voices = ""
    if rows:
        missed = sum(1 for row in rows if "class='bad'" in row)
        voices = ("<h2>Which voice each note ended up in</h2>"
                  "<p class='lead'>The page against homr's own output. This is the "
                  "half a fixture cannot see: a fixture failure is about the "
                  "picture and says nothing about what became of the note. "
                  f"<b>{missed}</b> notehead(s) are written in the wrong voice.</p>"
                  "<table><tr><th>where</th><th>the page</th>"
                  "<th>homr's output</th><th></th></tr>"
                  + "".join(rows) + "</table>")

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(name)} in full</title><style>{STYLE}</style></head><body>
<h1>{html.escape(name)} &mdash; every notehead</h1>
<p class="lead">One printed system, the reference beside what homr read off it.
The reference is the song's cleaned score imploded back to the printed two-staff
shape, so it says what the page says. Codes like <span class="id">{CODES.get(name, '??')}
-s1-1042p7</span> name a case and do not renumber when others are fixed.</p>
<div class="sum">
  <div><b>{totals['heads']}</b>noteheads compared</div>
  <div><b>{totals['agreed']}</b>agree, position and stem</div>
  <div><b>{totals['stem']}</b>stem read differently</div>
  <div><b>{totals['missing']}</b>not found by homr</div>
  <div><b>{totals['pitch']}</b>detected a step off the page</div>
  <div><b>{totals['extra']}</b>found but not printed</div>
</div>
<h2>The printed system</h2>
<img class="page" src="{name}-page.png" alt="the printed system">
{output}
<h2>Where they disagree</h2>
{''.join(cards)}
{voices}
<h2>Every notehead, as the detector sees it</h2>
<p class="lead">The layer the fixture tests: noteheads and stems found in the
picture. This is <b>not</b> what homr writes, and the two can differ in both
directions &mdash; a note absent here can still reach the output (in the wrong
voice, having no stem to place it), and a head detected a step off the page can
still be written at the right pitch, because the pitch is read from the image by
the transformer rather than taken from this geometry.</p>
{''.join(tables)}
</body></html>"""
    target = OUT / f"{name}.html"
    target.write_text(page)
    print(target)


if __name__ == "__main__":
    main()
