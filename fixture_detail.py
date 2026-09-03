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
from tests.fixture_reference import STEPS, reference_staffs  # noqa: E402

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
                    # Not a fixture failure: heads pair within
                    # MAX_POSITION_ERROR, which is the slack the stem check
                    # needs to survive a notehead sitting a little high in a
                    # scan. It does mean a pitch read one step out passes a
                    # stem-direction fixture in silence, so it is said here.
                    state, css = "stem right, pitch a step out", "warn"
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
        cards.append(
            f"<figure><img src='{picture}' alt='the scan'>"
            f"<img src='{mask}' alt='what the model segmented'>"
            f"<figcaption><span class='id'>{finding['id']}</span> "
            f"<span class='tag {finding['kind']}'>{finding['kind']}</span> "
            f"&nbsp;staff {finding['staff']} &middot; "
            f"<b>{html.escape(finding['title'])}</b><br>"
            f"page: <b>{html.escape(finding['expected'])}</b> &nbsp;&middot;&nbsp; "
            f"homr: <b>{html.escape(finding['detected'])}</b><br>"
            f"<b>{whose}</b> &mdash; {html.escape(why)}</figcaption></figure>")

    engraved = homr_output(image_path, OUT / f"{name}-homr.png")
    output = (f"<h2>What homr writes</h2><p class='lead'>The parse itself, "
              f"engraved &mdash; not the detection the tables below check. "
              f"The MusicXML is beside this page as "
              f"<span class='id'>{name}-homr.musicxml</span>.</p>"
              f"<img class='page' src='{engraved}' alt=\"homr's parse\">"
              if engraved else
              "<h2>What homr writes</h2><p class='lead'>(could not be produced)</p>")

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
  <div><b>{totals['pitch']}</b>stem right, pitch a step out</div>
  <div><b>{totals['extra']}</b>found but not printed</div>
</div>
<h2>The printed system</h2>
<img class="page" src="{name}-page.png" alt="the printed system">
{output}
<h2>Where they disagree</h2>
{''.join(cards)}
{''.join(tables)}
</body></html>"""
    target = OUT / f"{name}.html"
    target.write_text(page)
    print(target)


if __name__ == "__main__":
    main()
