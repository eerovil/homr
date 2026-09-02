#!/usr/bin/env python3
"""Draw every stem-direction fixture failure as a crop of the scan it is on.

Writes ``stem-failures/index.html`` plus one PNG per failure: the band of the
staff around the notehead in question, with the notehead circled, at the
resolution homr actually reads.  Run ``fixture_probe.py`` first -- this reads
the probes it leaves beside the fixtures.
"""

import html
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))

from homr.main import load_and_preprocess_predictions  # noqa: E402
from tests.fixture_matching import (  # noqa: E402
    Column,
    Head,
    detected_columns,
    match,
    reference_columns,
)
from tests.fixture_reference import reference_staffs  # noqa: E402

ROOT = Path(__file__).parent
FIXTURES = ROOT / "fixtures"
OUT = ROOT / "stem-failures"
ZOOM = 3
# Short, stable names to talk about a case by: the fixture, the printed staff,
# and where it is across the scan.  Nothing renumbers when a case is fixed.
CODES = {
    "hanget-soi": "HS",
    "kolme-kakea": "KK",
    "sammon-ryosto": "SR",
    "system4": "S4",
}
PAD_X = 95
PAD_Y = 90


def stems(head: Head) -> str:
    return " + ".join(sorted(head.stems)) or "no stem"


def place(head: Head, pairs: list[tuple[Head | None, Head | None]]) -> tuple[float, float]:
    """Where a reference notehead would be in the scan, from its neighbours."""
    anchors = sorted(
        (reference.x, found.scan) for reference, found in pairs if reference and found and found.scan
    )
    if not anchors:
        return head.x, 0.0
    below = [anchor for anchor in anchors if anchor[0] <= head.x]
    above = [anchor for anchor in anchors if anchor[0] >= head.x]
    left = below[-1] if below else anchors[0]
    right = above[0] if above else anchors[-1]
    if right[0] == left[0]:
        return left[1]
    share = (head.x - left[0]) / (right[0] - left[0])
    return (
        left[1][0] + share * (right[1][0] - left[1][0]),
        left[1][1] + share * (right[1][1] - left[1][1]),
    )


def finding_id(name: str, staff: int, head: Head, at: tuple[float, float]) -> str:
    return f"{CODES.get(name, name)}-s{staff}-{round(at[0])}p{head.position:g}"


def _moment(
    other: Head, columns: list[Column], pairs: list[tuple[Head | None, Head | None]]
) -> tuple[list[Head], list[Head]]:
    """The whole printed moment a detected notehead sits in.

    An "extra" is not a notehead invented out of white paper -- it is one head
    too many in a column that does hold noteheads, usually a clump the splitter
    cut into three where the page prints two.  So a card has to show the column,
    not just the one head, or it reads as a claim nobody can believe.
    """
    def where(head: Head) -> tuple[float, float] | None:
        return head.scan

    column = next(
        (group for group in columns if any(where(head) == where(other) for head in group.heads)),
        None,
    )
    if column is None:
        return [other], []
    inside = {where(head) for head in column.heads}
    printed = [
        reference
        for reference, found in pairs
        if reference is not None and found is not None and where(found) in inside
    ]
    return column.heads, printed


def findings(name: str) -> list[dict]:
    probe = json.loads((FIXTURES / f"{name}.fixture-probe.json").read_text())
    reference = reference_staffs(FIXTURES / f"{name}.musicxml")
    found = []
    for index in range(max(len(reference), len(probe["detected"]))):
        notes = reference[index]["notes"] if index < len(reference) else []
        detected = probe["detected"][index] if index < len(probe["detected"]) else []
        columns = detected_columns(detected)
        pairs = match(reference_columns(notes), columns)
        for head, other in pairs:
            if head is not None and other is not None and head.stems == other.stems:
                continue
            staff = index + 1
            if head is None:
                assert other is not None and other.scan is not None
                read, printed = _moment(other, columns, pairs)
                names = ", ".join(item.label.split(" ")[-1] for item in printed) or "nothing"
                found.append(
                    {
                        "kind": "extra",
                        "title": f"one notehead too many at staff position {other.position:g}",
                        "expected": f"{len(printed)} notehead(s) here ({names})",
                        "detected": f"{len(read)}, this one included",
                        "at": other.scan,
                        "others": [item.scan for item in read if item is not other],
                        "staff": staff,
                        "id": finding_id(name, staff, other, other.scan),
                    }
                )
            elif other is None:
                at = place(head, pairs)
                found.append(
                    {
                        "kind": "missing",
                        "title": f"{head.label} is not detected at all",
                        "expected": f"a notehead with {stems(head)}",
                        "detected": "nothing here",
                        "at": at,
                        "others": [],
                        "staff": staff,
                        "id": finding_id(name, staff, head, at),
                    }
                )
            else:
                assert other.scan is not None
                found.append(
                    {
                        "kind": "wrong",
                        "title": f"{head.label}",
                        "expected": stems(head),
                        "detected": stems(other),
                        "at": other.scan,
                        "others": [],
                        "staff": staff,
                        "id": finding_id(name, staff, head, other.scan),
                    }
                )
    return found


def crop(image, finding: dict, path: Path) -> None:
    """The scan around one case: the head in question red, its neighbours blue."""
    x, y = int(finding["at"][0]), int(finding["at"][1])
    top, bottom = max(0, y - PAD_Y), min(image.shape[0], y + PAD_Y)
    left, right = max(0, x - PAD_X), min(image.shape[1], x + PAD_X)
    band = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_GRAY2BGR)
    band = cv2.resize(band, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
    for other in finding["others"]:
        if other is None:
            continue
        cv2.circle(
            band,
            ((int(other[0]) - left) * ZOOM, (int(other[1]) - top) * ZOOM),
            8 * ZOOM,
            (200, 120, 0),
            3,
        )
    cv2.circle(band, ((x - left) * ZOOM, (y - top) * ZOOM), 8 * ZOOM, (0, 0, 220), 3)
    cv2.imwrite(str(path), band)


STYLE = """
body { font: 15px/1.5 system-ui, sans-serif; margin: 0 auto; max-width: 1100px;
       padding: 24px; color: #1a1a1a; }
h1 { font-size: 22px; margin-bottom: 4px; }
h2 { font-size: 18px; margin: 32px 0 4px; }
p.lead { color: #555; margin-top: 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 18px; }
figure { margin: 0; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }
figure img { display: block; width: 100%; background: #fff; }
figcaption { padding: 10px 12px; font-size: 13px; border-top: 1px solid #eee; }
.tag { display: inline-block; font-size: 11px; letter-spacing: .04em; text-transform: uppercase;
       padding: 2px 7px; border-radius: 999px; margin-bottom: 6px; }
.wrong { background: #fde2e2; color: #8a1c1c; }
.missing { background: #fdefd8; color: #8a5a10; }
.extra { background: #e4e9fb; color: #26379a; }
a.id { font: 600 12px ui-monospace, SFMono-Regular, Menlo, monospace; color: #0b6; 
       text-decoration: none; margin-right: 8px; }
a.id:hover { text-decoration: underline; }
figure:target { outline: 3px solid #0b6; }
.what { color: #555; }
.what b { color: #1a1a1a; font-weight: 600; }
"""


def stamp_manifest(manifest: dict) -> None:
    """Put each case's name on its entry in the manifest, so the two agree."""
    for name, entry in manifest["fixtures"].items():
        waiting = {kind: [] for kind in ("wrong", "missing", "extra")}
        for finding in findings(name):
            waiting[finding["kind"]].append(finding["id"])
        for gap in entry.get("known_gaps", []):
            kind = (
                "missing"
                if "no detected notehead" in gap["failure"]
                else "extra"
                if "extra notehead" in gap["failure"]
                else "wrong"
            )
            gap["id"] = waiting[kind].pop(0)
    path = FIXTURES / "stem-direction-fixtures.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    manifest = json.loads((FIXTURES / "stem-direction-fixtures.json").read_text())
    counts = {"wrong": 0, "missing": 0, "extra": 0}
    sections = []
    for name in sorted(manifest["fixtures"]):
        image_path = FIXTURES / manifest["fixtures"][name]["image"]
        predictions, _ = load_and_preprocess_predictions(str(image_path), False, False, False)
        image = predictions.preprocessed
        cards = []
        for finding in findings(name):
            counts[finding["kind"]] += 1
            picture = f"{finding['id']}.png"
            crop(image, finding, OUT / picture)
            cards.append(
                f"""<figure id="{finding['id']}">
  <img src="{picture}" alt="">
  <figcaption>
    <a class="id" href="#{finding['id']}">{finding['id']}</a>
    <span class="tag {finding['kind']}">{finding['kind']}</span><br>
    <b>staff {finding['staff']} &middot; {html.escape(finding['title'])}</b><br>
    <span class="what">page: <b>{html.escape(finding['expected'])}</b><br>
    homr: <b>{html.escape(finding['detected'])}</b></span>
  </figcaption>
</figure>"""
            )
        if cards:
            sections.append(
                f"<h2>{html.escape(name)}</h2>"
                f"<div class=\"grid\">{''.join(cards)}</div>"
            )
    total = sum(counts.values())
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stem-direction fixture failures</title><style>{STYLE}</style></head><body>
<h1>Stem-direction fixture failures</h1>
<p class="lead">{total} cases where homr's detected noteheads disagree with the
hand-corrected score: {counts['wrong']} with the wrong stems,
{counts['missing']} noteheads never found, {counts['extra']} found that the page
does not print. Each picture is the scan as homr reads it, with the notehead in
question circled in red -- and, where homr read more heads in that moment than
the page prints, the rest of them in blue. Each case has a fixed name -- fixture, printed staff, and
where it sits across the scan -- so nothing is renumbered when one is fixed.</p>
{''.join(sections)}
</body></html>"""
    (OUT / "index.html").write_text(page)
    stamp_manifest(manifest)
    print(OUT / "index.html", total, counts)
    for name in sorted(manifest["fixtures"]):
        for finding in findings(name):
            print(f"  {finding['id']:<14} {finding['kind']:<8} {finding['title']}")


if __name__ == "__main__":
    main()
