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


def findings(name: str) -> list[dict]:
    probe = json.loads((FIXTURES / f"{name}.fixture-probe.json").read_text())
    reference = reference_staffs(FIXTURES / f"{name}.musicxml")
    found = []
    for index in range(max(len(reference), len(probe["detected"]))):
        notes = reference[index]["notes"] if index < len(reference) else []
        detected = probe["detected"][index] if index < len(probe["detected"]) else []
        pairs = match(reference_columns(notes), detected_columns(detected))
        for head, other in pairs:
            if head is not None and other is not None and head.stems == other.stems:
                continue
            if head is None:
                assert other is not None and other.scan is not None
                found.append(
                    {
                        "kind": "extra",
                        "title": f"a notehead the page does not print, at staff position {other.position:g}",
                        "expected": "nothing",
                        "detected": stems(other),
                        "at": other.scan,
                        "staff": index + 1,
                    }
                )
            elif other is None:
                found.append(
                    {
                        "kind": "missing",
                        "title": f"{head.label} was not detected at all",
                        "expected": stems(head),
                        "detected": "no notehead",
                        "at": place(head, pairs),
                        "staff": index + 1,
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
                        "staff": index + 1,
                    }
                )
    return found


def crop(image, at: tuple[float, float], path: Path) -> None:
    x, y = int(at[0]), int(at[1])
    top, bottom = max(0, y - PAD_Y), min(image.shape[0], y + PAD_Y)
    left, right = max(0, x - PAD_X), min(image.shape[1], x + PAD_X)
    band = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_GRAY2BGR)
    band = cv2.resize(band, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
    cv2.circle(band, ((x - left) * ZOOM, (y - top) * ZOOM), 16 * ZOOM // 2, (0, 0, 220), 3)
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
.what { color: #555; }
.what b { color: #1a1a1a; font-weight: 600; }
"""


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
        for number, finding in enumerate(findings(name), start=1):
            counts[finding["kind"]] += 1
            picture = f"{name}-{number}.png"
            crop(image, finding["at"], OUT / picture)
            cards.append(
                f"""<figure>
  <img src="{picture}" alt="">
  <figcaption>
    <span class="tag {finding['kind']}">{finding['kind']}</span><br>
    <b>staff {finding['staff']} &middot; {html.escape(finding['title'])}</b><br>
    <span class="what">page says <b>{html.escape(finding['expected'])}</b>,
    homr reads <b>{html.escape(finding['detected'])}</b></span>
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
question circled.</p>
{''.join(sections)}
</body></html>"""
    (OUT / "index.html").write_text(page)
    print(OUT / "index.html", total, counts)


if __name__ == "__main__":
    main()
