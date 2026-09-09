from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

from homr.main import ProcessingConfig, process_image
from homr.music_xml_generator import XmlGeneratorArguments
from homr.system_crops import SystemBounds, render_system_crops

DPI = 200
PAD = 0.02
CASES = {"B1a", "B1b", "B2"}
CHOIR = Path(os.environ["CHOIR_REPO"])
OUT = Path(os.environ["BENCHMARK_OUT"])
OUT.mkdir(parents=True, exist_ok=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pdf_size_px(path: Path) -> tuple[int, int]:
    result = subprocess.run(["pdfinfo", str(path)], check=True, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("Page size:"):
            parts = line.split()
            width_pt, height_pt = float(parts[2]), float(parts[4])
            return math.ceil(width_pt * DPI / 72), math.ceil(height_pt * DPI / 72)
    raise RuntimeError(f"Could not read page size from {path}")


def poppler_crop(pdf: Path, bound: SystemBounds, destination: Path) -> None:
    width, height = pdf_size_px(pdf)
    top_fraction = max(0.0, bound.top - PAD)
    bottom_fraction = min(1.0, bound.bottom + PAD)
    top = max(0, min(height - 1, int(height * top_fraction)))
    band = max(1, min(height - top, int(height * bottom_fraction) - top))
    stem = destination.with_suffix("")
    subprocess.run(
        [
            "pdftoppm",
            "-r",
            str(DPI),
            "-f",
            str(bound.page),
            "-l",
            str(bound.page),
            "-x",
            "0",
            "-y",
            str(top),
            "-W",
            str(width),
            "-H",
            str(band),
            "-png",
            "-singlefile",
            str(pdf),
            str(stem),
        ],
        check=True,
        capture_output=True,
    )


def staff_numbers(part: ET.Element) -> list[int]:
    declared = 0
    for attrs in part.iter("attributes"):
        text = attrs.findtext("staves")
        if text and text.strip().isdigit():
            declared = max(declared, int(text.strip()))
    used = {
        int(node.text.strip())
        for node in part.iter("staff")
        if node.text and node.text.strip().isdigit()
    }
    return sorted(set(range(1, declared + 1)) | used) or [1]


def event_rows(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    rows: list[str] = []
    for part_index, part in enumerate(root.findall("part"), start=1):
        for measure_index, measure in enumerate(part.findall("measure"), start=1):
            cursor = 0
            onset = 0
            for child in measure:
                if child.tag == "backup":
                    cursor -= int(child.findtext("duration") or 0)
                elif child.tag == "forward":
                    cursor += int(child.findtext("duration") or 0)
                elif child.tag == "note":
                    chord = child.find("chord") is not None
                    if not chord:
                        onset = cursor
                    duration = int(child.findtext("duration") or 0)
                    staff = child.findtext("staff") or "1"
                    voice = child.findtext("voice") or "1"
                    if child.find("rest") is not None:
                        pitch = "rest"
                    else:
                        pitch_node = child.find("pitch")
                        if pitch_node is None:
                            pitch = "unpitched"
                        else:
                            pitch = ":".join(
                                [
                                    pitch_node.findtext("step") or "",
                                    pitch_node.findtext("alter") or "0",
                                    pitch_node.findtext("octave") or "",
                                ]
                            )
                    rows.append(
                        f"p{part_index}/m{measure_index}/o{onset}/s{staff}/v{voice}/"
                        f"{'chord' if chord else 'event'}/{pitch}/d{duration}"
                    )
                    if not chord:
                        cursor += duration
    return rows


def summary(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    parts = root.findall("part")
    rows = event_rows(path)
    return {
        "parts": len(parts),
        "staves": sum(len(staff_numbers(part)) for part in parts),
        "bars": max((len(part.findall("measure")) for part in parts), default=0),
        "events": len(rows),
        "event_sha256": hashlib.sha256("\n".join(rows).encode()).hexdigest(),
    }


def first_event_delta(old: Path, new: Path) -> list[str]:
    a, b = event_rows(old), event_rows(new)
    return list(difflib.unified_diff(a, b, fromfile="poppler", tofile="pdfium", n=2))[:20]


def recognize(path: Path) -> Path:
    config = ProcessingConfig(
        enable_debug=False,
        enable_cache=False,
        write_staff_positions=False,
        write_confidence=False,
        score_settings=None,
        read_staff_positions=False,
        selected_staff=-1,
        transformer_use_gpu=False,
        segnet_use_gpu=False,
        coreml_encoder=False,
        title_detection=False,
    )
    process_image(str(path), config, XmlGeneratorArguments())
    return path.with_suffix(".musicxml")


def main() -> None:
    manifest = json.loads((CHOIR / "fixtures/omr-benchmark/pages.json").read_text())
    selected = [page for page in manifest["pages"] if page["id"] in CASES]
    assert {page["id"] for page in selected} == CASES
    results: list[dict[str, object]] = []
    failures: list[str] = []

    for page in selected:
        pdf = CHOIR / page["pdf"]
        for item in page["systems"]:
            bound = SystemBounds(
                index=int(item["index"]),
                page=int(item["page"]),
                top=float(item["top"]),
                bottom=float(item["bottom"]),
                measure_start=int(item.get("measure_start", 0)),
                measure_end=int(item.get("measure_end", 0)),
            )
            case = f"{page['id']}-s{bound.index}"
            case_dir = OUT / case
            old_dir, new_dir = case_dir / "poppler", case_dir / "pdfium"
            old_dir.mkdir(parents=True, exist_ok=True)
            new_dir.mkdir(parents=True, exist_ok=True)
            old_png = old_dir / f"{case}.png"
            poppler_crop(pdf, bound, old_png)
            new_png = Path(
                render_system_crops(str(pdf), [bound], str(new_dir), dpi=DPI, pad=PAD)[0].path
            )

            old_image, new_image = cv2.imread(str(old_png)), cv2.imread(str(new_png))
            if old_image is None or new_image is None:
                raise RuntimeError(f"Could not read crop for {case}")

            old_xml = recognize(old_png)
            new_xml = recognize(new_png)
            old_summary, new_summary = summary(old_xml), summary(new_xml)
            structural = (
                old_summary["staves"] == new_summary["staves"]
                and old_summary["bars"] == new_summary["bars"]
            )
            if not structural:
                failures.append(case)
            result = {
                "case": case,
                "bound": item,
                "dpi": DPI,
                "pad": PAD,
                "poppler_crop": {
                    "shape": list(old_image.shape),
                    "sha256": sha(old_png),
                    "musicxml": old_summary,
                },
                "pdfium_crop": {
                    "shape": list(new_image.shape),
                    "sha256": sha(new_png),
                    "musicxml": new_summary,
                },
                "same_crop_bytes": sha(old_png) == sha(new_png),
                "same_event_stream": old_summary["event_sha256"] == new_summary["event_sha256"],
                "structural_match": structural,
                "first_event_delta": first_event_delta(old_xml, new_xml),
            }
            results.append(result)
            print(
                case,
                "old",
                old_summary,
                "new",
                new_summary,
                "structural",
                structural,
            )

    report = {
        "head": os.environ.get("GITHUB_SHA", ""),
        "dpi": DPI,
        "pad": PAD,
        "cases": results,
        "structural_failures": failures,
        "event_identical": sum(1 for result in results if result["same_event_stream"]),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit("renderer migration changed staff/bar structure: " + ", ".join(failures))


if __name__ == "__main__":
    main()
