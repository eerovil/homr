#!/usr/bin/env python3
"""Record what the notehead detector finds, so two versions can be compared.

A change to note detection is a change to every parse, and five fixtures are not
evidence about that. This writes down every detected notehead of every image
given -- staff, position, stem, size -- so the same images can be run before and
after a change and the difference read off rather than hoped about.

    .venv/bin/python detection_diff.py out.json <image> [<image> ...]
    .venv/bin/python detection_diff.py --compare before.json after.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def record(images: list[str]) -> dict:
    from tests.test_stem_direction_fixtures import detect

    found = {}
    for image in images:
        try:
            staffs = detect(Path(image))
        except Exception as error:  # a page homr cannot read is a fact too
            found[image] = {"error": str(error)[:200]}
            continue
        found[image] = {
            "staffs": [
                sorted(
                    (round(note["x"], 1), round(note["position"], 1),
                     "+".join(sorted(note["stems"])) or "none",
                     round(note["w"], 1), round(note.get("h", 0), 1))
                    for note in staff
                )
                for staff in staffs
            ]
        }
    return found


def compare(before: dict, after: dict) -> int:
    changed = 0
    for image in sorted(set(before) | set(after)):
        old, new = before.get(image, {}), after.get(image, {})
        if "error" in old or "error" in new:
            print(f"{image}: {old.get('error', 'ok')} -> {new.get('error', 'ok')}")
            changed += 1
            continue
        for index, (a, b) in enumerate(zip(old["staffs"], new["staffs"]), 1):
            gone = [note for note in a if note not in b]
            fresh = [note for note in b if note not in a]
            if not gone and not fresh:
                continue
            changed += 1
            print(f"{Path(image).name} staff {index}: "
                  f"{len(a)} -> {len(b)} noteheads")
            for note in gone:
                print(f"    lost  x={note[0]:7} position={note[1]:5} "
                      f"stem={note[2]:5} size=({note[3]}, {note[4]})")
            for note in fresh:
                print(f"    new   x={note[0]:7} position={note[1]:5} "
                      f"stem={note[2]:5} size=({note[3]}, {note[4]})")
        if len(old.get("staffs", [])) != len(new.get("staffs", [])):
            print(f"{image}: {len(old.get('staffs', []))} -> "
                  f"{len(new.get('staffs', []))} staffs")
            changed += 1
    print(f"\n{changed} staff(s) changed out of "
          f"{sum(len(v.get('staffs', [])) for v in before.values())}")
    return changed


def main() -> None:
    if sys.argv[1] == "--compare":
        before = json.loads(Path(sys.argv[2]).read_text())
        after = json.loads(Path(sys.argv[3]).read_text())
        compare(before, after)
        return
    out, images = sys.argv[1], sys.argv[2:]
    Path(out).write_text(json.dumps(record(images), indent=1))
    print(out)


if __name__ == "__main__":
    main()
