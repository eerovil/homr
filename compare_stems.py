#!/usr/bin/env python3
"""Compare two detection_diff records head by head, counting stem changes.

Heads are matched within a staff by staff position and nearest x (within
``TOLERANCE`` pixels), so a head that moved a pixel is the same head and only
its stem is asked about.

    .venv/bin/python compare_stems.py before.json after.json
"""

import json
import sys
from pathlib import Path

TOLERANCE = 6.0


def staffs(record: dict) -> dict:
    out = {}
    for image, data in record.items():
        for index, staff in enumerate(data.get("staffs", [])):
            out[(Path(image).name, index)] = [
                (float(x), int(position), stem) for x, position, stem, _w, _h in staff
            ]
    return out


def main() -> None:
    before, after = staffs(json.loads(Path(sys.argv[1]).read_text())), staffs(
        json.loads(Path(sys.argv[2]).read_text())
    )
    lost: list = []
    gained: list = []
    changed: list = []
    total_before = total_after = 0
    for key in sorted(set(before) | set(after)):
        old, new = list(before.get(key, [])), list(after.get(key, []))
        total_before += len(old)
        total_after += len(new)
        free = list(new)
        for x, position, stem in old:
            near = [
                item
                for item in free
                if item[1] == position and abs(item[0] - x) <= TOLERANCE
            ]
            if not near:
                lost.append((key, x, position, stem))
                continue
            match = min(near, key=lambda item: abs(item[0] - x))
            free.remove(match)
            if match[2] != stem:
                changed.append((key, x, position, stem, match[2]))
        gained.extend((key, *item) for item in free)

    print(f"heads: {total_before} -> {total_after}   lost {len(lost)}  gained {len(gained)}")
    stem_gained = [item for item in changed if item[3] == "none"]
    stem_lost = [item for item in changed if item[4] == "none"]
    other = [item for item in changed if item[3] != "none" and item[4] != "none"]
    print(
        f"stems: gained {len(stem_gained)}, lost {len(stem_lost)},"
        f" changed on a head that already had one {len(other)}"
        f"  (total heads whose stem moved: {len(changed)})"
    )
    kinds: dict[str, int] = {}
    for item in changed:
        kinds[f"{item[3]} -> {item[4]}"] = kinds.get(f"{item[3]} -> {item[4]}", 0) + 1
    for kind, count in sorted(kinds.items(), key=lambda item: -item[1]):
        print(f"    {kind}: {count}")
    for item in other:
        print(f"    ! {item[0][0]} staff {item[0][1]} x={item[1]} pos={item[2]}"
              f" {item[3]} -> {item[4]}")
    for name, items in (("lost heads", lost), ("gained heads", gained)):
        print(f"{name}: {len(items)}")
        for item in items[:15]:
            print(f"    {item}")


if __name__ == "__main__":
    main()
