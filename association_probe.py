#!/usr/bin/env python3
"""Explore geometry-only association of decoder notes to detected noteheads."""

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path


def cluster(notes, threshold):
    columns = []
    for note in sorted(notes, key=lambda item: item["x"]):
        if not columns or note["x"] - columns[-1][-1]["x"] > threshold:
            columns.append([note])
        else:
            columns[-1].append(note)
    return columns


def center(column):
    return sum(note["x"] for note in column) / len(column)


def expected_direction(note):
    # Temporary system-4 oracle, only to judge association quality.  The production
    # algorithm must not contain it.
    y = note["attention"][1]
    return "up" if y < 75 or 145 <= y < 185 else "down"


def main(path: Path):
    payload = json.loads(path.read_text())
    staff = payload["staffs"][0]
    detected = staff["detector_notes"]
    decoded = staff["decoded_notes"]
    detector_columns = cluster(detected, 12)
    decoded_columns = cluster(
        [{**note, "x": note["attention"][0], "y": note["attention"][1]} for note in decoded], 14
    )

    print("detector columns", [(round(center(column), 1), len(column)) for column in detector_columns])
    print("decoded columns", [(round(center(column), 1), len(column)) for column in decoded_columns])

    # Attention is on an encoder grid, not directly on detector pixels.  Recover
    # an affine x calibration from pitch-height-compatible pairs, without using
    # any ground-truth stems.  A printed notehead may validly serve multiple
    # decoded notes, so no pair is consumed.
    candidates = [
        (head, note)
        for head in detected
        for note in decoded
        if abs(head["y"] - note["attention"][1]) <= 10
    ]
    best = None
    for first, second in combinations(candidates, 2):
        hx1, dx1 = first[0]["x"], first[1]["attention"][0]
        hx2, dx2 = second[0]["x"], second[1]["attention"][0]
        if abs(hx2 - hx1) < 20:
            continue
        scale = (dx2 - dx1) / (hx2 - hx1)
        if not 0.8 <= scale <= 1.3:
            continue
        offset = dx1 - scale * hx1
        inliers = [
            (head, note)
            for head, note in candidates
            if abs(scale * head["x"] + offset - note["attention"][0]) <= 8
        ]
        score = (len(inliers), -sum(abs(scale * head["x"] + offset - note["attention"][0]) for head, note in inliers))
        if best is None or score > best[0]:
            best = score, scale, offset, inliers
    assert best is not None
    _, scale, offset, inliers = best
    print("geometry-only calibration", round(scale, 4), round(offset, 2), "inliers", len(inliers))

    results = []
    for note in decoded:
        x, y = note["attention"]
        # Prefer the nearest x after offset; then the closest staff height.  This
        # permits a single printed head to serve multiple decoded notes.
        candidates = sorted(
            detected,
            key=lambda head: ((scale * head["x"] + offset - x) ** 2 + (head["y"] - y) ** 2),
        )
        head = candidates[0]
        x_error = abs((scale * head["x"] + offset) - x)
        y_error = abs(head["y"] - y)
        directions = head["stem_directions"]
        expected = expected_direction(note)
        result = "covered" if expected in directions else "unknown" if not directions else "wrong"
        results.append(
            {
                "token": note["token"],
                "pitch": note["pitch"],
                "expected": expected,
                "directions": directions,
                "result": result,
                "head": [head["x"], head["y"]],
                "x_error": round(x_error, 1),
                "y_error": round(y_error, 1),
            }
        )

    print("results", Counter(item["result"] for item in results))
    for item in results:
        if item["result"] != "covered" or item["x_error"] > 10 or item["y_error"] > 12:
            print(item)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
