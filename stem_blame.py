#!/usr/bin/env python3
"""For every fixture case, ask whether the model saw it or the code lost it.

homr is a segmentation model followed by a lot of plain geometry.  A failure is
the model's when its own masks have nothing to work with at that place, and the
code's when the ink is in the mask and something after it threw the answer away.
This looks at the masks directly and says which.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from homr.main import load_and_preprocess_predictions, predict_symbols  # noqa: E402


class Quiet:
    def write_threshold_image(self, *args: object) -> None:
        pass

    def write_bounding_boxes(self, *args: object) -> None:
        pass


def blob_at(mask, x: int, y: int, reach: int = 6):
    """The connected piece of ink nearest a point, and its size."""
    window = mask[max(0, y - reach) : y + reach, max(0, x - reach) : x + reach]
    if not window.any():
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    ys, xs = np.nonzero(window)
    label = labels[max(0, y - reach) + ys[0], max(0, x - reach) + xs[0]]
    left, top, width, height, area = stats[label]
    return {"box": (int(left), int(top), int(width), int(height)), "area": int(area)}


def ink_beside(mask, x: int, y: int, unit: float, side: int) -> int:
    """The tallest run of ink in a column beside a notehead, on one side."""
    columns = range(int(x + side * unit * 0.3), int(x + side * unit * 1.1), 1 if side > 0 else -1)
    rows = range(max(0, int(y - unit * 3)), min(mask.shape[0], int(y + unit * 3)))
    best = 0
    for column in columns:
        if not 0 <= column < mask.shape[1]:
            continue
        values = (mask[list(rows), column] > 0).astype(int)
        changes = np.diff(np.pad(values, (1, 1)))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        for start, end in zip(starts, ends, strict=True):
            best = max(best, int(end - start))
    return best


def verdict(finding: dict, predictions, unit: float) -> tuple[str, str, dict]:
    """Whose failure this is, in a word, a sentence, and the numbers behind it.

    Returns the mask to show it against as well: a notehead question is
    answered by the notehead mask, a stem question by the stems mask.
    """
    x, y = int(finding["at"][0]), int(finding["at"][1])
    if finding["kind"] == "extra":
        blob = blob_at(predictions.notehead, x, y)
        inside = 1
        if blob is not None:
            left, top, width, height = blob["box"]
            inside += sum(
                1
                for other in finding["others"]
                if other
                and left <= other[0] <= left + width
                and top <= other[1] <= top + height
            )
        if blob and blob["box"][3] < unit * inside * 0.8:
            return (
                "code",
                f"the mask has one piece of ink {blob['box'][3]}px tall here, about"
                f" {blob['box'][3] / unit:.1f} noteheads' worth; the splitter cut"
                f" {inside} out of it",
                {"mask": "notehead"},
            )
        return ("model", "the mask really is that big here", {"mask": "notehead"})
    if finding["kind"] == "missing":
        blob = blob_at(predictions.notehead, x, y, reach=int(unit * 0.7))
        # A notehead is wider than it is tall, so half of one is half of each
        # -- a tall thin sliver of an outline is not a head the code could use.
        wide, tall = (blob["box"][2:] if blob else (0, 0))
        if blob is not None and wide >= unit * 1.286 * 1.5:
            # Ink far too wide for one head, and nothing in its shape says
            # where the join is: the model has run several heads together.
            return (
                "model",
                f"the mask runs this head into its neighbours as one"
                f" {wide}x{tall}px blob with no join in it",
                {"mask": "notehead"},
            )
        if blob is not None and wide >= unit * 1.286 * 0.5 and tall >= unit * 0.5:
            return (
                "code",
                f"a notehead-sized piece of ink ({blob['box'][2]}x{blob['box'][3]}px) is"
                " in the mask; something after the model dropped it",
                {"mask": "notehead"},
            )
        size = f"{blob['box'][2]}x{blob['box'][3]}px" if blob else "nothing"
        return (
            "model",
            f"the notehead mask holds {size} here, against a {unit:.0f}px notehead",
            {"mask": "notehead"},
        )
    left = ink_beside(predictions.stems_rest, x, y, unit, -1)
    right = ink_beside(predictions.stems_rest, x, y, unit, +1)
    wants_up = "up" in finding["expected"] and "up" not in finding["detected"]
    side, length = ("right", right) if wants_up else ("left", left)
    if length >= unit * 0.5:
        return (
            "code",
            f"the stems mask has a {length}px run on the {side} of this head;"
            " the geometry refused it",
            {"mask": "stems_rest"},
        )
    return (
        "model",
        f"the stems mask has only {length}px on the {side} of this head,"
        f" against a {unit:.0f}px notehead",
        {"mask": "stems_rest"},
    )


def main() -> None:
    from stem_failure_report import FIXTURES, findings

    manifest = json.loads((FIXTURES / "stem-direction-fixtures.json").read_text())
    counts: dict[str, int] = {}
    for name in sorted(manifest["fixtures"]):
        image = FIXTURES / manifest["fixtures"][name]["image"]
        predictions, _ = load_and_preprocess_predictions(str(image), False, False, False)
        symbols = predict_symbols(Quiet(), predictions)
        unit = float(np.median([notehead.size[1] for notehead in symbols.noteheads]))
        for finding in findings(name):
            whose, why, _ = verdict(finding, predictions, unit)
            counts[whose] = counts.get(whose, 0) + 1
            print(f"{finding['id']:<15} {finding['kind']:<8} {whose:<6} {why}")
    print(counts)


if __name__ == "__main__":
    main()
