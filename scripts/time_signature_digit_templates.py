"""Write `homr/time_signature_digits.json`: the printed digits a time signature is read against.

A time signature's digits are set in the score's music font, not a text font, so
the shapes are taken from three SMuFL fonts with different design traditions --
Bravura (Steinberg), Leipzig (Verovio's default) and Gootville (Gonville-like) --
whose glyph outlines Verovio ships beside its fonts. All three are under the SIL
Open Font License.

Usage (needs a Verovio install for its data directory):

    python scripts/time_signature_digit_templates.py <verovio>/data

Each glyph is drawn at `HEIGHT` rows, cropped to its ink and thresholded, and kept
as rows of '#' and '.', so the committed file can be read and diffed. The outlines
are filled here rather than through an SVG renderer: cairosvg loops forever on
these path strings.
"""

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

HEIGHT = 32
FONTS = ("Bravura", "Leipzig", "Gootville")
OUT = Path(__file__).resolve().parent.parent / "homr" / "time_signature_digits.json"


_TOKEN = re.compile(r"[MmLlHhVvCcSsQqTtZz]|-?(?:\d+\.?\d*|\.\d+)(?:e-?\d+)?")
_ARGS = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "Z": 0}


def _outline(path: str) -> list[np.ndarray]:
    """The closed polygons of an SVG path, curves flattened."""
    tokens = _TOKEN.findall(path)
    polygons: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    x = y = 0.0
    control: tuple[float, float] | None = None
    command = "M"
    i = 0

    def curve(points: list[tuple[float, float]]) -> None:
        for t in np.linspace(0, 1, 16)[1:]:
            if len(points) == 4:  # noqa: PLR2004
                (x0, y0), (x1, y1), (x2, y2), (x3, y3) = points
                u = 1 - t
                current.append(
                    (
                        u**3 * x0 + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t**3 * x3,
                        u**3 * y0 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t**3 * y3,
                    )
                )
            else:
                (x0, y0), (x1, y1), (x2, y2) = points
                u = 1 - t
                current.append(
                    (
                        u * u * x0 + 2 * u * t * x1 + t * t * x2,
                        u * u * y0 + 2 * u * t * y1 + t * t * y2,
                    )
                )

    while i < len(tokens):
        if tokens[i].isalpha():
            command = tokens[i]
            i += 1
        upper = command.upper()
        rel = command.islower()
        n = _ARGS[upper]
        if upper == "Z":
            if current:
                polygons.append(current)
            current = []
            continue
        args = [float(v) for v in tokens[i : i + n]]
        i += n
        ox, oy = (x, y) if rel else (0.0, 0.0)
        if upper == "M":
            if current:
                polygons.append(current)
            x, y = args[0] + ox, args[1] + oy
            current = [(x, y)]
            command = "l" if rel else "L"
            control = None
        elif upper == "L":
            x, y = args[0] + ox, args[1] + oy
            current.append((x, y))
            control = None
        elif upper == "H":
            x = args[0] + (x if rel else 0.0)
            current.append((x, y))
            control = None
        elif upper == "V":
            y = args[0] + (y if rel else 0.0)
            current.append((x, y))
            control = None
        elif upper in "CS":
            if upper == "C":
                c1 = (args[0] + ox, args[1] + oy)
                rest = args[2:]
            else:
                c1 = (2 * x - control[0], 2 * y - control[1]) if control else (x, y)
                rest = args
            c2 = (rest[0] + ox, rest[1] + oy)
            end = (rest[2] + ox, rest[3] + oy)
            curve([(x, y), c1, c2, end])
            control = c2
            x, y = end
        else:  # Q, T
            if upper == "Q":
                c1 = (args[0] + ox, args[1] + oy)
                end = (args[2] + ox, args[3] + oy)
            else:
                c1 = (2 * x - control[0], 2 * y - control[1]) if control else (x, y)
                end = (args[0] + ox, args[1] + oy)
            curve([(x, y), c1, end])
            control = c1
            x, y = end
    if current:
        polygons.append(current)
    return [np.array(polygon) for polygon in polygons]


def _glyph(data: Path, font: str, digit: int) -> list[str]:
    code = f"E08{digit}"
    match = re.search(r' d="([^"]+)"', (data / font / f"{code}.xml").read_text())
    if match is None:
        raise ValueError(f"No outline for {code} in {font}")
    polygons = _outline(match.group(1))
    scale = 8  # font units per em are 1000; draw large, then shrink
    points = np.concatenate(polygons)
    left, bottom = points.min(axis=0)
    right, top = points.max(axis=0)
    canvas = np.zeros(
        (
            int((top - bottom) / 1000 * HEIGHT * scale * 2) + 4,
            int((right - left) / 1000 * HEIGHT * scale * 2) + 4,
        ),
        np.uint8,
    )
    factor = (canvas.shape[0] - 4) / (top - bottom)
    shapes = [
        np.round(
            np.stack([(p[:, 0] - left) * factor + 2, (top - p[:, 1]) * factor + 2], axis=1)
        ).astype(np.int32)
        for p in polygons
    ]
    # Even-odd fill: a counter (the hole of a 0, 4, 6, 8, 9) is its own polygon.
    for shape in shapes:
        mask = np.zeros_like(canvas)
        cv2.fillPoly(mask, [shape], 255)
        canvas ^= mask
    width = max(1, round(canvas.shape[1] * HEIGHT / canvas.shape[0]))
    small = cv2.resize(canvas, (width, HEIGHT), interpolation=cv2.INTER_AREA)
    bits = small >= 128  # noqa: PLR2004
    return ["".join("#" if bit else "." for bit in row) for row in bits]


def main() -> None:
    data = Path(sys.argv[1])
    templates = {
        font: {str(digit): _glyph(data, font, digit) for digit in range(10)} for font in FONTS
    }
    OUT.write_text(json.dumps({"height": HEIGHT, "fonts": templates}, indent=1) + "\n")
    print("Wrote", OUT)  # noqa: T201


if __name__ == "__main__":
    main()
