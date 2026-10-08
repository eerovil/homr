"""
Find the slurs and ties printed on a system, from the picture.

homr's decoder writes one arc mark per note, shared by slurs and ties, and hangs
it on whichever voice it decodes; the page itself is unambiguous. Measured on the
owner-checked key of eerovil/musescore-choir-plugins#328 (27 systems, 110 slurs,
171 ties), every arc homr missed had a mark nearby: it saw the arc and wrote it
in the wrong place — two arcs sharing one mark, the other voice, one tie for a
whole chord, two nested slurs sharing one number. This module looks at the curves
themselves and points a person at every arc homr got wrong:

- ``find_curves`` takes the ink that is left once staff lines, noteheads, stems
  and signs are taken away, and keeps the thin, bent strokes long enough to join
  two notes. Dashed arcs are chained back together. Round two
  (eerovil/musescore-choir-plugins#333) went after the ties that were still
  neither found nor marked, and each had lost its curve a different way: a slur
  and a tie arriving at one note touch and come out as one blob (followed apart
  as two strands); a tie arriving from the line before is as short and deep as a
  fermata (a fermata now needs its dot); an arc grazing or lying along a staff
  line went with the line (only the line's own rows are cleared now); a speck
  the network called a notehead cut an arc in two (only blobs holding a detected
  head are taken out). A clef's curl, freed from the line by the same change, is
  dropped where the network says a clef is.
- ``attach`` hangs each end of a curve on a notehead the detector found, and each
  notehead on the notes homr wrote for it.
- ``mark`` compares those arcs with what homr wrote, one written arc for one
  curve, and **changes no arc**: every
  place the page shows a slur or tie that homr wrote differently or not at all
  gets a flag, which ``--mark-doubt`` turns into a red ``⚠ slur?`` or ``⚠ tie?``.
  Writing the picture's arcs instead was measured on the owner-checked key and
  dropped: it helped on the systems it was tuned on and not on held-out ones -- a
  scanned page's specks read as arcs -- while the flags alone pointed at 10 of
  the 14 wrongly set notes on the held-out systems and invented nothing.

Everything here is pure geometry on arrays the pipeline already has; it runs no
model.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import cv2
import numpy as np

from homr.model import InputPredictions, Note, Staff
from homr.type_definitions import NDArray


@dataclass
class StaffGeometry:
    """One printed staff: its five line heights at a few x positions, top first."""

    xs: list[float]
    lines: list[list[float]]  # per x, the five line y values
    unit: float
    min_x: float
    max_x: float

    def line_ys(self, x: float) -> list[float]:
        i = int(np.searchsorted(self.xs, x))
        if i <= 0:
            return self.lines[0]
        if i >= len(self.xs):
            return self.lines[-1]
        x0, x1 = self.xs[i - 1], self.xs[i]
        t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        return [a + t * (b - a) for a, b in zip(self.lines[i - 1], self.lines[i], strict=True)]

    def top(self, x: float) -> float:
        return self.line_ys(x)[0]

    def bottom(self, x: float) -> float:
        return self.line_ys(x)[-1]


@dataclass
class Head:
    """A notehead the detector found, in the coordinates of the processed page."""

    x: float
    y: float
    width: float
    height: float
    staff: int
    position: int


@dataclass
class Curve:
    """A thin bent stroke: its centre line sampled per column, and which way it bends."""

    xs: NDArray
    ys: NDArray
    over: bool  # bends upwards (an arc drawn above its notes)
    dashed: bool = False

    @property
    def left(self) -> tuple[float, float]:
        return float(self.xs[0]), float(self.ys[0])

    @property
    def right(self) -> tuple[float, float]:
        return float(self.xs[-1]), float(self.ys[-1])

    @property
    def width(self) -> float:
        return float(self.xs[-1] - self.xs[0])

    def sag(self) -> float:
        """How far the middle stands off the line joining the ends (positive = bulges up)."""
        mid = len(self.xs) // 2
        x0, y0 = self.left
        x1, y1 = self.right
        t = 0.0 if x1 == x0 else (self.xs[mid] - x0) / (x1 - x0)
        return float((y0 + t * (y1 - y0)) - self.ys[mid])


@dataclass
class CurveFinderSettings:
    ink_threshold: int = 150
    min_width_units: float = 1.0
    max_thickness_units: float = 0.45
    min_sag_px: float = 2.5
    min_sag_ratio: float = 0.008
    max_runs_share: float = 0.25
    dash_gap_units: float = 2.2


def _remove_staff_lines(ink: NDArray, staffs: list[StaffGeometry]) -> NDArray:
    """Take out the staff lines, but not a stroke that touches or crosses one.

    The detected line heights are a pixel or three out on a scan, so each line is
    found again in every column: the run of long horizontal ink nearest to where
    it should be. Only a run no thicker than a line is cleared; where an arc lies
    along a line the run is thicker and stays."""
    out = ink.copy()
    height, width = ink.shape
    for staff in staffs:
        reach = max(15, int(2 * staff.unit))
        horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, reach), np.uint8))
        x0 = max(0, int(staff.min_x) - 2)
        x1 = min(width, int(staff.max_x) + 3)
        for line in range(5):
            runs: dict[int, tuple[int, int]] = {}
            for x in range(x0, x1):
                y = staff.line_ys(float(x))[line]
                yi = int(round(y))
                lo, hi = max(0, yi - 4), min(height, yi + 5)
                rows = [r for r in range(lo, hi) if horizontal[r, x]]
                if not rows:
                    continue
                centre = min(rows, key=lambda r: abs(r - y))
                top = centre
                while top > 0 and ink[top - 1, x]:
                    top -= 1
                bottom = centre
                while bottom + 1 < height and ink[bottom + 1, x]:
                    bottom += 1
                runs[x] = (top, bottom)
            alone = {x: r for x, r in runs.items() if r[1] - r[0] + 1 <= 3}  # noqa: PLR2004
            span = max(8, int(4 * staff.unit))
            for x, (top, bottom) in runs.items():
                # where the line lies beside: an arc grazing it or running along it
                # adds rows on one side, and only the line's own rows are cleared
                beside = [alone[n] for n in range(x - span, x + span + 1) if n in alone]
                if len(beside) >= span // 2:
                    line_top = int(np.median([b[0] for b in beside]))
                    line_bottom = int(np.median([b[1] for b in beside]))
                    if top < line_top and bottom > line_bottom:
                        continue  # a stroke crossing the line keeps its ink
                    extra = (bottom - top) - (line_bottom - line_top)
                    if extra > max(3, 0.35 * staff.unit):
                        continue  # a beam or a notehead on the line, not an arc
                    out[max(top, line_top) : min(bottom, line_bottom) + 1, x] = 0
                elif bottom - top + 1 <= 3:  # noqa: PLR2004
                    out[top : bottom + 1, x] = 0
    return out


def _vertical_runs(ink: NDArray) -> NDArray:
    """For every ink pixel, how tall the unbroken run of ink in its column is."""
    height = ink.shape[0]
    down = np.zeros(ink.shape, dtype=np.int32)
    up = np.zeros(ink.shape, dtype=np.int32)
    for y in range(height):
        down[y] = (down[y - 1] + 1) * ink[y] if y else ink[y]
    for y in range(height - 1, -1, -1):
        up[y] = (up[y + 1] + 1) * ink[y] if y < height - 1 else ink[y]
    return np.where(ink > 0, down + up - 1, 0)


def _runs(column: NDArray) -> list[tuple[int, int]]:
    runs = []
    start = -1
    for i, value in enumerate(column):
        if value and start < 0:
            start = i
        elif not value and start >= 0:
            runs.append((start, i))
            start = -1
    if start >= 0:
        runs.append((start, len(column)))
    return runs


def _centre_line(component: NDArray) -> tuple[NDArray, NDArray, float, float, float]:
    """Per column, the middle of the ink; plus thickness, the share of columns with
    more than one run of ink, and the share of columns that have ink at all."""
    xs, ys, thickness, multi = [], [], [], 0
    columns = component.shape[1]
    for x in range(columns):
        runs = _runs(component[:, x])
        if not runs:
            continue
        if len(runs) > 1:
            multi += 1
        longest = max(runs, key=lambda r: r[1] - r[0])
        xs.append(x)
        ys.append((longest[0] + longest[1] - 1) / 2)
        thickness.append(longest[1] - longest[0])
    if not xs:
        return np.array([]), np.array([]), 0.0, 1.0, 0.0
    return (
        np.array(xs, dtype=float),
        np.array(ys, dtype=float),
        float(np.median(thickness)),
        multi / len(xs),
        len(xs) / max(1, columns),
    )


@dataclass
class _Stroke:
    x0: int
    y0: int
    xs: NDArray
    ys: NDArray
    thickness: float
    multi: float
    filled: float

    @property
    def gx(self) -> NDArray:
        return self.xs + self.x0

    @property
    def gy(self) -> NDArray:
        return self.ys + self.y0


def _strokes(residue: NDArray, unit: float, max_runs_share: float = 0.25) -> list[_Stroke]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(residue, connectivity=8)
    strokes = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < 2 or w < 2:
            continue
        component = (labels[y : y + h, x : x + w] == i).astype(np.uint8)
        xs, ys, thickness, multi, filled = _centre_line(component)
        if len(xs) == 0:
            continue
        strokes.append(_Stroke(int(x), int(y), xs, ys, thickness, multi, filled))
        if multi > max_runs_share:
            strokes += [_Stroke(int(x), int(y), *strand) for strand in _strands(component)]
    return strokes


def _strands(component: NDArray) -> list[tuple[NDArray, NDArray, float, float, float]]:
    """Two curves drawn one inside the other -- a slur and a tie arriving at one
    note -- touch where they meet the note and come out as one component with two
    runs of ink in most columns. Follow the upper and the lower run apart."""
    columns = component.shape[1]
    runs = [_runs(component[:, x]) for x in range(columns)]
    two = sum(1 for r in runs if len(r) == 2)  # noqa: PLR2004
    inked = sum(1 for r in runs if r)
    if inked == 0 or two < 0.5 * inked or any(len(r) > 2 for r in runs):  # noqa: PLR2004
        return []
    strands: list[tuple[list[float], list[float], list[float]]] = [([], [], []), ([], [], [])]
    for x, column in enumerate(runs):
        if len(column) == 2:  # noqa: PLR2004
            for strand, (a, b) in zip(strands, column, strict=True):
                strand[0].append(x)
                strand[1].append((a + b - 1) / 2)
                strand[2].append(b - a)
        elif len(column) == 1:
            a, b = column[0]
            y = (a + b - 1) / 2
            # where the two have met, the run belongs to the one it continues
            near = min(strands, key=lambda st: abs(st[1][-1] - y) if st[1] else math.inf)
            if near[1] and b - a <= 1.5 * float(np.median(near[2])) + 1:
                near[0].append(x)
                near[1].append(y)
                near[2].append(b - a)
    out: list[tuple[NDArray, NDArray, float, float, float]] = []
    for columns_at, centres, thickness in strands:
        if len(columns_at) < 5:  # noqa: PLR2004
            return []
        out.append(
            (
                np.array(columns_at, dtype=float),
                np.array(centres, dtype=float),
                float(np.median(thickness)),
                0.0,
                len(columns_at) / (columns_at[-1] - columns_at[0] + 1),
            )
        )
    # a letter or a clef's loop also has two runs of ink in a column; two arcs
    # are two clean parabolas bending the same way over mostly the same columns
    (xa, ya, *_), (xb, yb, *_) = out
    bends = [_bend(xa, ya), _bend(xb, yb)]
    overlap = min(xa[-1], xb[-1]) - max(xa[0], xb[0])
    if (
        bends[0] * bends[1] <= 0
        or min(abs(b) for b in bends) < 2.5  # noqa: PLR2004
        or max(_fit_error(xa, ya), _fit_error(xb, yb)) > 1.5  # noqa: PLR2004
        or overlap < 0.7 * min(xa[-1] - xa[0], xb[-1] - xb[0])
    ):
        return []
    return out


def _bend(xs: NDArray, ys: NDArray) -> float:
    """The quadratic coefficient of a fit, scaled to pixels of sag over the width."""
    if len(xs) < 5:
        return 0.0
    width = xs[-1] - xs[0]
    if width <= 0:
        return 0.0
    t = (xs - xs[0]) / width
    a = np.polyfit(t, ys, 2)[0]
    # y grows downwards, so an arc bulging upwards opens downwards in y: a > 0
    return float(a / 4)


def _fit_error(xs: NDArray, ys: NDArray) -> float:
    if len(xs) < 5:
        return 0.0
    t = (xs - xs[0]) / max(1.0, xs[-1] - xs[0])
    coefficients = np.polyfit(t, ys, 2)
    return float(np.sqrt(np.mean((np.polyval(coefficients, t) - ys) ** 2)))


def find_curves(
    gray: NDArray,
    notehead: NDArray,
    stems_rest: NDArray,
    clefs_keys: NDArray,
    symbols: NDArray,
    staffs: list[StaffGeometry],
    settings: CurveFinderSettings | None = None,
    heads: list[Head] | None = None,
) -> list[Curve]:
    settings = settings or CurveFinderSettings()
    unit = float(np.median([s.unit for s in staffs])) if staffs else 10.0
    ink = (gray < settings.ink_threshold).astype(np.uint8)
    ink = _remove_staff_lines(ink, staffs)
    take_out: NDArray = cv2.dilate(_real_heads(notehead, heads), np.ones((3, 3), np.uint8))
    # The network labels an arc touching an accidental or a stem as part of it, so
    # those masks are only trusted where the ink stands upright: an arc is never
    # more than a few pixels tall in any one column.
    upright = _vertical_runs(ink) > max(4, int(0.4 * unit))
    for mask in (stems_rest, clefs_keys, symbols):
        solid = mask.astype(np.uint8) & upright.astype(np.uint8)
        take_out = take_out | cv2.dilate(solid, np.ones((1, 3), np.uint8))
    residue: NDArray = (ink & (1 - take_out)).astype(np.uint8)
    # close the small gaps left where a curve crossed a line or a stem
    residue = cv2.morphologyEx(residue, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    strokes = _strokes(residue, unit, settings.max_runs_share)
    thin = [
        s
        for s in strokes
        if s.thickness <= max(2.5, settings.max_thickness_units * unit)
        and s.multi <= settings.max_runs_share
    ]
    # a fermata's dot is looked for in the ink that is not a notehead: the head an
    # arc arrives at stands right beside it
    near_head: NDArray = cv2.dilate(notehead.astype(np.uint8), np.ones((5, 5), np.uint8))
    head_free: NDArray = (ink & (1 - near_head)).astype(np.uint8)
    curves = _solid_curves(thin, unit, settings, head_free)
    curves += [
        c
        for c in _dashed_curves(thin, unit, settings, curves)
        if not _along_lines(c, staffs) and abs(c.sag()) >= 0.3 * unit
    ]
    # a clef's curl is a clean curve too; the network says where the clefs are
    clef = cv2.dilate(clefs_keys.astype(np.uint8), np.ones((5, 5), np.uint8))
    return [c for c in curves if _share_on(c, clef) <= 0.5]  # noqa: PLR2004


def _real_heads(notehead: NDArray, heads: list[Head] | None) -> NDArray:
    """The notehead mask, less the blobs no head was detected in: the network
    marks a speck of an arc as a notehead now and then, and taking it out cuts
    the arc in two."""
    mask = (notehead > 0).astype(np.uint8)
    if heads is None:
        return mask
    count, labels = cv2.connectedComponents(mask, connectivity=8)
    keep = np.zeros(count, dtype=np.uint8)
    for head in heads:
        x0, x1 = int(head.x - head.width / 2) - 1, int(head.x + head.width / 2) + 2
        y0, y1 = int(head.y - head.height / 2) - 1, int(head.y + head.height / 2) + 2
        keep[np.unique(labels[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)])] = 1
    keep[0] = 0
    return keep[labels]


def _share_on(curve: Curve, mask: NDArray) -> float:
    xs = np.clip(curve.xs.astype(int), 0, mask.shape[1] - 1)
    ys = np.clip(np.round(curve.ys).astype(int), 0, mask.shape[0] - 1)
    return float(np.mean(mask[ys, xs] > 0))


def _along_lines(curve: Curve, staffs: list[StaffGeometry]) -> bool:
    """A scanned staff line breaks into dashes; a chain lying on one is the line."""
    near = 0
    for x, y in zip(curve.xs, curve.ys, strict=True):
        for staff in staffs:
            if staff.min_x - 5 <= x <= staff.max_x + 5:
                if min(abs(y - line) for line in staff.line_ys(float(x))) <= 2.5:
                    near += 1
                    break
    return near >= 0.4 * len(curve.xs)


def _is_curve(xs: NDArray, ys: NDArray, unit: float, settings: CurveFinderSettings) -> bool:
    width = xs[-1] - xs[0]
    if width < settings.min_width_units * unit:
        return False
    sag = _bend(xs, ys)
    if abs(sag) < max(settings.min_sag_px, settings.min_sag_ratio * width):
        return False
    # a bent stroke follows a parabola; letters and wedges do not
    return _fit_error(xs, ys) <= max(1.5, 0.12 * unit, 0.25 * abs(sag))


def _turns(ys: NDArray, threshold: float) -> list[tuple[int, int]]:
    """The interior turning points of a line, as (index, +1 for a low point on the
    page / -1 for a high one), ignoring wiggles smaller than ``threshold``."""
    turns: list[tuple[int, int]] = []
    direction = 0
    extreme = 0
    for i in range(1, len(ys)):
        if direction >= 0 and ys[i] > ys[extreme]:
            extreme = i if direction > 0 or ys[i] - ys[0] > threshold else extreme
            if direction == 0 and ys[i] - ys[0] > threshold:
                direction, extreme = 1, i
        elif direction <= 0 and ys[i] < ys[extreme]:
            if direction == 0 and ys[0] - ys[i] > threshold:
                direction, extreme = -1, i
            elif direction < 0:
                extreme = i
        if direction > 0 and ys[extreme] - ys[i] > threshold:
            turns.append((extreme, 1))
            direction, extreme = -1, i
        elif direction < 0 and ys[i] - ys[extreme] > threshold:
            turns.append((extreme, -1))
            direction, extreme = 1, i
    return turns


def _split(xs: NDArray, ys: NDArray, unit: float) -> list[tuple[NDArray, NDArray]]:
    """Ties one after another touch at the notes they share and come out as one
    stroke, scalloped: cut it where it turns back towards the notes."""
    turns = _turns(ys, max(2.0, 0.2 * unit))
    if len(turns) < 2:
        return [(xs, ys)]
    # the apexes are the turns of the commoner kind; cut at the others
    lows = sum(1 for _, kind in turns if kind > 0)
    apex = 1 if lows * 2 >= len(turns) else -1
    cuts = [i for i, kind in turns if kind != apex]
    pieces = []
    previous = 0
    for cut in cuts + [len(xs) - 1]:
        if cut - previous >= 3:
            pieces.append((xs[previous : cut + 1], ys[previous : cut + 1]))
        previous = cut
    return pieces


def _fermata(xs: NDArray, ys: NDArray, unit: float, ink: NDArray) -> bool:
    """A fermata's arc is short and deep, with a dot inside it.

    Anything this deep and narrower than 1.6 spaces goes too: it is a piece of a
    letter or a flag far more often than an arc. Between that and 2.2 spaces the
    dot is what tells them apart: a tie arriving from the line before is just as
    deep where it meets the system's first note, and has none."""
    width = xs[-1] - xs[0]
    if width > 2.2 * unit or abs(_bend(xs, ys)) < 0.25 * width:
        return False
    return width < 1.6 * unit or _dot_inside(xs, ys, unit, ink)


def _dot_inside(xs: NDArray, ys: NDArray, unit: float, ink: NDArray) -> bool:
    """A small solid blob between the arc and the line joining its ends."""
    width = xs[-1] - xs[0]
    chord = (ys[0] + ys[-1]) / 2
    apex = ys[len(ys) // 2]
    x0 = max(0, int(xs[0] + 0.2 * width))
    x1 = min(ink.shape[1], int(xs[-1] - 0.2 * width) + 1)
    # inside the arc only: from its apex to just past the line joining its ends
    if apex < chord:
        y0, y1 = int(apex), int(chord + 0.4 * unit) + 1
    else:
        y0, y1 = int(chord - 0.4 * unit), int(apex) + 1
    y0, y1 = max(0, y0), min(ink.shape[0], y1)
    window = ink[y0:y1, x0:x1].copy()
    if window.size == 0:
        return False
    # the arc's own ink is not a dot, nor is a staff line left thick where it runs
    # along the arc: a fermata's dot stands clear of its arc
    clear = int(0.35 * unit) + 1
    for x, y in zip(xs, ys, strict=True):
        if x0 <= x < x1:
            window[max(0, int(y) - clear - y0) : max(0, int(y) + clear + 1 - y0), int(x) - x0] = 0
    count, _, stats, _ = cv2.connectedComponentsWithStats(window, connectivity=8)
    for i in range(1, count):
        _, _, w, h, area = stats[i]
        if 2 <= w <= 0.8 * unit and 2 <= h <= 0.8 * unit and area >= 0.5 * w * h:  # noqa: PLR2004
            return True
    return False


def _solid_curves(
    strokes: list[_Stroke], unit: float, settings: CurveFinderSettings, ink: NDArray
) -> list[Curve]:
    curves = []
    for stroke in strokes:
        if stroke.filled < 0.8:
            continue
        pieces = _split(stroke.gx, stroke.gy, unit)
        if not any(_is_curve(xs, ys, unit, settings) for xs, ys in pieces):
            # lumps where an arc merges with a staff line are not the turns
            # between ties: an arc cut into nothing but scraps is one arc
            pieces = [(stroke.gx, stroke.gy)]
        for xs, ys in pieces:
            if not _is_curve(xs, ys, unit, settings):
                continue
            over = _bend(xs, ys) > 0
            if _fermata(xs, ys, unit, ink):
                continue
            curves.append(Curve(xs, ys, over=over))
    return curves


def _dashed_curves(
    strokes: list[_Stroke], unit: float, settings: CurveFinderSettings, solid: list[Curve]
) -> list[Curve]:
    """Chain short dashes lying along one smooth line into a curve."""
    dashes = [
        s
        for s in strokes
        if (s.gx[-1] - s.gx[0]) <= 2.0 * unit and s.thickness <= max(2.5, 0.3 * unit)
    ]
    dashes.sort(key=lambda s: s.gx[0])
    used: set[int] = set()
    curves = []
    for i, first in enumerate(dashes):
        if i in used:
            continue
        chain = [i]
        last = first
        while True:
            best, best_cost = -1, math.inf
            for j in range(chain[-1] + 1, len(dashes)):
                if j in used or j in chain:
                    continue
                cand = dashes[j]
                gap = cand.gx[0] - last.gx[-1]
                if gap < -1:
                    continue
                if gap > settings.dash_gap_units * unit:
                    break
                dy = abs(cand.gy[0] - last.gy[-1])
                if dy > 0.6 * unit + 0.25 * gap:
                    continue
                cost = gap + 2 * dy
                if cost < best_cost:
                    best, best_cost = j, cost
            if best < 0:
                break
            chain.append(best)
            last = dashes[best]
        if len(chain) < 4:
            continue
        xs = np.concatenate([dashes[k].gx for k in chain])
        ys = np.concatenate([dashes[k].gy for k in chain])
        order = np.argsort(xs, kind="stable")
        xs, ys = xs[order], ys[order]
        if not _is_curve(xs, ys, unit, settings):
            continue
        used.update(chain)
        curves.append(Curve(xs, ys, over=_bend(xs, ys) > 0, dashed=True))
    return curves


# ----------------------------------------------------------------------------
# The notes homr wrote, and which detected notehead each one is
# ----------------------------------------------------------------------------

_STEPS = "CDEFGAB"


@dataclass
class XmlNote:
    element: ET.Element
    part: int
    staff: int  # printed staff of the whole system, from 0
    voice: str
    bar: int  # measure index in the part, from 0
    onset: Fraction
    end: Fraction
    pitch: tuple[str, int, int]  # step, alter, octave
    imgpos: tuple[float, float] | None
    number: str = ""  # the measure's own number
    head: int = -1  # index into the detected heads, -1 when unmatched

    @property
    def diatonic(self) -> int:
        return _STEPS.index(self.pitch[0]) + 7 * self.pitch[2]


def _imgpos(note: ET.Element) -> tuple[float, float] | None:
    for child in note:
        if callable(child.tag) and child.text and "imgpos:" in child.text:
            x, y = child.text.split("imgpos:", 1)[1].split(",")
            return float(x), float(y)
    return None


def read_notes(xml: ET.Element) -> list[XmlNote]:
    """Every pitched note, in the order written, with where it sits in time."""
    notes: list[XmlNote] = []
    printed = 0
    for part_index, part in enumerate(xml.findall("part")):
        staves = max((int(n.text or 1) for n in part.iter("staves")), default=1)
        base, printed = printed, printed + staves
        divisions = Fraction(1)
        for bar, measure in enumerate(part.findall("measure")):
            for attributes in measure.findall("attributes"):
                declared = attributes.findtext("divisions")
                if declared:
                    divisions = Fraction(declared) or Fraction(1)
            at = previous = Fraction(0)
            for node in measure:
                if node.tag in ("backup", "forward"):
                    step = Fraction(node.findtext("duration", "0")) / divisions
                    at = at - step if node.tag == "backup" else at + step
                    continue
                if node.tag != "note":
                    continue
                grace = node.find("grace") is not None
                length = (
                    Fraction(0) if grace else Fraction(node.findtext("duration", "0")) / divisions
                )
                chord = node.find("chord") is not None
                onset = previous if chord else at
                if not chord:
                    previous, at = at, at + length
                pitch = node.find("pitch")
                if pitch is None:
                    continue
                notes.append(
                    XmlNote(
                        node,
                        part_index,
                        base + int(node.findtext("staff", "1")) - 1,
                        node.findtext("voice", "1"),
                        bar,
                        onset,
                        onset + length,
                        (
                            pitch.findtext("step", "C"),
                            int(float(pitch.findtext("alter", "0") or 0)),
                            int(pitch.findtext("octave", "4")),
                        ),
                        _imgpos(node),
                        measure.get("number", str(bar + 1)),
                    )
                )
    return notes


def match_heads(
    notes: list[XmlNote],
    heads: list[Head],
    staffs: list[StaffGeometry],
    to_page: Callable[[tuple[float, float]], tuple[float, float]],
) -> bool:
    """Give each written note the detected notehead it stands for.

    homr's own position for a note comes from the decoder's attention and is a
    staff space or two out, so it only says roughly where to look: the head is the
    one at the note's staff position nearest to it. Returns False when the staffs
    written and the staffs detected do not line up, and nothing can be trusted."""
    if not notes or max(n.staff for n in notes) >= len(staffs):
        return False
    by_staff: dict[int, list[int]] = defaultdict(list)
    for index, head in enumerate(heads):
        by_staff[head.staff].append(index)
    for staff in range(len(staffs)):
        mine = [n for n in notes if n.staff == staff and n.imgpos is not None]
        candidates = by_staff.get(staff, [])
        if not mine or not candidates:
            continue
        unit = staffs[staff].unit
        # the clef: a staff position is the pitch's step plus one fixed offset
        guesses = []
        for note in mine:
            x, _ = to_page(note.imgpos)  # type: ignore[arg-type]
            nearest = min(candidates, key=lambda i: abs(heads[i].x - x))
            guesses.append(heads[nearest].position - note.diatonic)
        offset = Counter(guesses).most_common(1)[0][0]
        for note in mine:
            x, _ = to_page(note.imgpos)  # type: ignore[arg-type]
            want = note.diatonic + offset
            fitting = [
                i
                for i in candidates
                if abs(heads[i].position - want) <= 1 and abs(heads[i].x - x) <= 4 * unit
            ]
            if fitting:
                note.head = min(
                    fitting,
                    key=lambda i: abs(heads[i].x - x) + unit * abs(heads[i].position - want),
                )
    return True


def anchors(
    notes: list[XmlNote],
    heads: list[Head],
    staffs: list[StaffGeometry],
    to_page: Callable[[tuple[float, float]], tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    """Where each note's head is on the page: the detected head, or for a note no
    head was matched to, homr's x and the y its pitch puts it at on the staff."""
    out: dict[int, tuple[float, float]] = {}
    offsets: dict[int, Counter[int]] = defaultdict(Counter)
    for note in notes:
        if note.head >= 0:
            offsets[note.staff][heads[note.head].position - note.diatonic] += 1
    for i, note in enumerate(notes):
        if note.head >= 0:
            out[i] = (heads[note.head].x, heads[note.head].y)
        elif note.imgpos is not None and offsets[note.staff]:
            x, _ = to_page(note.imgpos)
            position = note.diatonic + offsets[note.staff].most_common(1)[0][0]
            lines = staffs[note.staff].line_ys(x)
            bottom, unit = lines[-1], staffs[note.staff].unit
            out[i] = (x, bottom - (position - 1) * unit / 2)
    return out


@dataclass
class PictureArc:
    kind: str  # "slur" or "tie"
    start: int | None  # index into the notes, None = from the system before
    stop: int | None  # None = into the next system
    staff: int
    over: bool


@dataclass
class _End:
    note: int
    cost: float


def _end_candidates(
    end: tuple[float, float],
    left: bool,
    over: bool,
    notes: list[XmlNote],
    where: dict[int, tuple[float, float]],
    staffs: list[StaffGeometry],
    width: float,
) -> list[_End]:
    out = []
    ex, ey = end
    for i, (hx, hy) in where.items():
        unit = staffs[notes[i].staff].unit
        dx = (ex - hx) if left else (hx - ex)  # how far the end sits inside the arc from the head
        # a long phrase arc may start well before its note; a tie never does
        outside = 2.6 * unit if width > 6 * unit else 1.0 * unit
        if not -outside <= dx <= 3.2 * unit:
            continue
        dy = (hy - ey) if over else (ey - hy)  # how far the end sits off the head, outwards
        reach = 5.5 * unit if width > 3 * unit else 3.0 * unit
        if not -0.6 * unit <= dy <= reach:
            continue
        cost = abs(dx - 0.5 * unit) / unit + 0.5 * max(0.0, dy - 1.2 * unit) / unit
        if width <= 4 * unit:
            # a tie sits right against its own head: of a chord, the nearest one
            cost += 0.4 * abs(dy) / unit
        if notes[i].head < 0:
            cost += 0.3
        out.append(_End(i, cost))
    out.sort(key=lambda e: e.cost)
    return out


def _onset_key(note: XmlNote) -> tuple[int, Fraction]:
    return note.bar, note.onset


def attach(
    curves: list[Curve],
    notes: list[XmlNote],
    where: dict[int, tuple[float, float]],
    staffs: list[StaffGeometry],
    unplaced: list[int] | None = None,
) -> list[PictureArc]:
    """Hang each curve on the two notes it joins, or on one note and the system edge.

    A clear curve that could not be hung on anything is not dropped in silence:
    the note nearest to it goes into ``unplaced``, for a mark saying so."""
    arcs = []
    first_x = {
        s: min((where[i][0] for i in where if notes[i].staff == s), default=math.inf)
        for s in range(len(staffs))
    }
    last_x = {
        s: max((where[i][0] for i in where if notes[i].staff == s), default=-math.inf)
        for s in range(len(staffs))
    }
    for curve in curves:
        width = curve.width
        lefts = _end_candidates(curve.left, True, curve.over, notes, where, staffs, width)
        rights = _end_candidates(curve.right, False, curve.over, notes, where, staffs, width)
        staff_here = _staff_at(curve, staffs)
        edge_left = (
            staff_here is not None
            and curve.left[0] < first_x[staff_here] - 0.5 * staffs[staff_here].unit
        )
        edge_right = staff_here is not None and (
            curve.right[0] > last_x[staff_here] + 1.0 * staffs[staff_here].unit
            or (
                curve.right[0] >= staffs[staff_here].max_x - 1.5 * staffs[staff_here].unit
                and curve.right[0] > last_x[staff_here] + 0.3 * staffs[staff_here].unit
            )
        )
        if edge_right:
            rights = []
        best: tuple[float, int | None, int | None] | None = None
        left_choices: list[_End | None] = list(lefts[:4]) or ([None] if edge_left else [])
        right_choices: list[_End | None] = list(rights[:4]) or ([None] if edge_right else [])
        for a in left_choices:
            for b in right_choices:
                if a is None and b is None:
                    continue
                na = notes[a.note] if a else None
                nb = notes[b.note] if b else None
                if na and nb:
                    if na.staff != nb.staff or _onset_key(na) >= _onset_key(nb):
                        continue
                cost = (a.cost if a else 0.8) + (b.cost if b else 0.8)
                if na and nb and na.pitch == nb.pitch and width <= 6 * staffs[na.staff].unit:
                    cost -= 0.6  # a short arc between two equal notes is a tie
                if best is None or cost < best[0]:
                    best = (cost, a.note if a else None, b.note if b else None)
        if best is None:
            if (
                unplaced is not None
                and staff_here is not None
                and _clear(curve, staffs[staff_here])
            ):
                nearest = _nearest_note(curve, staff_here, notes, where)
                if nearest is not None:
                    unplaced.append(nearest)
            continue
        _, start, stop = best
        staff = notes[start if start is not None else stop].staff  # type: ignore[index]
        start, stop = _pick_voice(start, notes, curve.over), _pick_voice(stop, notes, curve.over)
        kind = "slur"
        if start is not None and stop is not None and _tied(notes[start], notes[stop], notes):
            kind = "tie"
        arcs.append(PictureArc(kind, start, stop, staff, curve.over))
    return arcs


def _clear(curve: Curve, staff: StaffGeometry) -> bool:
    """A curve worth a mark: long enough to join two notes, and by a staff."""
    if curve.width < 2 * staff.unit:
        return False
    x, y = curve.xs[len(curve.xs) // 2], curve.ys[len(curve.ys) // 2]
    return staff.top(float(x)) - 5 * staff.unit <= y <= staff.bottom(float(x)) + 5 * staff.unit


def _nearest_note(
    curve: Curve, staff: int, notes: list[XmlNote], where: dict[int, tuple[float, float]]
) -> int | None:
    mine = [i for i in where if notes[i].staff == staff]
    if not mine:
        return None
    x = curve.left[0]
    return min(mine, key=lambda i: abs(where[i][0] - x))


def _staff_at(curve: Curve, staffs: list[StaffGeometry]) -> int | None:
    x, y = curve.xs[len(curve.xs) // 2], curve.ys[len(curve.ys) // 2]
    best, distance = None, math.inf
    for i, staff in enumerate(staffs):
        top, bottom = staff.top(float(x)), staff.bottom(float(x))
        d = 0.0 if top <= y <= bottom else min(abs(y - top), abs(y - bottom))
        if d < distance:
            best, distance = i, d
    return best


def _pick_voice(index: int | None, notes: list[XmlNote], over: bool) -> int | None:
    """Two voices sharing one notehead: an arc above belongs to the upper voice."""
    if index is None:
        return None
    note = notes[index]
    if note.head < 0:
        return index
    twins = [
        i for i, other in enumerate(notes) if other.head == note.head and other.staff == note.staff
    ]
    if len(twins) == 1:
        return index
    twins.sort(key=lambda i: int(notes[i].voice) if notes[i].voice.isdigit() else 0)
    return twins[0] if over else twins[-1]


def _tied(a: XmlNote, b: XmlNote, notes: list[XmlNote]) -> bool:
    """Same pitch, same voice, and nothing of that voice sounding between them."""
    if a.pitch != b.pitch or a.voice != b.voice or a.staff != b.staff:
        return False
    for other in notes:
        if other.staff == a.staff and other.voice == a.voice and other is not a and other is not b:
            if _onset_key(a) < _onset_key(other) < _onset_key(b) and other.onset != a.onset:
                return False
    return True


def extend_over_ties(arcs: list[PictureArc], ties: list[tuple[int, int]]) -> None:
    """A slur drawn over tied notes ends on the last of them, though its curve may
    come down nearer an earlier one (eerovil/musescore-choir-plugins#318)."""
    following = dict(ties)
    for arc in arcs:
        if arc.kind != "slur" or arc.stop is None:
            continue
        seen = {arc.stop}
        while arc.stop in following and following[arc.stop] not in seen:
            arc.stop = following[arc.stop]
            seen.add(arc.stop)


# ----------------------------------------------------------------------------
# Comparing with what homr wrote
# ----------------------------------------------------------------------------

#: Set on a note, valued ``slur`` or ``tie``, where the page shows an arc homr's
#: output does not match; ``--mark-doubt`` turns it into a red mark
#: (homr/doubt.py), and ``forget`` takes it off otherwise.
ARC_DOUBT = "homr-arc-doubt"


@dataclass
class WrittenArc:
    kind: str
    start: int | None
    stop: int | None
    staff: int


def read_written(notes: list[XmlNote]) -> list[WrittenArc]:
    """The slurs and ties homr wrote, paired the way a reader of the file pairs them:
    slurs per staff and number, ties per staff and pitch, in the order written."""
    arcs: list[WrittenArc] = []
    open_: dict[tuple, list[int]] = defaultdict(list)
    for i, note in enumerate(notes):
        marks: list[tuple[str, str, str]] = []
        for slur in note.element.iter("slur"):
            marks.append(("slur", slur.get("number", "1"), slur.get("type", "")))
        for tied in note.element.iter("tied"):
            marks.append(("tie", str(note.pitch), tied.get("type", "")))
        marks.sort(key=lambda m: m[2] != "stop")
        for kind, label, what in marks:
            key = (kind, note.staff, label)
            if what == "start":
                open_[key].append(i)
            elif what == "stop":
                start = open_[key].pop(0) if open_[key] else None
                arcs.append(WrittenArc(kind, start, i, note.staff))
    for (kind, staff, _), waiting in open_.items():
        arcs.extend(WrittenArc(kind, start, None, staff) for start in waiting)
    return arcs


def _place(notes: list[XmlNote], index: int | None) -> tuple | None:
    if index is None:
        return None
    note = notes[index]
    return note.bar, note.onset


def _agrees(notes: list[XmlNote], arc: PictureArc, written: WrittenArc) -> bool:
    """homr wrote the arc the picture shows -- or a tie between two equal notes of
    a chord where the picture, hanging the curve on the chord, chose another of its
    notes: a chord's tie is to its own note, so homr's is the one to believe."""
    if written.kind == arc.kind and (written.start, written.stop) == (arc.start, arc.stop):
        return True
    return (
        written.kind == "tie"
        and written.staff == arc.staff
        and _place(notes, written.start) == _place(notes, arc.start)
        and _place(notes, written.stop) == _place(notes, arc.stop)
        and written.start is not None
        and written.stop is not None
        and _tied(notes[written.start], notes[written.stop], notes)
        and (arc.kind == "slur" or notes[arc.start or 0].pitch == notes[written.start].pitch)
    )


@dataclass
class Outcome:
    agreed: int = 0
    marked: int = 0


def mark(notes: list[XmlNote], picture: list[PictureArc], unplaced: list[int]) -> Outcome:
    """Flag every arc on the page that homr's output does not match, changing none.

    The flag goes on the arc's first note and names the picture's kind, so a tie
    homr missed is marked ``tie?``. A curve that could not be hung on notes at all
    flags the note beside it as ``slur?``."""
    outcome = Outcome()
    written = read_written(notes)
    # each arc homr wrote answers for one arc on the page: a slur and a tie both
    # arriving at one note are two curves, and homr writing one of them leaves the
    # other to be marked
    unused = list(written)
    for arc in picture:
        match = next((w for w in unused if _agrees(notes, arc, w)), None)
        if match is not None:
            unused.remove(match)
            outcome.agreed += 1
            continue
        index = arc.start if arc.start is not None else arc.stop
        if index is not None:
            _flag(notes[index].element, arc.kind)
            outcome.marked += 1
    for index in unplaced:
        _flag(notes[index].element, "slur")
        outcome.marked += 1
    return outcome


def _flag(note: ET.Element, kind: str) -> None:
    # a tie flag never hides a slur flag on the same note: both are words of the mark
    kinds = set(filter(None, (note.get(ARC_DOUBT) or "").split(","))) | {kind}
    note.set(ARC_DOUBT, ",".join(sorted(kinds)))


# ----------------------------------------------------------------------------
# The pipeline's door
# ----------------------------------------------------------------------------


def staff_geometry(staff: Staff) -> StaffGeometry:
    points = sorted(staff.grid, key=lambda p: p.x)
    return StaffGeometry(
        [p.x for p in points],
        [sorted(p.y)[:5] for p in points],
        float(staff.average_unit_size),
        float(staff.min_x),
        float(staff.max_x),
    )


def detected_heads(staffs: list[Staff]) -> list[Head]:
    heads = []
    for index, staff in enumerate(staffs):
        for symbol in staff.symbols:
            if isinstance(symbol, Note):
                (x, y), (width, height), _ = symbol.box.box
                heads.append(Head(x, y, width, height, index, symbol.position))
    return heads


def invert(
    mapping: Callable[[tuple[float, float]], tuple[float, float]],
) -> Callable[[tuple[float, float]], tuple[float, float]]:
    """The pipeline's page-to-input mapping is a scale and a shift: undo it."""
    ox, oy = mapping((0.0, 0.0))
    ax, ay = mapping((1000.0, 1000.0))
    sx, sy = (ax - ox) / 1000.0, (ay - oy) / 1000.0

    def back(point: tuple[float, float]) -> tuple[float, float]:
        return (point[0] - ox) / sx, (point[1] - oy) / sy

    return back


def mark_arcs(
    xml: ET.Element,
    predictions: InputPredictions,
    staffs: list[Staff],
    page_to_input: Callable[[tuple[float, float]], tuple[float, float]],
) -> Outcome | None:
    """Flag every slur and tie on the page that ``xml`` does not match.

    ``staffs`` are the printed staffs as detected, before grand staffs are joined.
    Returns None, flagging nothing, when the written staffs and the detected ones
    do not line up: a mark on the wrong staff sends a person to the wrong place."""
    printed = sorted(staffs, key=lambda s: float(np.mean([np.mean(p.y) for p in s.grid])))
    geometry = [staff_geometry(s) for s in printed]
    if any(len(g.lines[0]) != 5 for g in geometry):  # noqa: PLR2004
        return None
    return mark_page(
        xml,
        predictions.preprocessed,
        [predictions.notehead, predictions.stems_rest, predictions.clefs_keys, predictions.symbols],
        geometry,
        detected_heads(printed),
        invert(page_to_input),
    )


def mark_page(
    xml: ET.Element,
    gray: NDArray,
    masks: list[NDArray],
    geometry: list[StaffGeometry],
    heads: list[Head],
    to_page: Callable[[tuple[float, float]], tuple[float, float]],
) -> Outcome | None:
    """``mark_arcs`` on what the pipeline has already worked out: the page, the
    network's notehead, stem, clef and symbol masks, the staffs and the heads."""
    notes = read_notes(xml)
    if not match_heads(notes, heads, geometry, to_page):
        return None
    where = anchors(notes, heads, geometry, to_page)
    notehead, stems_rest, clefs_keys, symbols = masks
    curves = find_curves(gray, notehead, stems_rest, clefs_keys, symbols, geometry, heads=heads)
    unplaced: list[int] = []
    arcs = attach(curves, notes, where, geometry, unplaced)
    extend_over_ties(
        arcs,
        [
            (a.start, a.stop)
            for a in arcs
            if a.kind == "tie" and a.start is not None and a.stop is not None
        ]
        + [
            (w.start, w.stop)
            for w in read_written(notes)
            if w.kind == "tie" and w.start is not None and w.stop is not None
        ],
    )
    return mark(notes, arcs, unplaced)


def forget(xml: ET.Element) -> None:
    """Take the working marks off: they are for --mark-doubt, not for the file."""
    for element in xml.iter():
        element.attrib.pop(ARC_DOUBT, None)
