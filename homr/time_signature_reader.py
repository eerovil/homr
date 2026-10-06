"""Read a time signature's printed digits off the page.

The model's vocabulary holds only a time signature's **denominator**
(`timeSignature/4`); the numerator has always been guessed afterwards from how
long the decoded bars came out. Where the decoder misread the rhythms, that guess
is wrong in the same way, and a system that changes meter part-way through gets
no token at all at the change: on Legenda system 10
(eerovil/musescore-choir-plugins#267) the page prints 3/4, 4/4, 3/4 and the
decoder emitted one `timeSignature/4`, at the start.

So the digits are read here, from the page image, by matching their shapes
against the digits of three music fonts (`time_signature_digits.json`, made by
`scripts/time_signature_digit_templates.py`). Nothing is trained. A time
signature is two digit stacks on one staff -- the numerator filling the space
from the top line to the middle line, the denominator from the middle line to the
bottom one -- and that geometry is what finds them: anything else that fills
exactly those two halves of the staff, side by side, is rare, and the digit match
has to agree as well.

What comes back is evidence, not a verdict: `score_reconstruction` weighs it
against the bar lengths the notes add up to (`meter_lengths`).
"""

import json
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from homr.model import Staff
from homr.point_mapping import PointMapping
from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol

_TEMPLATES = Path(__file__).with_name("time_signature_digits.json")

#: A digit is accepted when its shape correlates with a font's digit at least this
#: well, and better than with any other digit by `_MARGIN`. Measured on Legenda's
#: 24 systems: printed digits score 0.75-0.93 against their own digit, while the
#: best wrong digit stays at least 0.08 behind.
_MIN_SCORE = 0.5
_MARGIN = 0.1

_DENOMINATORS = (1, 2, 4, 8, 16, 32)
_ANY_DIGIT = frozenset(range(10))
_LEADING = frozenset(range(1, 10))
_MAX_NUMERATOR = 16


@dataclass(frozen=True)
class PrintedMeter:
    """One time signature read off one staff: where it stands, and what it says."""

    x: float
    numerator: int
    denominator: int
    #: The weaker of the digits' match scores, 0..1.
    score: float

    @property
    def bar_length(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}@{self.x:.0f}"


@cache
def _templates() -> dict[int, list[NDArray[np.float64]]]:
    data = json.loads(_TEMPLATES.read_text())
    out: dict[int, list[NDArray[np.float64]]] = {digit: [] for digit in range(10)}
    for font in data["fonts"].values():
        for digit, rows in font.items():
            out[int(digit)].append(
                np.array([[ch == "#" for ch in row] for row in rows], dtype=np.float64)
            )
    return out


def _correlate(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    a = a - a.mean()
    b = b - b.mean()
    norm = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / norm) if norm > 0 else 0.0


def classify_digit(
    ink: NDArray[np.bool_],
    on_line: NDArray[np.bool_] | None = None,
    allowed: frozenset[int] = _ANY_DIGIT,
) -> tuple[int, float, float]:
    """The digit `ink` (cropped to its own extent) looks most like.

    `on_line` marks the rows a staff line runs through. Those rows say nothing
    about the digit -- the line fills them whatever is printed there -- so they
    are left out of the comparison rather than scrubbed out of the image, which
    at this resolution takes the digit's own thin strokes with it.

    Only digits in `allowed` are candidates, so a digit that cannot stand here
    is neither chosen nor counted as the runner-up.

    Returns the digit, its score, and the score of the best other digit.
    """
    height, width = ink.shape
    keep = np.ones(height, dtype=bool) if on_line is None else ~on_line
    if keep.sum() < 2 or width < 1:  # noqa: PLR2004
        return 0, 0.0, 0.0
    sample = ink.astype(np.float64)[keep]
    scores: dict[int, float] = {}
    for digit, shapes in _templates().items():
        if digit not in allowed:
            continue
        best = -1.0
        for shape in shapes:
            resized = cv2.resize(
                shape.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA
            ).astype(np.float64)
            best = max(best, _correlate(sample, resized[keep]))
        scores[digit] = best
    ranked = sorted(scores.items(), key=lambda item: -item[1])
    return ranked[0][0], ranked[0][1], ranked[1][1]


def _line_rows(staff: Staff, x0: int, x1: int, top: int) -> list[NDArray[np.float64]]:
    """Each staff line's row (relative to `top`) in every column from `x0` to `x1`."""
    xs = np.array([point.x for point in staff.grid], dtype=np.float64)
    order = np.argsort(xs)
    columns = np.arange(x0, x1, dtype=np.float64)
    return [
        np.interp(columns, xs[order], np.array([point.y[line] for point in staff.grid])[order])
        - top
        for line in range(5)
    ]


def _binarize(image: NDArray) -> NDArray[np.bool_]:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # noqa: PLR2004
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary > 0


def _without_lines(
    ink: NDArray[np.bool_], line_rows: list[NDArray[np.float64]], thickness: int
) -> NDArray[np.bool_]:
    """`ink` with the staff lines taken out where nothing crosses them.

    Column by column, a run of ink no taller than a staff line and lying on one
    is the line; a run that is taller is something drawn across it and stays.
    `line_rows` holds each line's row in every column, so a tilted staff is
    followed.
    """
    out = ink.copy()
    height, width = ink.shape
    limit = thickness + 1
    for x in range(width):
        column = ink[:, x]
        y = 0
        while y < height:
            if not column[y]:
                y += 1
                continue
            end = y
            while end < height and column[end]:
                end += 1
            if end - y <= limit:
                centre = (y + end - 1) / 2
                if any(abs(centre - rows[x]) <= thickness for rows in line_rows):
                    out[y:end, x] = False
            y = end
    return out


def _line_thickness(ink: NDArray[np.bool_], line_rows: list[NDArray[np.float64]]) -> int:
    """How many rows a staff line is drawn in: the commonest run of ink on one."""
    lengths: list[int] = []
    height, width = ink.shape
    for rows in line_rows:
        for x in range(0, width, 3):
            y = round(rows[x])
            if not 0 <= y < height or not ink[y, x]:
                continue
            first = y
            while first > 0 and ink[first - 1, x]:
                first -= 1
            last = y
            while last < height - 1 and ink[last + 1, x]:
                last += 1
            lengths.append(last - first + 1)
    if not lengths:
        return 1
    return max(1, int(np.bincount(lengths).argmax()))


def _runs(columns: NDArray[np.bool_] | np.bool_, gap: int) -> list[tuple[int, int]]:
    """The stretches of True in `columns`, joined across gaps of up to `gap`."""
    columns = np.atleast_1d(np.asarray(columns, dtype=bool))
    runs: list[tuple[int, int]] = []
    start = None
    last = -(10**9)
    for x, on in enumerate(columns):
        if not on:
            continue
        if start is None or x - last > gap + 1:
            if start is not None:
                runs.append((start, last + 1))
            start = x
        last = x
    if start is not None:
        runs.append((start, last + 1))
    return runs


def _read_number(
    raw: NDArray[np.bool_], clean: NDArray[np.bool_], on_line: NDArray[np.bool_], unit: float
) -> tuple[int, float] | None:
    """The number written in these rows, one or two digits side by side.

    `clean` (the lines taken out) says where the digits are; `raw` is what is
    compared, with the rows in `on_line` left out.
    """
    # Measured off the rows between the lines: a digit standing on the middle line
    # touches the digit across it there.
    body = clean[~on_line]
    columns = np.where(body.any(axis=0))[0]
    if len(columns) == 0:
        return None
    left, right = int(columns[0]), int(columns[-1]) + 1
    pieces = [(left, right)]
    # A digit is about as wide as it is tall over two; a number twice that wide
    # is two digits (12), cut where the gap between them is widest. Cutting at
    # every gap instead would cut a 3 whose thin strokes the scan lost.
    if right - left >= 2 * unit:
        gaps = _runs(~body[:, left:right].any(axis=0), gap=0)
        if not gaps:
            return None
        gap_left, gap_right = max(gaps, key=lambda gap: gap[1] - gap[0])
        pieces = [(left, left + gap_left), (left + gap_right, right)]
    value = 0
    weakest = 1.0
    for index, (left, right) in enumerate(pieces):
        # No time signature number starts with 0 (numerators run 1-16, the
        # denominators are 1, 2, 4, 8, 16 and 32). Leaving 0 out of a leading
        # digit matters because 0 is the digit a 6 looks most like: Lempilintu's
        # printed 6 scored 0.76 as a 6 and 0.72 as a 0, too close to be read,
        # and next best after 0 was 9 at 0.51 (eerovil/musescore-choir-plugins#274).
        allowed = _LEADING if index == 0 else _ANY_DIGIT
        digit, score, runner_up = classify_digit(raw[:, left:right], on_line, allowed)
        if score < _MIN_SCORE or score - runner_up < _MARGIN:
            return None
        value = value * 10 + digit
        weakest = min(weakest, score)
    return value, weakest


def find_time_signatures(image: NDArray, staff: Staff) -> list[PrintedMeter]:
    """Every time signature printed on `staff`, in page coordinates, left to right."""
    unit = float(staff.average_unit_size)
    if unit <= 0 or not staff.grid:
        return []
    x0 = int(max(0, staff.min_x))
    x1 = int(min(image.shape[1], staff.max_x))
    top = int(max(0, staff.min_y - 2 * unit))
    bottom = int(min(image.shape[0], staff.max_y + 2 * unit))
    if x1 - x0 < unit or bottom - top < unit:
        return []
    band = _binarize(image[top:bottom, x0:x1])
    lines = _line_rows(staff, x0, x1, top)
    thickness = _line_thickness(band, lines)
    clean = _without_lines(band, lines, thickness)
    height = clean.shape[0]
    rows = np.arange(height)[:, None]
    in_upper = (rows >= lines[0][None, :] - thickness) & (rows <= lines[2][None, :])
    in_lower = (rows >= lines[2][None, :]) & (rows <= lines[4][None, :] + thickness)
    upper = (clean & in_upper).any(axis=0)
    lower = (clean & in_lower).any(axis=0)
    gap = max(1, round(unit * 0.15))
    # A signature stands where both halves of the staff hold ink side by side.
    # Failing a reading there, the whole of the ink around it is tried, since a
    # two-digit numerator is wider than the digit under it (12 over 8); not
    # first, because on a crowded staff that run takes in the notes beside it.
    either = _runs(upper | lower, gap=gap)
    found: list[PrintedMeter] = []
    for run in _runs(upper & lower, gap=gap):
        whole = next(((a, b) for a, b in either if a <= run[0] and run[1] <= b), run)
        meter = None
        left, right = run
        for span in dict.fromkeys([run, whole]):
            if not unit * 0.8 <= span[1] - span[0] <= unit * 4.5:
                continue
            centre = (span[0] + span[1]) // 2
            line_rows = [float(line[centre]) for line in lines]
            meter = _read_stack(band, clean, line_rows, span[0], span[1], unit, thickness)
            if meter is not None:
                left, right = span
                break
        if meter is None:
            continue
        numerator, denominator, score = meter
        x = x0 + (left + right) / 2
        if found and found[-1].x == x:
            continue
        found.append(
            PrintedMeter(
                x=x,
                numerator=numerator,
                denominator=denominator,
                score=score,
            )
        )
    return found


def _refine_lines(
    raw: NDArray[np.bool_], rows: list[float], left: int, right: int, unit: float
) -> list[float]:
    """The staff lines' rows where they actually are around these columns.

    The staff grid is a fit across the whole staff and can be a few pixels off
    any one line, which is most of a line's thickness; each line is moved to the
    darkest row near where the grid puts it, measured either side of the symbol.
    """
    reach = int(unit * 2)
    width = raw.shape[1]
    sides = [raw[:, max(0, left - reach) : left], raw[:, right : min(width, right + reach)]]
    profile = sum((side.sum(axis=1) for side in sides if side.size), np.zeros(raw.shape[0]))
    if not np.any(profile):
        return rows
    # Five lines, evenly spaced: the spacing and the top line that put the most
    # ink under all five at once. Moving each line on its own lets one of them
    # jump to a digit's horizontal stroke.
    best = (-1.0, rows)
    for spacing in np.arange(unit * 0.85, unit * 1.15, 0.25):
        for first in np.arange(rows[0] - unit * 0.5, rows[0] + unit * 0.5, 0.5):
            candidate = [first + k * spacing for k in range(5)]
            if candidate[-1] >= len(profile) - 1 or candidate[0] < 0:
                continue
            score = float(sum(np.interp(candidate, np.arange(len(profile)), profile)))
            if score > best[0]:
                best = (score, candidate)
    return [float(row) for row in best[1]]


def _inside_staff(
    column: NDArray[np.bool_], rows: list[float], tolerance: float
) -> NDArray[np.bool_] | None:
    """The ink of these columns that belongs to the staff, or None if it does not stay there.

    A shape that reaches into the staff and out past its top or bottom line -- a
    stem, a ledger line -- is not a time signature. A shape wholly outside the
    staff (a tuplet number, a dynamic, a hairpin) is something else's and is
    left out.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        column.astype(np.uint8), connectivity=8
    )
    kept = np.zeros_like(column)
    for label in range(1, count):
        top = stats[label, cv2.CC_STAT_TOP]
        bottom = top + stats[label, cv2.CC_STAT_HEIGHT] - 1
        if bottom < rows[0] - tolerance or top > rows[4] + tolerance:
            continue
        if top < rows[0] - tolerance or bottom > rows[4] + tolerance:
            return None
        kept |= labels == label
    return kept if kept.any() else None


def _read_stack(
    raw: NDArray[np.bool_],
    clean: NDArray[np.bool_],
    rows: list[float],
    left: int,
    right: int,
    unit: float,
    thickness: int,
) -> tuple[int, int, float] | None:
    """The two numbers stacked in columns `left:right`, if that is what they are."""
    tolerance = unit * 0.45
    rows = _refine_lines(raw, rows, left, right, unit)
    column = _inside_staff(clean[:, left:right], rows, tolerance)
    if column is None:
        return None
    middle = round(rows[2])
    numbers = []
    for first, last, start, stop in (
        (rows[0], rows[2], 0, middle + 1),
        (rows[2], rows[4], middle, column.shape[0]),
    ):
        filled = np.where(column[start:stop].any(axis=1))[0] + start
        if len(filled) == 0:
            return None
        top, bottom = int(filled[0]), int(filled[-1])
        if abs(top - first) > tolerance or abs(bottom - last) > tolerance:
            return None
        # The digit is as tall as the two spaces it fills, whatever the scan kept
        # of its outline: its edges are the outer edges of the lines bounding it.
        top = max(0, round(first - thickness / 2))
        bottom = min(column.shape[0] - 1, round(last + thickness / 2))
        here = np.arange(top, bottom + 1)
        on_line = np.zeros(len(here), dtype=bool)
        for line in rows:
            on_line |= np.abs(here - line) <= thickness / 2 + 0.5
        number = _read_number(
            raw[top : bottom + 1, left:right], column[top : bottom + 1], on_line, unit
        )
        if number is None:
            return None
        numbers.append(number)
    (numerator, top_score), (denominator, bottom_score) = numbers
    if denominator not in _DENOMINATORS or not 1 <= numerator <= _MAX_NUMERATOR:
        return None
    if numerator == 1 and denominator == 1:
        # A 1 is a vertical stroke, so a 1 over a 1 is any stroke crossing both
        # halves of the staff -- a barline, a stem -- and 1/1 is a meter nobody
        # prints. Read as one, it put a 1/1 the page does not have into three
        # songs of eerovil/musescore-choir-plugins#274.
        return None
    return numerator, denominator, min(top_score, bottom_score)


def printed_staffs(staff: Staff) -> list[Staff]:
    """The five-line staffs a staff the model reads as one is made of."""
    merged = getattr(staff, "merged_from", None)
    if merged:
        return [part for whole in merged for part in printed_staffs(whole)]
    return [staff]


def _is_timed(symbol: EncodedSymbol) -> bool:
    return symbol.rhythm.startswith(("note", "rest"))


def attach_printed_meters(
    symbols: list[EncodedSymbol],
    image: NDArray,
    staff: Staff,
    page_to_input_image: PointMapping,
) -> list[EncodedSymbol]:
    """Put the digits printed on `staff` onto the time signatures in its decoded stream.

    The decoder writes at most a `timeSignature/<denominator>` at the head of the
    system; a change of meter after a barline usually gets no token. So each
    signature found on the page is placed by its x against the decoded barlines:

    - one in the bar where the decoder already put a signature is attached to it;
    - one opening a bar that has none gets a signature token of its own there,
      with the denominator the page prints;
    - one after the last barline with nothing to sound after it is the courtesy
      signature for the next system, and is left out: that system prints its own.

    Each printed staff is read separately and both readings are kept, so the
    reconstruction can tell digits every staff agrees on from digits one staff
    alone showed (`printed_meter`).
    """
    if (
        not isinstance(staff, Staff) or not isinstance(image, np.ndarray) or image.ndim < 2
    ):  # noqa: PLR2004
        # Nothing to read the digits off: a caller that hands in no detected staff
        # or no page gets the stream back as decoded.
        return symbols
    found: list[tuple[float, PrintedMeter]] = []
    for part in printed_staffs(staff):
        for meter in find_time_signatures(image, part):
            y = float(np.mean(_line_rows(part, int(meter.x), int(meter.x) + 1, 0)[2]))
            found.append((page_to_input_image((meter.x, y))[0], meter))
    if not found:
        return symbols
    unit = float(staff.average_unit_size)
    reach = page_to_input_image((2 * unit, 0))[0] - page_to_input_image((0, 0))[0]
    # One signature printed on several staffs is one column of the system.
    columns: list[list[tuple[float, PrintedMeter]]] = []
    for x, meter in sorted(found, key=lambda item: item[0]):
        if columns and x - columns[-1][-1][0] <= reach:
            columns[-1].append((x, meter))
        else:
            columns.append([(x, meter)])

    barlines = [
        (index, symbol.image_coordinates[0])
        for index, symbol in enumerate(symbols)
        if symbol.rhythm.startswith("barline") and symbol.image_coordinates is not None
    ]
    out = list(symbols)
    insertions: list[tuple[int, EncodedSymbol]] = []
    for column in columns:
        x = float(np.mean([position for position, _ in column]))
        readings = tuple((meter.numerator, meter.denominator) for _, meter in column)
        # The bar this column opens starts after the last barline left of it.
        before = [index for index, position in barlines if position < x]
        start = before[-1] + 1 if before else 0
        end = next((index for index, _ in barlines if index >= start), len(out))
        if not any(_is_timed(symbol) for symbol in out[start:end]):
            eprint("Time signature", readings, "after the last bar: a courtesy, left out")
            continue
        existing = next(
            (symbol for symbol in out[start:end] if symbol.rhythm.startswith("timeSignature")),
            None,
        )
        if existing is not None:
            existing.printed_meters = readings
            eprint("Time signature", existing.rhythm, "reads", readings, "on the page")
            continue
        denominators = {denominator for _, denominator in readings}
        if len(denominators) != 1:
            continue
        token = EncodedSymbol(f"timeSignature/{denominators.pop()}")
        token.printed_meters = readings
        at = start
        while at < end and out[at].rhythm.startswith(("clef", "keySignature")):
            at += 1
        insertions.append((at, token))
        eprint("Time signature", readings, "printed at a barline the model read none at")
    for index, token in sorted(insertions, key=lambda item: -item[0]):
        out.insert(index, token)
    return out
