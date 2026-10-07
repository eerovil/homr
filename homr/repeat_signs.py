"""Find a start-repeat sign that opens a staff when the decoder did not write one.

A start repeat printed at the head of a system -- clef, key, time, then the
thick and thin barline with two dots -- is often missing from the decoder's
output: it reads the clef and the key and goes straight on to the first note
(eerovil/musescore-choir-plugins#312: Vieläkö huvittaisi bar 46 and Kantajani
bar 11 on every staff, Kantajani bar 27 on two staves of four). The same sign in
the middle of a staff is read. Without it the score repeats from the wrong bar,
which a singer hears as the practice track going back to the wrong place.

The sign is easy to see in the pixels, and what makes it a repeat rather than a
plain barline is the **two dots**, one in each of the two middle spaces of the
staff, just right of the barline. So this looks, between the signature the
decoder read and the first note it read, for:

- a barline with a thick stroke: a run of columns inked from the top staff line
  to the bottom one, at least a quarter of a staff space wide,
- and right of it, within a staff space and a half, two small round blobs, one
  centred in the second space from the top and one in the third, side by side,
- with nothing in the outer two spaces beside them. That last rule is what keeps
  a time signature out: with the staff lines taken away, the right-hand edges of
  a ``2/2`` fall apart into one small blob per space, all four of them.

Only the head of a staff is looked at, and only when the decoder wrote nothing
there but a clef, key and time: a start repeat is inserted where the decoder
already wrote one nowhere near it, never moved or doubled. A grand staff fused
into one image is checked on its upper five lines.
"""

import cv2
import numpy as np

from homr.model import Staff
from homr.point_mapping import PointMapping
from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray

#: Tokens that may stand before a start repeat at the head of a staff. A grand
#: staff read as one image opens ``clef, chord, clef, key``: the ``chord`` says
#: the second clef stands beside the first, and has no place of its own.
_HEAD = ("clef", "keySignature", "timeSignature", "chord")
#: Share of a column between the top and bottom staff lines that must be ink
#: for it to be a barline column.
_BAR_FILL = 0.85
#: How far right of the barline the dots may start, in staff spaces.
_DOT_REACH = 1.6
#: Smallest and largest a dot may be across, in staff spaces.
_DOT_MIN, _DOT_MAX = 0.2, 0.75
#: How far a dot's centre may sit from the middle of its space, in staff spaces.
_DOT_OFF_CENTRE = 0.3
#: How far apart the two dots may stand sideways, in staff spaces.
_DOT_SIDEWAYS = 0.4
#: How wide the thick stroke of the sign is at least, in staff spaces.
_THICK = 0.25
#: Share of a row inked that makes it a staff line rather than a dot.
_LINE_ROW = 0.8


def _center(symbol: EncodedSymbol) -> tuple[float, float] | None:
    if symbol.coordinates is None:
        return None
    center = np.asarray(symbol.coordinates, dtype=np.float64).reshape(-1)
    if len(center) < 2 or np.isnan(center[0]) or np.isnan(center[1]):  # noqa: PLR2004
        return None
    return float(center[0]), float(center[1])


def _head(symbols: list[EncodedSymbol]) -> int:
    """How many tokens open the staff with a clef, key or time signature."""
    count = 0
    for symbol in symbols:
        if not symbol.rhythm.startswith(_HEAD):
            break
        count += 1
    return count


def start_repeat_at(gray: NDArray, staff: Staff, x_from: float, x_to: float) -> bool:
    """Whether a start-repeat sign stands on ``staff`` between two x positions.

    ``gray`` is the page as a grayscale image with dark ink, ``staff`` in its
    coordinates.
    """
    point = staff.get_at((x_from + x_to) / 2)
    if point is None or len(point.y) < 5 or x_to <= x_from:  # noqa: PLR2004
        return False
    lines = sorted(point.y)[:5]
    unit = float(np.mean(np.diff(lines)))
    if unit <= 0:
        return False
    top, bottom = round(lines[0]), round(lines[4])
    left, right = max(0, round(x_from)), min(gray.shape[1], round(x_to))
    if right - left < 3 or top < 0 or bottom >= gray.shape[0]:  # noqa: PLR2004
        return False
    region = gray[top : bottom + 1, left:right]
    _, ink = cv2.threshold(region, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    columns = ink.mean(axis=0) >= _BAR_FILL
    if not columns.any() or _longest_run(columns) < _THICK * unit:
        return False
    # The rightmost barline column is the thin line of the sign; the dots follow it.
    bar_right = left + int(np.flatnonzero(columns)[-1]) + 1

    reach = min(gray.shape[1], round(bar_right + _DOT_REACH * unit))
    window = gray[top : bottom + 1, bar_right:reach]
    if window.shape[1] < 2:  # noqa: PLR2004
        return False
    _, dots = cv2.threshold(window, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Take the staff lines out, or they join the dots into one shape with them:
    # where the staff says they are, and any row inked nearly all the way across,
    # since after dewarping a line can sit a few pixels off its estimate.
    thickness = max(1, round(unit * 0.12))
    for line in lines:
        y = round(line) - top
        dots[max(0, y - thickness) : y + thickness + 1, :] = 0
    dots[(dots > 0).mean(axis=1) >= _LINE_ROW, :] = 0
    count, _, stats, centroids = cv2.connectedComponentsWithStats(dots.astype(np.uint8))
    found: dict[int, list[float]] = {0: [], 1: [], 2: [], 3: []}
    for label in range(1, count):
        width, height = stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
        if width < _DOT_MIN * unit or height < _DOT_MIN * unit:
            continue
        if width > 2 * height:
            continue  # a piece of staff line the removal above missed
        x, y = centroids[label]
        if width > _DOT_MAX * unit or height > _DOT_MAX * unit:
            # Too big for a dot, but ink beside them all the same.
            found[0 if y < (lines[2] - top) else 3].append(float(x))
            continue
        for space in range(4):
            middle = (lines[space] + lines[space + 1]) / 2 - top
            if abs(y - middle) <= _DOT_OFF_CENTRE * unit:
                found[space].append(float(x))
    near = _DOT_SIDEWAYS * unit
    for upper in found[1]:
        for lower in found[2]:
            if abs(upper - lower) > near:
                continue
            middle = (upper + lower) / 2
            if not any(abs(x - middle) <= near for x in found[0] + found[3]):
                return True
    return False


def _longest_run(columns: NDArray) -> int:
    longest = current = 0
    for inked in columns:
        current = current + 1 if inked else 0
        longest = max(longest, current)
    return longest


def add_missing_start_repeat(
    symbols: list[EncodedSymbol],
    staff: Staff,
    gray: NDArray,
    to_page: PointMapping,
) -> bool:
    """Insert a ``repeatStart`` after the staff's opening signatures if the page prints one.

    ``symbols`` carry coordinates in the staff image; ``to_page`` takes them to
    the page, where ``staff`` and ``gray`` are.
    """
    head = _head(symbols)
    if head == 0 or head >= len(symbols):
        return False
    following = symbols[head]
    if following.rhythm.startswith(("repeat", "barline", "volta")):
        return False
    signatures = [symbol for symbol in symbols[:head] if symbol.rhythm != "chord"]
    if not signatures or signatures[0].rhythm.startswith(("keySignature", "timeSignature")):
        return False
    starts = [_center(symbol) for symbol in signatures]
    end = _center(following)
    if end is None or any(start is None for start in starts):
        return False
    x_from = max(to_page(start)[0] for start in starts if start is not None)
    x_to = to_page(end)[0] - 0.4 * staff.average_unit_size
    if not start_repeat_at(gray, staff, x_from, x_to):
        return False
    symbols.insert(head, EncodedSymbol("repeatStart"))
    return True


def log_added(index: int) -> None:
    eprint("Staff", index, "opens with a start repeat the decoder did not write: added")
