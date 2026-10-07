"""Find the "1." and "2." brackets over an end repeat when the decoder did not write them.

A volta bracket stands high above the top staff of a system -- on Shakkitarina
(eerovil/musescore-choir-plugins#319) about seven staff spaces above the top line,
above the chord names -- and the staff image the decoder reads keeps only four
spaces above a staff, so it never sees one. None of the scans on the choir host
had a single ``voltaStart``. Without the brackets the score plays the bars under
"1." on both passes and never skips to "2.".

The bracket is easy to see in the pixels: a thin horizontal line, at least a few
staff spaces long, with a short stroke hanging down from its left end. That hook
is what tells it from a lyric extender line, and the line's thinness from a beam
with its stems. So this looks, between the staff above and a staff space and a
half over this one, for such a line crossing an end repeat the decoder read:

- the "1." bracket ends at the repeat's barline, give or take a staff space and a
  half, and starts somewhere to the left of it in the same staff;
- the "2." bracket, if one is there, starts just right of that barline.

The repeat is placed by the decoder's own position for its token, which lands
within a few pixels of the printed barline; the other barline tokens can drift by
a bar's width, and the page's barline detection can miss a thick repeat barline
on the top staff (Shakkitarina bar 13) or count a start repeat at the head of the
system as a barline (bar 1). So a bracket's length in bars is counted only from
the page's barlines standing *inside* it, and it is laid back from the repeat:
neither mistake reaches across a bracket. The "2." bracket is
closed with ``voltaStop`` when it has a hook at its right end too and
``voltaDiscontinue`` when it is left open, as the page draws it. Nothing is added
on a staff where the decoder already wrote a volta token.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from homr.brace_dot_detection import interior_bar_lines
from homr.model import Staff
from homr.point_mapping import PointMapping
from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray

#: Shortest a bracket line is, in staff spaces.
_MIN_LENGTH = 3.0
#: Thickest a bracket line is, in staff spaces. A beam is about half a space.
_MAX_THICK = 0.3
#: Shortest the hook hanging down from a bracket's end is, in staff spaces.
_MIN_HOOK = 0.8
#: Widest a bracket is, as a share of the staff's width: no wider is a staff line.
_MAX_WIDTH = 0.8
#: How far from a barline a bracket may end or start and still be about it.
_NEAR_BAR = 1.5
#: How far above the top staff line a bracket is looked for, at most, in spaces.
_REACH = 12.0
#: How close to the top staff line it may come, in staff spaces.
_CLEARANCE = 1.5
#: How many pixels a line may step up or down across its length (a slight tilt).
_ROW_GAP = 1

#: Tokens that close a bar. A start repeat after music closes one too, as the
#: generator reads it: Shakkitarina bar 5 ends ``||:`` with no barline token.
_BAR_ENDS = (
    "barline",
    "doublebarline",
    "bolddoublebarline",
    "repeatEnd",
    "repeatEndStart",
    "repeatStart",
)


@dataclass
class Bracket:
    left: float
    right: float
    y: float
    left_hook: bool
    right_hook: bool


def _lines(ink: NDArray, unit: float) -> list[tuple[int, int, int, int]]:
    """Long thin horizontal runs as ``(top, bottom, left, right)``, inclusive."""
    min_length = _MIN_LENGTH * unit
    runs: list[tuple[int, int, int]] = []
    for y in range(ink.shape[0]):
        row = np.concatenate(([0], ink[y].astype(np.int8), [0]))
        edges = np.flatnonzero(np.diff(row))
        for start, stop in zip(edges[::2], edges[1::2], strict=True):
            if stop - start >= min_length:
                runs.append((y, int(start), int(stop) - 1))
    # Join a run to the line above it when they overlap: the line's thickness, or
    # a line drawn a pixel lower towards its far end.
    segments: list[list[int]] = []
    for y, left, right in runs:
        for segment in segments:
            if y - segment[1] <= _ROW_GAP and left <= segment[3] and right >= segment[2]:
                segment[1] = y
                segment[2], segment[3] = min(segment[2], left), max(segment[3], right)
                break
        else:
            segments.append([y, y, left, right])
    return [(top, bottom, left, right) for top, bottom, left, right in segments]


def _hook(ink: NDArray, below: int, x: int, unit: float) -> bool:
    """Whether a stroke hangs down from row ``below`` at column ``x``."""
    reach = max(1, round(0.3 * unit))
    columns = ink[below:, max(0, x - reach) : x + reach + 1]
    if columns.size == 0:
        return False
    inked = np.asarray(columns.any(axis=1), dtype=bool)
    gaps = np.flatnonzero(~inked)
    length = int(gaps[0]) if gaps.size else len(inked)
    return length >= _MIN_HOOK * unit


def brackets_above(gray: NDArray, staff: Staff, ceiling: float) -> list[Bracket]:
    """Volta-shaped lines between ``ceiling`` and the top of ``staff``, left to right."""
    unit = float(staff.average_unit_size)
    if unit <= 0:
        return []
    # Half a space clear of the staff above, whose bottom line is the ceiling.
    top = max(
        0, round(ceiling + 0.5 * unit if ceiling > 0 else 0), round(staff.min_y - _REACH * unit)
    )
    bottom = round(staff.min_y - _CLEARANCE * unit)
    left, right = max(0, round(staff.min_x)), min(gray.shape[1], round(staff.max_x + unit))
    if bottom - top < 2 or right - left < _MIN_LENGTH * unit:  # noqa: PLR2004
        return []
    region = gray[top:bottom, left:right]
    _, ink = cv2.threshold(region, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if ink.mean() > 0.5:  # noqa: PLR2004 -- a blank strip thresholds to noise
        return []
    found = []
    for line_top, line_bottom, x0, x1 in _lines(ink, unit):
        if line_bottom - line_top + 1 > _MAX_THICK * unit:
            continue
        if x1 - x0 > _MAX_WIDTH * (right - left):
            continue  # a staff line, not a bracket
        left_hook = _hook(ink, line_bottom + 1, x0, unit)
        if not left_hook:
            continue
        found.append(
            Bracket(
                left=float(left + x0),
                right=float(left + x1),
                y=float(top + (line_top + line_bottom) / 2),
                left_hook=True,
                right_hook=_hook(ink, line_bottom + 1, x1, unit),
            )
        )
    return sorted(found, key=lambda b: b.left)


def _center(symbol: EncodedSymbol) -> tuple[float, float] | None:
    if symbol.coordinates is None:
        return None
    center = np.asarray(symbol.coordinates, dtype=np.float64).reshape(-1)
    if len(center) < 2 or np.isnan(center[0]) or np.isnan(center[1]):  # noqa: PLR2004
        return None
    return float(center[0]), float(center[1])


def _bar_ends(symbols: list[EncodedSymbol]) -> list[int]:
    """Token indices of the barlines that close a bar holding a note or rest."""
    ends = []
    start = 0
    for index, symbol in enumerate(symbols):
        if symbol.rhythm in _BAR_ENDS:
            if any(s.rhythm.startswith(("note", "rest")) for s in symbols[start:index]):
                ends.append(index)
            start = index + 1
    return ends


def _first_music(symbols: list[EncodedSymbol], start: int) -> int:
    """Index of the first note, rest or chord at or after ``start``."""
    for index in range(start, len(symbols)):
        if symbols[index].rhythm.startswith(("note", "rest", "chord")):
            return index
    return len(symbols)


def add_missing_voltas(
    symbols: list[EncodedSymbol],
    staff: Staff,
    gray: NDArray,
    ceiling: float,
    to_page: PointMapping,
) -> int:
    """Insert volta tokens over each end repeat the page brackets. Returns how many brackets.

    ``symbols`` carry coordinates in the staff image and ``to_page`` takes them to
    the page, where ``staff`` and ``gray`` are; ``ceiling`` is the bottom of the
    staff above, or 0.
    """
    if any(symbol.rhythm.startswith("volta") for symbol in symbols):
        return 0
    ends = _bar_ends(symbols)
    repeats = [i for i, at in enumerate(ends) if symbols[at].rhythm.startswith("repeatEnd")]
    if not repeats:
        return 0
    brackets = brackets_above(gray, staff, ceiling)
    if not brackets:
        return 0
    unit = float(staff.average_unit_size)
    near = _NEAR_BAR * unit
    barlines = interior_bar_lines(staff)

    def bars_under(bracket: Bracket) -> int:
        """How many bars a bracket spans: the page's barlines inside it, plus one."""
        return 1 + sum(1 for x in barlines if bracket.left + near < x < bracket.right - near)

    inserts: list[tuple[int, int, str]] = []  # (token index, order, rhythm)
    added = 0
    for bar in repeats:
        # The repeat token's own position: the decoder places a barline it read
        # well, and the page's barline detection can miss a thick one.
        center = _center(symbols[ends[bar]])
        if center is None:
            continue
        at = to_page(center)[0]
        first = next(
            (b for b in brackets if b.left < at - unit and abs(b.right - at) <= near), None
        )
        if first is None:
            continue
        opens = bar - bars_under(first) + 1
        if opens < 0:
            continue
        start = ends[opens - 1] + 1 if opens > 0 else _first_music(symbols, 0)
        inserts.append((start, 0, "voltaStart"))
        inserts.append((ends[bar], 0, "voltaStop"))
        added += 1
        second = next((b for b in brackets if b is not first and abs(b.left - at) <= near), None)
        if second is None or _first_music(symbols, ends[bar] + 1) == len(symbols):
            continue
        last = bar + bars_under(second)
        close_at = ends[last] if last < len(ends) else len(symbols)
        inserts.append((ends[bar] + 1, 1, "voltaStart"))
        inserts.append((close_at, 0, "voltaStop" if second.right_hook else "voltaDiscontinue"))
        added += 1
    # Insert from the back so earlier indices stay put; at one index the lower
    # order lands first.
    for index, _, rhythm in sorted(inserts, key=lambda t: (t[0], t[1]), reverse=True):
        symbols.insert(index, EncodedSymbol(rhythm))
    return added


def log_added(index: int, count: int) -> None:
    eprint("Staff", index, "has", count, "volta bracket(s) the decoder did not write: added")
