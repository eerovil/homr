"""Conservative geometry-based stem hints for decoded notes.

The transformer supplies approximate attention coordinates while segmentation
supplies physical noteheads and stems.  Their x coordinates use different
scales, so this module first recovers a per-staff affine calibration from
height-compatible pairs.  A hint is emitted only for a uniquely associated
notehead with exactly one detected stem direction.
"""

from itertools import combinations
from numbers import Real

from homr.model import Note, StemDirection
from homr.transformer.vocabulary import EncodedSymbol

_ANCHOR_Y_TOLERANCE = 10.0
_ANCHOR_X_TOLERANCE = 8.0
_MATCH_X_TOLERANCE = 12.0
_MATCH_Y_TOLERANCE = 16.0
_MIN_ANCHORS = 8
#: Written on a notehead that carries an up stem and a down stem at once.
SHARED = "both"


def _note_coordinates(symbol: EncodedSymbol) -> tuple[float, float] | None:
    coordinates = symbol.coordinates
    if coordinates is None or len(coordinates) < 2:
        return None
    x, y = coordinates[:2]
    if not isinstance(x, Real) or not isinstance(y, Real):
        return None
    return float(x), float(y)


def _calibrate(notes: list[Note], symbols: list[EncodedSymbol]) -> tuple[float, float] | None:
    candidates = [
        (note, coordinates)
        for note in notes
        for symbol in symbols
        if (coordinates := _note_coordinates(symbol)) is not None
        and abs(note.center[1] - coordinates[1]) <= _ANCHOR_Y_TOLERANCE
    ]
    best: tuple[tuple[int, float], float, float] | None = None
    for (left_note, left), (right_note, right) in combinations(candidates, 2):
        note_distance = right_note.center[0] - left_note.center[0]
        if abs(note_distance) < 20:
            continue
        scale = (right[0] - left[0]) / note_distance
        if not 0.8 <= scale <= 1.3:
            continue
        offset = left[0] - scale * left_note.center[0]
        inliers = [
            (note, coordinates)
            for note, coordinates in candidates
            if abs(scale * note.center[0] + offset - coordinates[0]) <= _ANCHOR_X_TOLERANCE
        ]
        score = (
            len(inliers),
            -sum(abs(scale * note.center[0] + offset - coordinates[0]) for note, coordinates in inliers),
        )
        if best is None or score > best[0]:
            best = score, scale, offset
    if best is None or best[0][0] < _MIN_ANCHORS:
        return None
    return best[1], best[2]


def add_stem_voice_hints(symbols: list[EncodedSymbol], notes: list[Note]) -> int:
    """Set ``stem_direction`` on safely matched decoded notes and return its count."""
    decoded_notes = [symbol for symbol in symbols if symbol.rhythm.startswith("note")]
    calibration = _calibrate(notes, decoded_notes)
    if calibration is None:
        return 0
    scale, offset = calibration
    hinted = 0
    for symbol in decoded_notes:
        coordinates = _note_coordinates(symbol)
        if coordinates is None:
            continue
        x, y = coordinates
        matches = sorted(
            (
                (
                    (scale * note.center[0] + offset - x) ** 2 + (note.center[1] - y) ** 2,
                    note,
                )
                for note in notes
                if abs(scale * note.center[0] + offset - x) <= _MATCH_X_TOLERANCE
                and abs(note.center[1] - y) <= _MATCH_Y_TOLERANCE
            ),
            key=lambda match: match[0],
        )
        if not matches:
            continue
        distance, note = matches[0]
        if len(matches) > 1 and matches[1][0] - distance < 16:
            continue
        if not note.stem_directions:
            continue
        if len(note.stem_directions) > 1:
            # Two voices meeting on one printed notehead.  Which voice this
            # note is cannot be read off a head that carries both stems, but
            # that the staff has two of them can.
            symbol.stem_direction = SHARED
            continue
        direction = note.stem_directions[0]
        symbol.stem_direction = "up" if direction == StemDirection.UP else "down"
        hinted += 1
    return hinted
