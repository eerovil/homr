"""Read a decoded note's stem off the notehead the segmentation found for it.

The decoder reports where it was attending when it emitted each note, and
segmentation reports every physical notehead with the stems drawn on it.  Both
are in the coordinates of the staff image the model was given, so a decoded note
is matched to the nearest notehead outright.  A hint is emitted only where that
match is unambiguous and the notehead carries exactly one stem; a head carrying
both is two voices meeting, which is recorded as such and left to the voice
rebalancer rather than read as a voice.
"""

from numbers import Real

from homr.model import Note, StemDirection
from homr.transformer.vocabulary import EncodedSymbol

_MATCH_X_TOLERANCE = 12.0
_MATCH_Y_TOLERANCE = 16.0
#: How much closer the nearest notehead must be than the next one.
_MARGIN = 16.0
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


def _nearest(notes: list[Note], x: float, y: float) -> Note | None:
    """The one notehead this decoded note is at, or nothing if two are as close."""
    matches = sorted(
        (
            (note.center[0] - x) ** 2 + (note.center[1] - y) ** 2,
            index,
            note,
        )
        for index, note in enumerate(notes)
        if abs(note.center[0] - x) <= _MATCH_X_TOLERANCE
        and abs(note.center[1] - y) <= _MATCH_Y_TOLERANCE
    )
    if not matches:
        return None
    if len(matches) > 1 and matches[1][0] - matches[0][0] < _MARGIN:
        return None
    return matches[0][2]


def add_stem_voice_hints(symbols: list[EncodedSymbol], notes: list[Note]) -> int:
    """Set ``stem_direction`` on safely matched decoded notes and return its count."""
    hinted = 0
    for symbol in symbols:
        if not symbol.rhythm.startswith("note"):
            continue
        coordinates = _note_coordinates(symbol)
        if coordinates is None:
            continue
        note = _nearest(notes, *coordinates)
        if note is None or not note.stem_directions:
            continue
        if len(note.stem_directions) > 1:
            # Two voices meeting on one printed notehead.  Which voice this note
            # is cannot be read off a head that carries both stems, but that the
            # staff has two of them can.
            symbol.stem_direction = SHARED
            continue
        direction = note.stem_directions[0]
        symbol.stem_direction = "up" if direction == StemDirection.UP else "down"
        hinted += 1
    return hinted
