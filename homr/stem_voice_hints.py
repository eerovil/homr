"""Read a decoded note's stem off the notehead the segmentation found for it.

The decoder reports where it was attending when it emitted each note, and
segmentation reports every physical notehead with the stems drawn on it.  Both
are in the coordinates of the staff image the model was given, so a decoded note
is matched to the nearest notehead outright.  A hint is emitted only where that
match is unambiguous and the notehead carries exactly one stem; a head carrying
both is two voices meeting, which is recorded as such and left to the voice
rebalancer rather than read as a voice.
"""

import re
from numbers import Real

from homr.model import Note, StemDirection
from homr.transformer.vocabulary import EncodedSymbol

_MATCH_X_TOLERANCE = 12.0
#: The decoder's attention point sits along the stem rather than on the head, so
#: it is displaced away from the head in the direction the stem is drawn --
#: measured on laulun-aika-s2, 13 to 17px above an up-stem head and 3 to 8px
#: below a down-stem one. At 16 an up-stem head could fall outside its own note's
#: reach and get no hint at all, which is how a chord of two voices came to carry
#: one stem: no hint, no stem, nothing to say the chord was two lines. The margin
#: rule below is what keeps this from grabbing a neighbour.
_MATCH_Y_TOLERANCE = 24.0
#: How much closer the nearest notehead must be than the next one.
_MARGIN = 16.0
#: Written on a notehead that carries an up stem and a down stem at once.
SHARED = "both"
_STEPS = "CDEFGAB"
#: Where each clef sign puts its reference pitch: the note on the line the sign
#: names. G2 names G4 on line 2, F4 names F3 on line 4, C3 names C4 on line 3.
_CLEF_REFERENCE = {"G": ("G", 4), "F": ("F", 3), "C": ("C", 4)}
_CLEF = re.compile(r"^clef_([GFC])(\d)$")


def _diatonic(step: str, octave: int) -> int:
    return 7 * octave + _STEPS.index(step)


def expected_position(pitch: str, clef: tuple[str, int]) -> int | None:
    """Where a decoded pitch must sit on the staff, given the clef in force.

    Bottom line 1, one step per line or space -- the same numbering the detected
    noteheads carry, so the two can be compared outright.
    """
    match = re.match(r"^([A-G])([#b]*)(-?\d+)$", pitch)
    if match is None:
        return None
    step, _, octave = match.groups()
    sign, line = clef
    reference_step, reference_octave = _CLEF_REFERENCE[sign]
    return (
        2 * line - 1 + _diatonic(step, int(octave)) - _diatonic(reference_step, reference_octave)
    )


def _clefs_in_force(symbols: list[EncodedSymbol]) -> list[tuple[str, int] | None]:
    """The clef governing each symbol, tracked per staff of the group.

    A grand staff arrives as one stream with each symbol marked `upper` or
    `lower`, and each half has its own clef -- so tracking a single current clef
    would read every bass note against the treble.
    """
    current: dict[str, tuple[str, int]] = {}
    found: list[tuple[str, int] | None] = []
    for symbol in symbols:
        match = _CLEF.match(symbol.rhythm)
        if match is not None:
            current[symbol.position] = (match.group(1), int(match.group(2)))
        found.append(current.get(symbol.position))
    return found


def _note_coordinates(symbol: EncodedSymbol) -> tuple[float, float] | None:
    coordinates = symbol.coordinates
    if coordinates is None or len(coordinates) < 2:
        return None
    x, y = coordinates[:2]
    if not isinstance(x, Real) or not isinstance(y, Real):
        return None
    return float(x), float(y)


def _at_position(notes: list[Note], x: float, y: float, position: int) -> Note | None:
    """The notehead this decoded pitch names, found by where it must sit.

    The decoder reports where it was attending, and that is reliable across the
    staff and unreliable up it: measured, the attention point sits along the stem
    rather than on the head, and on one note of Sammon ryosto it landed 50px away
    in the gap between two staves, equidistant from heads on both. No tolerance
    can rescue that safely -- widening far enough to reach the right head reaches
    the wrong staff's too.

    But the decoded note says which pitch it is, and the clef says where that
    pitch sits, and the segmentation already recorded a position for every head.
    So the column is found by x, where the attention is trustworthy, and the head
    within it by the position the music demands. Where two staves offer a head at
    the same position, the nearer in y wins -- staves are far enough apart that
    even a 50px error picks the right one.
    """
    candidates = [
        note
        for note in notes
        if abs(note.center[0] - x) <= _MATCH_X_TOLERANCE and note.position == position
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda note: abs(note.center[1] - y))


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
    clefs = _clefs_in_force(symbols)
    for index, symbol in enumerate(symbols):
        if not symbol.rhythm.startswith("note"):
            continue
        coordinates = _note_coordinates(symbol)
        if coordinates is None:
            continue
        clef = clefs[index]
        position = expected_position(symbol.pitch, clef) if clef else None
        note = None
        if position is not None:
            note = _at_position(notes, *coordinates, position)
        if note is None:
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
