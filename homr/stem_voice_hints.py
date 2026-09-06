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
#: How far from a decoded note a head may sit and still be in its column. Wide,
#: because this is only ever used to look at a pair that is already known to be
#: one moment on one staff.
_COLUMN_REACH = 60.0
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


def pitch_at(position: int, clef: tuple[str, int]) -> str | None:
    """The pitch a staff position spells, given the clef -- the inverse of above."""
    sign, line = clef
    reference_step, reference_octave = _CLEF_REFERENCE[sign]
    note = position - (2 * line - 1) + _diatonic(reference_step, reference_octave)
    if note < 0:
        return None
    return f"{_STEPS[note % 7]}{note // 7}"


def _columns(
    symbols: list[EncodedSymbol], clefs: list[tuple[str, int] | None]
) -> list[list[int]]:
    """The decoded notes of one staff, grouped into the moments they sound at.

    Grouped by how close the notes are, not by a grid: two notes of one moment
    sit a fraction of a pixel apart and a fixed bucket puts them either side of
    its boundary. This one did -- 184.0 and 184.1 became bucket 11 and bucket
    12 -- and the pair this exists for was never looked at.
    """
    per_staff: dict[str, list[tuple[float, int]]] = {}
    for index, symbol in enumerate(symbols):
        if not symbol.rhythm.startswith("note"):
            continue
        coordinates = _note_coordinates(symbol)
        if coordinates is None or clefs[index] is None:
            continue
        per_staff.setdefault(symbol.position, []).append((coordinates[0], index))
    columns: list[list[int]] = []
    for places in per_staff.values():
        group: list[int] = []
        last: float | None = None
        for x, index in sorted(places):
            if last is not None and x - last > _MATCH_X_TOLERANCE:
                columns.append(group)
                group = []
            group.append(index)
            last = x
        if group:
            columns.append(group)
    return columns


def rescue_duplicate_pitches(symbols: list[EncodedSymbol], notes: list[Note]) -> int:
    """Re-pitch a note that duplicates its neighbour and is about to be deleted.

    The decoder can read one head of a close-set pair as the other's pitch. The
    two decoded notes are then identical, `_remove_duplicated_piches` keeps one
    -- correctly, since a chord cannot hold the same pitch twice -- and a note
    the page prints vanishes from the score entirely. On Heraa Suomi's fourth
    system the page has a C and an A a third apart, the segmentation finds both
    heads with the right positions and opposite stems, and the output has only
    the C.

    A dropped note is the worst kind of fault here: a wrong stem or a wrong voice
    still writes the note and can be argued with afterwards, and nothing
    downstream can recover a note that is not there.

    So where two decoded notes of one moment on one staff carry the same pitch,
    and the segmentation found exactly two heads there at two different
    positions, and one of those positions is the pitch they agree on, the other
    note is re-pitched to the head nobody claimed. Deliberately no wider than
    that: it only ever acts where a note is about to be lost, so the worst it can
    do is replace a deletion with a wrong pitch, and every other reading is left
    exactly as the decoder produced it.
    """
    clefs = _clefs_in_force(symbols)
    rescued = 0
    for members in _columns(symbols, clefs):
        if len(members) != 2:
            continue
        first, second = (symbols[i] for i in members)
        if first.pitch != second.pitch:
            continue
        x = _note_coordinates(first)[0]
        clef = clefs[members[0]]
        claimed = expected_position(first.pitch, clef)
        here = [
            note for note in notes
            if abs(note.center[0] - x) <= _MATCH_X_TOLERANCE
            and abs(note.center[1] - _note_coordinates(first)[1]) <= _COLUMN_REACH
            and abs(note.center[1] - _note_coordinates(second)[1]) <= _COLUMN_REACH
        ]
        positions = sorted({note.position for note in here})
        if len(positions) != 2 or claimed not in positions:
            continue
        other = positions[0] if positions[1] == claimed else positions[1]
        spelled = pitch_at(other, clef)
        if spelled is None:
            continue
        # The lower head takes the lower pitch: re-pitch whichever of the two is
        # drawn on the side the free position sits.
        target = second if (other < claimed) == (
            _note_coordinates(second)[1] > _note_coordinates(first)[1]) else first
        target.pitch = spelled
        # And the stem the head carries. Without it the rescued note has nothing
        # to say which line it belongs to and lands in whichever voice the chord
        # is assigned -- a note present but in the wrong part, which is what the
        # first version of this produced.
        owner = next((note for note in here if note.position == other), None)
        if owner is not None and len(owner.stem_directions) == 1:
            target.stem_direction = (
                "up" if owner.stem_directions[0] == StemDirection.UP else "down")
        rescued += 1
    return rescued


def pair_unison_stems(symbols: list[EncodedSymbol], notes: list[Note]) -> int:
    """Give each head of a two-head unison its own stem, so both survive.

    An engraver may draw a unison as one head with two stems -- which
    ``SHARED`` carries -- or as two heads side by side at the same staff
    position, one open with its stem one way and one filled with its stem the
    other. The second shape reaches here intact: the segmentation finds both
    heads and the decoder emits both notes. What loses one is that both decoded
    notes match the *same* head (they claim one position and their attention
    points are a pixel apart in y), so both take the same stem, and
    ``_remove_duplicated_piches`` then reads them as one note written twice.

    So where a column holds two decoded notes of one pitch and the segmentation
    found exactly two heads at that position carrying opposite stems, the two
    are matched one to one, left to right. That is the only reliable axis: the
    attention point is trustworthy across the staff and not up it.
    """
    clefs = _clefs_in_force(symbols)
    paired = 0
    for column in _columns(symbols, clefs):
        # Within the moment, only the notes of one pitch: the rest of the chord
        # shares the column and says nothing about this pair.
        same_pitch: dict[str, list[int]] = {}
        for index in column:
            same_pitch.setdefault(symbols[index].pitch, []).append(index)
        for members in same_pitch.values():
            if len(members) != 2:
                continue
            first, second = (symbols[i] for i in members)
            clef = clefs[members[0]]
            position = expected_position(first.pitch, clef)
            if position is None:
                continue
            paired += _pair(first, second, notes, position)
    return paired


def _pair(
    first: EncodedSymbol, second: EncodedSymbol, notes: list[Note], position: int
) -> int:
    """Match the two notes to the two heads, left to right, and return 1 if done."""
    x, y = _note_coordinates(first)
    here = [
        note
        for note in notes
        if abs(note.center[0] - x) <= _MATCH_X_TOLERANCE
        and note.position == position
        and abs(note.center[1] - y) <= _COLUMN_REACH
    ]
    if len(here) != 2 or any(len(note.stem_directions) != 1 for note in here):
        return 0
    if here[0].stem_directions[0] == here[1].stem_directions[0]:
        return 0
    heads = sorted(here, key=lambda note: note.center[0])
    tokens = sorted((first, second), key=lambda s: _note_coordinates(s)[0])
    for token, head in zip(tokens, heads, strict=True):
        token.stem_direction = "up" if head.stem_directions[0] == StemDirection.UP else "down"
    return 1


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
