import cv2
import cv2.typing as cvt
import numpy as np

from homr import constants
from homr.bounding_boxes import BoundingEllipse, DebugDrawable, RotatedBoundingBox
from homr.model import Note, Staff, StemDirection
from homr.simple_logging import eprint
from homr.type_definitions import NDArray

# How far short of a notehead a stem may stop and still count as attached to
# it: beam and staff-line removal leaves real stems a little short.
ATTACHMENT_SLACK = 0.3
# How far past a stem's starting end a notehead may sit, in noteheads, and still
# be read as sharing that stem rather than as the neighbouring voice reaching for
# it. A second between two voices is half a notehead, so the room has to be less
# than that or the case this guards against walks straight through -- and it has
# to be some, because a stem's own end is read a pixel or two off its head.
# Measured over 80 real system crops: at 0.4 the three two-voice columns that
# report both directions keep on doing so, and at 0.1 seven heads that were read
# correctly flip, the up stem of an upper voice being taken for the lower one's.
OWNERSHIP_SLACK = 0.25
# How much narrower than its shoulders the ink has to get before it counts as
# the waist between two noteheads rather than one head's own rounded end.
WAIST_DEPTH = 0.8
# The longest a stem can be, in noteheads, when joining its broken pieces.
MAX_STEM_HEIGHT = 5.0
# How many times a chord's stem may be handed on from one notehead to the next.
CHORD_PASSES = 2
# How tall a hole in a stroke may be, in pixels, and still be read as the staff
# line that was taken out of it rather than as paper between two strokes. A
# staff line is three or four pixels of ink on these scans.
STAFF_LINE_BRIDGE = 7
# A notehead's own ink is thick top to bottom. Ink at the outer end of a clump
# thinner than this share of the clump's thickest column is something else that
# ran into it -- almost always the staff line the head sits on.
THIN_END_SHARE = 0.35
# The widest a notehead may be, in staff units, before `add_notes_to_staffs`
# throws it away. Trimming is gated on the same number so it can only rescue a
# head that was going to be discarded, and can never re-shape one that is
# already being read.
MAX_NOTEHEAD_WIDTH = 3.0


class NoteheadWithStem(DebugDrawable):
    def __init__(
        self,
        notehead: BoundingEllipse,
        stem: RotatedBoundingBox | None,
        stem_direction: StemDirection | None = None,
        stem_directions: list[StemDirection] | None = None,
    ):
        self.notehead = notehead
        self.stem = stem
        self.stem_direction = stem_direction
        self.stem_directions = stem_directions or ([] if stem_direction is None else [stem_direction])

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        self.notehead.draw_onto_image(img, color)
        if self.stem is not None:
            self.stem.draw_onto_image(img, color)


def adjust_bbox(bbox: cvt.Rect, noteheads: NDArray) -> cvt.Rect:
    region = noteheads[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    ys, _ = np.where(region > 0)
    if len(ys) == 0:
        # Invalid note. Will be eliminated with zero height.
        return bbox
    top = np.min(ys) + bbox[1] - 1
    bottom = np.max(ys) + bbox[1] + 1
    return (bbox[0], int(top), bbox[2], int(bottom))


def get_center(bbox: cvt.Rect) -> tuple[int, int]:
    cen_y = int(round((bbox[1] + bbox[3]) / 2))
    cen_x = int(round((bbox[0] + bbox[2]) / 2))
    return cen_x, cen_y


def shed_thin_ends(bbox: cvt.Rect, noteheads: NDArray, unit_size: float) -> cvt.Rect:
    """Trim ink at a wide clump's ends that is too thin to be a notehead.

    A head printed *on* a staff line is drawn through that line, and the
    segmentation sometimes keeps a run of it: two pixels tall, tens of pixels
    long, joined to the head and to nothing else. The clump then measures
    several noteheads wide, and `add_notes_to_staffs` throws away anything wider
    than three -- so the head goes with it, and a note the model found perfectly
    well never reaches the score.

    Cutting it as a waist is wrong: a waist needs a shoulder on both sides and a
    staff line has one only where it meets the head. What separates the two is
    thickness, not narrowing, so that is what is measured -- and only at the
    ends, so a genuine pair of heads side by side, both thick, is untouched.

    This is a last resort, applied to a box the splitting has already finished
    with and that is still too wide to be accepted. Two looser versions were
    measured over 80 real system crops first, and both recovered the same seven
    noteheads while also disturbing notes that were being read correctly --
    trimming every wide clump moved two heads by a staff position and flipped a
    stem, and trimming every clump wider than the limit still did, because a
    stack or a pair is legitimately wider than one head and tightening it moves
    where it splits. Rescuing a note nobody was going to get is worth a change;
    re-reading a note that was already right is not.
    """
    region = noteheads[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    if region.size == 0 or bbox[2] - bbox[0] <= MAX_NOTEHEAD_WIDTH * unit_size:
        return bbox
    columns = (region > 0).sum(axis=0)
    if not columns.any():
        return bbox
    thick = np.where(columns >= max(2.0, THIN_END_SHARE * columns.max()))[0]
    if len(thick) == 0:
        return bbox
    return (bbox[0] + int(thick[0]), bbox[1], bbox[0] + int(thick[-1]) + 1, bbox[3])


def check_bbox_size(bbox: cvt.Rect, noteheads: NDArray, unit_size: float) -> list[cvt.Rect]:
    """Split a clump of ink into noteheads, and rescue what is left too wide."""
    return [
        box
        if box[2] - box[0] <= MAX_NOTEHEAD_WIDTH * unit_size
        else shed_thin_ends(box, noteheads, unit_size)
        for box in _split_bbox(bbox, noteheads, unit_size)
    ]


def _split_bbox(bbox: cvt.Rect, noteheads: NDArray, unit_size: float) -> list[cvt.Rect]:
    note_w = constants.NOTEHEAD_SIZE_RATIO * unit_size
    note_h = unit_size
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    new_bbox: list[cvt.Rect] = []
    region = noteheads[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    columns = (region > 0).sum(axis=0).astype(float) if region.size else np.zeros(0)
    gaps = _blank_gaps(columns) if w > MAX_NOTEHEAD_WIDTH * unit_size else []
    waists = sorted(
        {column for column in [*gaps, *_waists(columns, note_w)] if 0 < column < w}
    )
    if waists:
        # Heads side by side: cut where the ink narrows between them, not down
        # the middle, so a notehead with a ledger line growing out of it is not
        # halved into two heads.
        cuts = [0, *waists, w]
        for left, right in zip(cuts, cuts[1:], strict=False):
            part = adjust_bbox((bbox[0] + left, bbox[1], bbox[0] + right, bbox[3]), noteheads)
            new_bbox.extend(_split_bbox(part, noteheads, unit_size))
    else:
        bands = split_stack(bbox, noteheads, note_h)
        if len(bands) <= 1:
            return bands
        # A band cut off a stack keeps the whole clump's width, so trim it to
        # its own ink and look across it again: three heads where one sits
        # beside a stacked pair are two cuts, not one.
        for band in bands:
            trimmed = _trim(band, noteheads)
            new_bbox.extend(
                _split_bbox(trimmed, noteheads, unit_size)
                if trimmed[2] - trimmed[0] < w
                else [trimmed]
            )

    return new_bbox


def _trim(bbox: cvt.Rect, noteheads: NDArray) -> cvt.Rect:
    """Shrink a box sideways onto the ink it holds."""
    region = noteheads[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    _, xs = np.where(region > 0)
    if len(xs) == 0:
        return bbox
    return (int(np.min(xs) + bbox[0]), bbox[1], int(np.max(xs) + bbox[0] + 1), bbox[3])


def split_stack(bbox: cvt.Rect, noteheads: NDArray, note_h: float) -> list[cvt.Rect]:
    """Cut a column of touching noteheads apart where the ink narrows.

    Two heads a third apart touch, and their blob is a figure of eight: wide,
    narrow, wide.  Dividing its height by a notehead and rounding gets that
    blob wrong -- 2.6 noteheads' worth of ink is two heads, not three -- so the
    cut is made at the waist instead, and the count comes out of the shape.
    Ink with no waist in it is one head, however tall it reads.
    """
    region = noteheads[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    if region.size == 0:
        return []
    profile = (region > 0).sum(axis=1).astype(float)
    waists = [row for row in _waists(profile, note_h) if row > 0]
    if not waists:
        # No waist: trust the height only when there is far too much of it for
        # one head, and fall back to the old even division.
        count = int(round(len(profile) / note_h))
        if count <= 1:
            return [bbox]
        step = len(profile) / count
        waists = [int(step * index) for index in range(1, count)]
    cuts = [0, *waists, len(profile)]
    return [
        adjust_bbox((bbox[0], bbox[1] + top, bbox[2], bbox[1] + bottom), noteheads)
        for top, bottom in zip(cuts, cuts[1:], strict=False)
        if bottom > top
    ]


def _blank_gaps(columns: NDArray) -> list[int]:
    """The middle of each stretch of blank paper with ink on both sides.

    `_waists` cannot find these, and the reason is worth writing down because it
    looks like a threshold problem and is not.  A waist needs a shoulder within
    one notehead on each side (`above`/`below` look back and forward
    `int(note_h)` columns); the two heads this rescues are 58 blank columns
    apart, so every column in between has a shoulder on one side and nothing but
    paper on the other, and `min(above, below) == 0` declines all of them.
    `WAIST_DEPTH` never gets a say.  Widening those windows instead would make
    every eighth column of the gap a cut and slice the blank into slivers.

    Measured on `laulun-aika-3-s3`, the clump at (1099, 118), 128px wide against
    a notehead's 16, with `unit` 12.8 and the limit 38.5::

        cols  0..5    0
        cols  6..20   1          a run of staff line
        cols 21..40   5..14      the upper voice's first notehead
        cols 41..98   0          blank paper
        cols 99..118  5..14      the upper voice's second notehead

    So the two heads are joined by no ink whatsoever inside this box: the
    segmentation's own component runs through a staff line a few rows below it,
    and the ellipse fitted over the group is too short to include that line while
    being wide enough to span both heads.  `add_notes_to_staffs` then threw the
    whole 128px clump away and both notes with it.

    Cutting blank paper is the one cut that cannot be wrong -- a notehead is a
    connected piece of ink, so no head has blank columns through the middle of
    it, and either side of the cut is ink that was there already.  It is still
    asked for only when the box is over-wide, on the same principle as
    `shed_thin_ends`: rescuing a note nobody was going to get is worth a change,
    re-shaping a note that is already being read is not.
    """
    ink = np.flatnonzero(columns > 0)
    if len(ink) < 2:
        return []
    return [
        int((left + right + 1) // 2)
        for left, right in zip(ink[:-1], ink[1:], strict=False)
        if right - left > 1
    ]


def _waists(profile: NDArray, note_h: float) -> list[int]:
    """The rows where a stack of noteheads pinches in between two of them."""
    room = max(2, int(note_h * 0.4))
    found: list[int] = []
    for row in range(room, len(profile) - room):
        window = profile[row - room : row + room + 1]
        if profile[row] > window.min():
            continue
        above = profile[max(0, row - int(note_h)) : row].max(initial=0)
        below = profile[row + 1 : row + 1 + int(note_h)].max(initial=0)
        # A waist has a shoulder on both sides and is clearly narrower than
        # both of them; a notehead's own rounded end is not.
        if min(above, below) == 0 or profile[row] > min(above, below) * WAIST_DEPTH:
            continue
        if found and row - found[-1] < note_h * 0.5:
            continue
        found.append(row)
    return found


def split_clumps_of_noteheads(
    notehead: NoteheadWithStem, noteheads: NDArray, staff: Staff
) -> list[NoteheadWithStem]:
    """
    Note heads might be clumped together by the notehead detection algorithm.
    """
    split_noteheads = split_notehead_ellipse(notehead.notehead, noteheads, staff.average_unit_size)
    if len(split_noteheads) <= 1:
        return [notehead]
    return [
        NoteheadWithStem(
            split_notehead,
            notehead.stem,
            notehead.stem_direction,
            notehead.stem_directions,
        )
        for split_notehead in split_noteheads
    ]


def split_notehead_ellipse(
    notehead: BoundingEllipse, noteheads: NDArray, unit_size: float
) -> list[BoundingEllipse]:
    bbox = [
        int(notehead.top_left[0]),
        int(notehead.top_left[1]),
        int(notehead.bottom_right[0]),
        int(notehead.bottom_right[1]),
    ]
    split_boxes = check_bbox_size(bbox, noteheads, unit_size)
    # One box back is usually the head unchanged -- but it can also be the head
    # with a staff line trimmed off it, and handing back the original ellipse
    # then throws the trim away and the note with it.
    if len(split_boxes) == 1 and list(split_boxes[0]) == bbox:
        return [notehead]
    if not split_boxes:
        return [notehead]
    result: list[BoundingEllipse] = []
    for box in split_boxes:
        center = get_center(box)
        size = (box[2] - box[0], box[3] - box[1])
        result.append(
            BoundingEllipse(
                (center, size, 0), notehead.contours, notehead.debug_id
            )
        )
    return result


def stem_direction(
    notehead: BoundingEllipse, stem: RotatedBoundingBox
) -> StemDirection | None:
    """Which way a stem points, by the side of the notehead it is drawn on.

    Standard engraving puts an up stem on the right of the head and a down stem
    on the left, so one printed head can carry both when two voices meet on it.
    """
    side_offset = notehead.size[0] * 0.1
    above = stem.center[1] < notehead.center[1]
    right = stem.center[0] >= notehead.center[0] + side_offset
    left = stem.center[0] <= notehead.center[0] - side_offset
    if above and right:
        return StemDirection.UP
    if not above and left:
        return StemDirection.DOWN
    return None


def is_attached(notehead: BoundingEllipse, stem: RotatedBoundingBox) -> bool:
    """Whether a stem meets this notehead rather than merely passing nearby.

    A stem runs alongside every notehead of its chord and past the last of
    them, so what it must do is reach this head's middle -- not begin there.
    The head below in a column fails that test against the head above's stem,
    which is the case the two rules have to tell apart.  The slack is for
    scans where beam and staff-line removal leaves the stem short.
    """
    if abs(stem.center[0] - notehead.center[0]) > notehead.size[0] * 0.75 + stem.size[0] / 2:
        return False
    slack = notehead.size[1] * ATTACHMENT_SLACK
    stem_top = stem.center[1] - stem.size[1] / 2
    stem_bottom = stem.center[1] + stem.size[1] / 2
    note_top = notehead.center[1] - notehead.size[1] / 2
    note_bottom = notehead.center[1] + notehead.size[1] / 2
    if stem_direction(notehead, stem) == StemDirection.UP:
        return stem_bottom >= note_top - slack and stem_top <= notehead.center[1]
    return stem_top <= note_bottom + slack and stem_bottom >= notehead.center[1]


def is_plausible_stem(notehead: BoundingEllipse, stem: RotatedBoundingBox) -> bool:
    """Reject tiny and horizontal fragments from the stems/rests segmentation class."""
    stem_width, stem_height = stem.size
    notehead_width, notehead_height = notehead.size
    return (
        stem_direction(notehead, stem) is not None
        # Low-resolution scans often break a real stem into the notehead
        # plus a remaining fragment about half a notehead high.  Side and
        # attachment checks below still reject nearby rests and barlines.
        and stem_height >= notehead_height * 0.5
        and stem_width <= notehead_width * 1.3
        and is_attached(notehead, stem)
    )


def _without_horizontal_ink(source_image: NDArray) -> tuple[NDArray, NDArray]:
    """The scan's ink, and the same with its long horizontal runs taken out."""
    ink = (source_image < 180).astype(np.uint8)
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, 12), np.uint8))
    return ink, ink & (1 - horizontal)


def vertical_ink(source_image: NDArray) -> NDArray:
    """The scan's ink with staff lines and beams taken out, for stem recovery."""
    _, kept = _without_horizontal_ink(source_image)
    return cv2.morphologyEx(kept, cv2.MORPH_CLOSE, np.ones((1, 12), np.uint8))


def bridged_ink(source_image: NDArray) -> NDArray:
    """The same ink, with the cut a staff line leaves across a stem put back.

    A stem crossing a staff line is the ordinary case, not the exception, and
    the removal above takes the crossing rows out of the stem as well as out of
    the line -- so a stem arrives in `vertical_ink` in pieces, each too short or
    too far from its notehead to be read as a stem.

    What is put back is decided by the scan, not guessed: a hole is filled only
    where ink survived both above and below it, and only with pixels the scan
    really has (`& ink`).  Where the gap is paper, the close fills nothing, so
    two separate strokes are never joined into one -- and no stroke can grow
    into white space, which is what sank every version of this that closed the
    ink vertically without asking the scan.
    """
    ink, kept = _without_horizontal_ink(source_image)
    joined = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, np.ones((STAFF_LINE_BRIDGE, 1), np.uint8))
    return kept | (joined & ink)


def _reach_towards_head(bridge: NDArray, column: int, near: int, edge: int, limit: int) -> int:
    """How far the mended ink carries this run's near end towards the notehead.

    Only towards, and never past the head's own edge, so this can lengthen a run
    onto its note and can never lengthen one away from it -- which is what an
    unanchored gap-joiner does, and it hands every stub of noise the length the
    floor below was keeping it short of.
    """
    step = 1 if edge > near else -1
    reached = near
    for offset in range(1, limit + 1):
        row = near + step * offset
        if (edge - row) * step < 0 or not 0 <= row < bridge.shape[0] or not bridge[row, column]:
            break
        reached = row
    return reached


def _longest_run(
    ink: NDArray,
    notehead: BoundingEllipse,
    columns: range,
    rows: range,
    direction: StemDirection,
    bridge: NDArray | None = None,
) -> RotatedBoundingBox | None:
    x, y = map(int, notehead.center)
    height = int(notehead.size[1])
    edge = y - height // 2 if direction == StemDirection.UP else y + height // 2
    best: tuple[int, int, int, int] | None = None
    for column in columns:
        if not 0 <= column < ink.shape[1]:
            continue
        values = ink[list(rows), column].astype(int)
        changes = np.diff(np.pad(values, (1, 1)))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        for start, end in zip(starts, ends, strict=True):
            length = int(end - start)
            top, bottom = rows[start], rows[end - 1]
            # Beam and staff-line removal can leave a short white gap
            # between a real source-image stem and its notehead.
            towards = 1 if direction == StemDirection.UP else -1

            def reaches(near: int, towards: int = towards) -> bool:
                return -max(2, height * 0.1) <= (edge - near) * towards <= height * 0.65

            near = bottom if direction == StemDirection.UP else top
            if not reaches(near) and bridge is not None:
                # The gap may be the staff line this stem crosses rather than
                # paper. Ask the mended ink, and only for a run that was going
                # to be thrown away: a run already reaching its head is left
                # exactly as it was read.
                near = _reach_towards_head(bridge, column, near, edge, int(max(2, height)))
                if direction == StemDirection.UP:
                    bottom = near
                else:
                    top = near
            attached = reaches(near)
            if attached and (best is None or length > best[0]):
                best = (length, column, top, bottom)
    if best is None or not height * 0.5 <= best[0] <= height * 5:
        return None
    _, column, top, bottom = best
    return RotatedBoundingBox(
        ((column, (top + bottom) / 2), (3, bottom - top + 1), 0), np.empty((0, 2))
    )


def source_stem_candidates(
    notehead: BoundingEllipse, ink: NDArray | None, bridge: NDArray | None = None
) -> list[RotatedBoundingBox]:
    """Recover a visibly printed stem when its segmentation class is empty."""
    if ink is None:
        return []
    x, y = map(int, notehead.center)
    width, height = map(int, notehead.size)
    found = [
        _longest_run(
            ink,
            notehead,
            range(x, x + width + 16),
            range(max(0, y - 80), y + 1),
            StemDirection.UP,
            bridge,
        ),
        _longest_run(
            ink,
            notehead,
            range(max(0, x - width - 16), x + 1),
            range(y, min(ink.shape[0], y + 81)),
            StemDirection.DOWN,
            bridge,
        ),
    ]
    return [
        candidate
        for candidate in found
        if candidate is not None
        and stem_direction(notehead, candidate) is not None
        and is_attached(notehead, candidate)
        and candidate.size[1] >= notehead.size[1] * 0.5
        and candidate.size[0] <= notehead.size[0] * 0.75
    ]


def on_the_stems_side(
    notehead: BoundingEllipse, stem: RotatedBoundingBox, direction: StemDirection | None
) -> bool:
    """Whether this head is on the side of the stem a head that owns it sits on.

    An up stem is drawn on the right of the head it rises from and a down stem
    on the left of the head it hangs from.  `stem_direction` asks the same
    question and then asks whether the stem runs away from the head as well,
    which is right for reading a direction and wrong for asking whose end this
    is: a stem's own end lies level with, and often inside, its notehead.
    """
    side_offset = notehead.size[0] * 0.1
    if direction == StemDirection.UP:
        return stem.center[0] >= notehead.center[0] + side_offset
    return stem.center[0] <= notehead.center[0] - side_offset


def belongs_to_another_notehead(
    notehead: BoundingEllipse,
    stem: RotatedBoundingBox,
    noteheads: list[BoundingEllipse],
) -> bool:
    """Whether a stem starts at some other notehead rather than at this one.

    A stem is drawn from one head: an up stem rises from the head at its bottom
    end, a down stem hangs from the head at its top end.  The rest of its length
    runs alongside the other heads of the same chord, which is why a head is not
    asked to *begin* the stem -- but it is asked to lie on the stem's own side of
    that starting head.  A head sitting past the starting end is a different
    voice's, sharing the column: the up stem on the right of the upper head and
    the down stem on the left of the lower head are two stems, not one head with
    two.

    Only a head that is really past the end, with another head sitting at that
    end, is refused.  A stem the segmentation cut short of its own head has no
    other head there to hand it to, so it stays where it was read.
    """
    stem_top = stem.center[1] - stem.size[1] / 2
    stem_bottom = stem.center[1] + stem.size[1] / 2
    direction = stem_direction(notehead, stem)
    if direction == StemDirection.UP:
        near_y = stem_bottom
        past_the_end = notehead.center[1] - near_y
    else:
        near_y = stem_top
        past_the_end = near_y - notehead.center[1]
    if past_the_end <= notehead.size[1] * OWNERSHIP_SLACK:
        return False
    return any(
        other is not notehead
        and abs(other.center[0] - stem.center[0]) <= other.size[0] * 0.8
        and abs(other.center[1] - near_y) <= other.size[1]
        # A head the stem is drawn on the wrong side of cannot be whose it is:
        # at a second the engraver moves one head across the stem, and that
        # displaced head sits at the stem's end without owning it.  Only the
        # side is asked, not `stem_direction`, because the ink at a stem's own
        # end overlaps its head and so is neither above nor below it.
        and on_the_stems_side(other, stem, direction)
        for other in noteheads
    )


def join_stem_fragments(
    stems: list[RotatedBoundingBox], unit: float
) -> list[RotatedBoundingBox]:
    """Put a stem back together where staff lines have cut it into pieces.

    The segmentation loses a stem's ink where a staff line crosses it, so a long
    stem arrives as two or three short boxes in one column -- each too short to
    be believed on its own.  Pieces that line up and nearly touch are one stem.
    Joining is repeated until nothing more joins, since a piece can bridge two
    that were too far apart to reach each other.
    """
    joined = list(stems)
    merged = True
    while merged:
        merged = False
        for index, stem in enumerate(joined):
            partner = _joinable(joined, index, stem, unit)
            if partner is None:
                continue
            other = joined[partner]
            top = min(other.center[1] - other.size[1] / 2, stem.center[1] - stem.size[1] / 2)
            bottom = max(other.center[1] + other.size[1] / 2, stem.center[1] + stem.size[1] / 2)
            joined[index] = RotatedBoundingBox(
                (
                    ((other.center[0] + stem.center[0]) / 2, (top + bottom) / 2),
                    (max(other.size[0], stem.size[0]), bottom - top),
                    0,
                ),
                np.empty((0, 2)),
            )
            joined.pop(partner)
            merged = True
            break
    return joined


def _joinable(
    stems: list[RotatedBoundingBox], index: int, stem: RotatedBoundingBox, unit: float
) -> int | None:
    """The piece of stem this one continues, if any."""
    if stem.size[0] > unit * 0.4:
        return None
    for other_index, other in enumerate(stems):
        if other_index == index or other.size[0] > unit * 0.4:
            continue
        if abs(other.center[0] - stem.center[0]) > max(other.size[0], stem.size[0], 3):
            continue
        top = min(other.center[1] - other.size[1] / 2, stem.center[1] - stem.size[1] / 2)
        bottom = max(other.center[1] + other.size[1] / 2, stem.center[1] + stem.size[1] / 2)
        # A stem is a few noteheads long; ink that keeps going past that is a
        # barline, or a stem joined to the next one down through a rest.
        if bottom - top > unit * MAX_STEM_HEIGHT:
            continue
        if (bottom - top) - (other.size[1] + stem.size[1]) <= unit * 0.5:
            return other_index
    return None


def stems_of_notehead(
    notehead: BoundingEllipse,
    stems: list[RotatedBoundingBox],
    ink: NDArray | None,
    noteheads: list[BoundingEllipse] | None = None,
    bridge: NDArray | None = None,
) -> list[RotatedBoundingBox]:
    """Every stem drawn on one notehead: at most one up and one down."""
    # Not "does the stem touch the notehead's outline": a head is drawn as an
    # ellipse, so a stem alongside it can miss that curve by a pixel or two and
    # be thrown out before any stem rule is asked.  The side and attachment
    # rules already say whether a stem belongs to this head.
    candidates = [
        stem
        for stem in stems
        if is_plausible_stem(notehead, stem)
        and not belongs_to_another_notehead(notehead, stem, noteheads or [])
    ]
    learned_directions = {stem_direction(notehead, stem) for stem in candidates}

    def from_source(bridge: NDArray | None) -> list[RotatedBoundingBox]:
        return [
            stem
            for stem in source_stem_candidates(notehead, ink, bridge)
            # The scan is only consulted for a direction the segmentation missed,
            # and never for a stem that starts at a different notehead.
            if stem_direction(notehead, stem) not in learned_directions
            and not belongs_to_another_notehead(notehead, stem, noteheads or [])
        ]

    candidates.extend(from_source(None))
    if not candidates and bridge is not None:
        # Nothing at all was found for this head, so it is about to be reported
        # stemless. Only now is the staff line's cut mended and the scan asked
        # again: mending is a last resort, so a head that is already being read
        # cannot be re-read by it. Letting it run on every head was measured --
        # it gains a little and disturbs four times as much, mostly by handing
        # the upper head of a two-voice column the lower head's stem.
        candidates.extend(from_source(bridge))
    longest = {
        direction: max(
            (stem for stem in candidates if stem_direction(notehead, stem) == direction),
            key=lambda candidate: candidate.size[1],
            default=None,
        )
        for direction in StemDirection
    }
    return [longest[direction] for direction in StemDirection if longest[direction] is not None]


def combine_noteheads_with_stems(
    noteheads: list[BoundingEllipse],
    stems: list[RotatedBoundingBox],
    source_image: NDArray | None = None,
) -> list[NoteheadWithStem]:
    """
    Combines noteheads with their stems as this tells us
    what vertical lines are stems and which are bar lines.
    """
    ink = vertical_ink(source_image) if source_image is not None else None
    bridge = bridged_ink(source_image) if source_image is not None else None
    unit = float(np.median([notehead.size[1] for notehead in noteheads])) if noteheads else 0.0
    stems = join_stem_fragments(stems, unit)
    result = []
    for notehead in sorted(noteheads, key=lambda notehead: notehead.box[0][1]):
        found = stems_of_notehead(notehead, stems, ink, noteheads, bridge)
        if not found:
            result.append(NoteheadWithStem(notehead, None, None))
            continue
        directions = [stem_direction(notehead, stem) for stem in found]
        stem = max(found, key=lambda candidate: candidate.size[1])
        direction = directions[0] if len(directions) == 1 else None
        result.append(NoteheadWithStem(notehead, stem, direction, directions))
    share_stems_within_chords(result)
    return result


def share_stems_within_chords(noteheads: list[NoteheadWithStem]) -> None:
    """Give a chord's other noteheads the stem drawn once for all of them.

    A chord's stem runs from the head at its far end, and the part of it that
    lies alongside the other heads is inside their ink, so the segmentation
    often has only the stub sticking out past the last one.  A head with no
    stem of its own therefore takes its chord-mate's.

    Stacked heads are settled first and heads side by side after, because a
    second between two voices and a second within one chord look alike from
    above -- but a head that is part of a stack has already been answered by
    the stack, and asking sideways as well is how it picks up the other voice's
    stem instead.
    """
    for stacked in (True, False):
        for _ in range(CHORD_PASSES):
            for item in noteheads:
                if item.stem_directions:
                    continue
                other = _chord_mate(item, noteheads, stacked)
                if other is None:
                    continue
                item.stem_directions = list(other.stem_directions)
                item.stem_direction = other.stem_directions[0]
                item.stem = item.stem or other.stem


def _chord_mate(
    item: NoteheadWithStem, noteheads: list[NoteheadWithStem], stacked: bool
) -> NoteheadWithStem | None:
    """The notehead of the same chord whose stem this one should be given."""
    for other in noteheads:
        if other is item or len(other.stem_directions) != 1:
            continue
        width = max(item.notehead.size[0], other.notehead.size[0])
        height = max(item.notehead.size[1], other.notehead.size[1])
        sideways = abs(other.notehead.center[0] - item.notehead.center[0])
        offset = item.notehead.center[1] - other.notehead.center[1]
        # A chord's heads sit one above the other, except at an interval of a
        # second, where the engraver puts one beside the other with the stem
        # between them.
        beside = sideways > width * 0.5
        if beside == stacked or sideways > width * 1.1:
            continue
        if not 0 < abs(offset) <= height * (0.8 if beside else 1.3):
            continue
        # Only a down stem is handed upwards: it hangs below the chord, so the
        # head that sees its stub is the lowest one.  An up stem already runs
        # alongside the heads above it, and handing one downwards is exactly how
        # a head steals the stem of the head above it.
        if beside or (other.stem_directions[0] == StemDirection.DOWN and offset < 0):
            return other
    return None


def add_notes_to_staffs(
    staffs: list[Staff], noteheads: list[NoteheadWithStem], symbols: NDArray, notehead_pred: NDArray
) -> list[Note]:
    result = []
    for staff in staffs:
        for notehead_chunk in noteheads:
            if not staff.is_on_staff_zone(notehead_chunk.notehead):
                continue
            center = notehead_chunk.notehead.center
            point = staff.get_at(center[0])
            if point is None:
                continue
            width, height = notehead_chunk.notehead.size
            unit = point.average_unit_size
            if not (
                0.5 * unit <= width <= MAX_NOTEHEAD_WIDTH * unit
                and 0.5 * unit <= height <= 2 * unit
            ):
                continue
            position = point.find_position_in_unit_sizes(notehead_chunk.notehead)
            note = Note(
                notehead_chunk.notehead,
                position,
                notehead_chunk.stem,
                notehead_chunk.stem_direction,
                notehead_chunk.stem_directions,
            )
            result.append(note)
            staff.add_symbol(note)
    number_of_notes = 0
    for staff in staffs:
        number_of_notes += len(staff.get_notes())
    eprint("Found", number_of_notes, "notes during segmentation")
    return result
