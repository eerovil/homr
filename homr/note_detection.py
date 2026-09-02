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
# How much narrower than its shoulders the ink has to get before it counts as
# the waist between two noteheads rather than one head's own rounded end.
WAIST_DEPTH = 0.8
# The longest a stem can be, in noteheads, when joining its broken pieces.
MAX_STEM_HEIGHT = 5.0


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


def check_bbox_size(bbox: cvt.Rect, noteheads: NDArray, unit_size: float) -> list[cvt.Rect]:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    note_w = constants.NOTEHEAD_SIZE_RATIO * unit_size
    note_h = unit_size

    new_bbox: list[cvt.Rect] = []
    region = noteheads[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    columns = (region > 0).sum(axis=0).astype(float) if region.size else np.zeros(0)
    waists = [column for column in _waists(columns, note_w) if 0 < column < w]
    if waists:
        # Heads side by side: cut where the ink narrows between them, not down
        # the middle, so a notehead with a ledger line growing out of it is not
        # halved into two heads.
        cuts = [0, *waists, w]
        for left, right in zip(cuts, cuts[1:], strict=False):
            part = adjust_bbox((bbox[0] + left, bbox[1], bbox[0] + right, bbox[3]), noteheads)
            new_bbox.extend(check_bbox_size(part, noteheads, unit_size))
    else:
        new_bbox.extend(split_stack(bbox, noteheads, note_h))

    return new_bbox


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
    if len(split_boxes) <= 1:
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
        and stem_width <= notehead_width * 0.75
        and is_attached(notehead, stem)
    )


def vertical_ink(source_image: NDArray) -> NDArray:
    """The scan's ink with staff lines and beams taken out, for stem recovery."""
    ink = (source_image < 180).astype(np.uint8)
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, 12), np.uint8))
    ink &= 1 - horizontal
    return cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((1, 12), np.uint8))


def _longest_run(
    ink: NDArray,
    notehead: BoundingEllipse,
    columns: range,
    rows: range,
    direction: StemDirection,
) -> RotatedBoundingBox | None:
    x, y = map(int, notehead.center)
    height = int(notehead.size[1])
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
            if direction == StemDirection.UP:
                gap = y - height // 2 - bottom
            else:
                gap = top - (y + height // 2)
            # Beam and staff-line removal can leave a short white gap
            # between a real source-image stem and its notehead.
            attached = -max(2, height * 0.1) <= gap <= height * 0.65
            if attached and (best is None or length > best[0]):
                best = (length, column, top, bottom)
    if best is None or not height * 0.5 <= best[0] <= height * 5:
        return None
    length, column, top, bottom = best
    return RotatedBoundingBox(((column, (top + bottom) / 2), (3, length), 0), np.empty((0, 2)))


def source_stem_candidates(
    notehead: BoundingEllipse, ink: NDArray | None
) -> list[RotatedBoundingBox]:
    """Recover a visibly printed stem when its segmentation class is empty."""
    if ink is None:
        return []
    x, y = map(int, notehead.center)
    width, height = map(int, notehead.size)
    found = [
        _longest_run(
            ink, notehead, range(x, x + width + 16), range(max(0, y - 80), y + 1), StemDirection.UP
        ),
        _longest_run(
            ink,
            notehead,
            range(max(0, x - width - 16), x + 1),
            range(y, min(ink.shape[0], y + 81)),
            StemDirection.DOWN,
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


def belongs_to_another_notehead(
    notehead: BoundingEllipse,
    stem: RotatedBoundingBox,
    noteheads: list[BoundingEllipse],
) -> bool:
    """Whether a stem starts at some other notehead rather than at this one.

    Voices stacked in one column sit close enough that one head can reach the
    next head's stem: the gap it has to cross is smaller than the room a real
    stem needs.  A stem drawn *alongside* this head is its own or its chord's,
    so only one that stops short of it is asked whose it is, and the answer is
    whichever head its near end lands on.
    """
    stem_top = stem.center[1] - stem.size[1] / 2
    stem_bottom = stem.center[1] + stem.size[1] / 2
    note_top = notehead.center[1] - notehead.size[1] / 2
    note_bottom = notehead.center[1] + notehead.size[1] / 2
    if stem_top <= note_bottom and stem_bottom >= note_top:
        return False
    if stem_direction(notehead, stem) == StemDirection.UP:
        near_y = stem_bottom
    else:
        near_y = stem_top
    return any(
        other is not notehead
        and abs(other.center[0] - stem.center[0]) <= other.size[0] * 0.8
        and abs(other.center[1] - near_y) <= other.size[1]
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
    candidates.extend(
        stem
        for stem in source_stem_candidates(notehead, ink)
        # The scan is only consulted for a direction the segmentation missed,
        # and never for a stem that starts at a different notehead.
        if stem_direction(notehead, stem) not in learned_directions
        and not belongs_to_another_notehead(notehead, stem, noteheads or [])
    )
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
    unit = float(np.median([notehead.size[1] for notehead in noteheads])) if noteheads else 0.0
    stems = join_stem_fragments(stems, unit)
    result = []
    for notehead in sorted(noteheads, key=lambda notehead: notehead.box[0][1]):
        found = stems_of_notehead(notehead, stems, ink, noteheads)
        if not found:
            result.append(NoteheadWithStem(notehead, None, None))
            continue
        directions = [stem_direction(notehead, stem) for stem in found]
        stem = max(found, key=lambda candidate: candidate.size[1])
        direction = directions[0] if len(directions) == 1 else None
        result.append(NoteheadWithStem(notehead, stem, direction, directions))
    return result


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
            if not (0.5 * unit <= width <= 3 * unit and 0.5 * unit <= height <= 2 * unit):
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
