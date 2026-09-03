import cv2
import numpy as np

from homr.bounding_boxes import BoundingEllipse, RotatedBoundingBox
from homr.model import StemDirection
from homr.note_detection import (
    bridged_ink,
    combine_noteheads_with_stems,
    split_notehead_ellipse,
    stems_of_notehead,
    vertical_ink,
)


empty = np.array([])


def test_stem_matching_skips_short_or_horizontal_fragments() -> None:
    notehead = BoundingEllipse(((50, 50), (16, 12), 0), empty)
    horizontal_fragment = RotatedBoundingBox(((50, 56), (19, 3), 0), empty)
    real_up_stem = RotatedBoundingBox(((58, 22), (2, 50), 0), empty)

    matched = combine_noteheads_with_stems([notehead], [horizontal_fragment, real_up_stem])

    assert matched[0].stem == real_up_stem
    assert matched[0].stem_direction == StemDirection.UP


def test_stem_matching_reports_unknown_when_only_noise_overlaps() -> None:
    notehead = BoundingEllipse(((50, 50), (16, 12), 0), empty)
    tiny_fragment = RotatedBoundingBox(((50, 52), (0.5, 2), 0), empty)

    matched = combine_noteheads_with_stems([notehead], [tiny_fragment])

    assert matched[0].stem is None
    assert matched[0].stem_direction is None


def test_stem_matching_rejects_a_nearby_barline() -> None:
    notehead = BoundingEllipse(((50, 50), (16, 12), 0), empty)
    real_up_stem = RotatedBoundingBox(((58, 22), (2, 50), 0), empty)
    nearby_barline = RotatedBoundingBox(((65, 50), (2, 120), 0), empty)

    matched = combine_noteheads_with_stems([notehead], [nearby_barline, real_up_stem])

    assert matched[0].stem == real_up_stem
    assert matched[0].stem_direction == StemDirection.UP


def test_stem_matching_keeps_both_directions_on_a_shared_notehead() -> None:
    notehead = BoundingEllipse(((50, 50), (16, 12), 0), empty)
    down_stem = RotatedBoundingBox(((42, 58), (2, 8), 0), empty)
    up_stem = RotatedBoundingBox(((58, 22), (2, 50), 0), empty)

    matched = combine_noteheads_with_stems([notehead], [down_stem, up_stem])

    assert matched[0].stem_directions == [StemDirection.UP, StemDirection.DOWN]


def test_stem_matching_accepts_a_notehead_length_stem_on_the_correct_side() -> None:
    notehead = BoundingEllipse(((50, 50), (16, 12), 0), empty)
    down_stem = RotatedBoundingBox(((42, 58), (2, 12), 0), empty)

    matched = combine_noteheads_with_stems([notehead], [down_stem])

    assert matched[0].stem_direction == StemDirection.DOWN


def test_split_noteheads_before_matching_keeps_adjacent_stems_separate() -> None:
    noteheads = np.zeros((100, 100), dtype=np.uint8)
    noteheads[40:64, 42:58] = 1
    merged = BoundingEllipse(((50, 52), (16, 24), 0), empty)
    up_stem = RotatedBoundingBox(((58, 27), (2, 30), 0), empty)
    down_stem = RotatedBoundingBox(((42, 77), (2, 30), 0), empty)

    split = split_notehead_ellipse(merged, noteheads, 12)
    matched = combine_noteheads_with_stems(split, [up_stem, down_stem])

    assert [note.stem_direction for note in matched] == [StemDirection.UP, StemDirection.DOWN]


def test_a_head_does_not_take_the_stem_of_the_head_above_it() -> None:
    upper = BoundingEllipse(((50, 40), (16, 12), 0), empty)
    lower = BoundingEllipse(((50, 54), (16, 12), 0), empty)
    # The upper head's own up stem, drawn on its right and stopping above it.
    up_stem = RotatedBoundingBox(((58, 22), (2, 30), 0), empty)

    matched = combine_noteheads_with_stems([upper, lower], [up_stem])

    assert [note.stem_directions for note in matched] == [[StemDirection.UP], []]


def test_a_chord_note_keeps_a_stem_drawn_alongside_it() -> None:
    top = BoundingEllipse(((50, 40), (16, 12), 0), empty)
    bottom = BoundingEllipse(((50, 54), (16, 12), 0), empty)
    # One down stem for the chord: it runs past both heads and below them.
    down_stem = RotatedBoundingBox(((42, 62), (2, 60), 0), empty)

    matched = combine_noteheads_with_stems([top, bottom], [down_stem])

    assert [note.stem_directions for note in matched] == [
        [StemDirection.DOWN],
        [StemDirection.DOWN],
    ]


def test_two_touching_noteheads_are_split_at_their_waist_not_by_height() -> None:
    # Two heads a third apart: the ink is 2.6 noteheads tall, and dividing that
    # by a notehead and rounding used to make three of them.
    mask = np.zeros((120, 100), dtype=np.uint8)
    cv2.ellipse(mask, (50, 44), (10, 7), 0, 0, 360, 1, -1)
    cv2.ellipse(mask, (50, 58), (10, 7), 0, 0, 360, 1, -1)
    clump = BoundingEllipse(((50, 51), (20, 28), 0), empty)

    split = split_notehead_ellipse(clump, mask, 12)

    assert len(split) == 2


def test_a_notehead_with_a_ledger_line_stays_one_notehead() -> None:
    mask = np.zeros((80, 120), dtype=np.uint8)
    cv2.ellipse(mask, (50, 40), (10, 7), 0, 0, 360, 1, -1)
    mask[39:42, 34:70] = 1
    clump = BoundingEllipse(((52, 40), (36, 14), 0), empty)

    split = split_notehead_ellipse(clump, mask, 12)

    assert len(split) == 1


def test_a_stem_cut_into_pieces_by_staff_lines_is_read_as_one() -> None:
    notehead = BoundingEllipse(((50, 50), (16, 12), 0), empty)
    # One stem, lost where two staff lines crossed it.
    pieces = [
        RotatedBoundingBox(((58, 38), (2, 10), 0), empty),
        RotatedBoundingBox(((58, 24), (2, 12), 0), empty),
        RotatedBoundingBox(((58, 10), (2, 12), 0), empty),
    ]

    matched = combine_noteheads_with_stems([notehead], pieces)

    assert matched[0].stem_direction == StemDirection.UP
    assert matched[0].stem is not None
    assert matched[0].stem.size[1] > 30


def test_a_barline_is_not_joined_into_one_long_stem() -> None:
    notehead = BoundingEllipse(((50, 50), (16, 12), 0), empty)
    barline = [
        RotatedBoundingBox(((58, 40), (2, 30), 0), empty),
        RotatedBoundingBox(((58, 8), (2, 30), 0), empty),
    ]

    matched = combine_noteheads_with_stems([notehead], barline)

    assert matched[0].stem is None or matched[0].stem.size[1] <= 12 * 5


def test_a_chord_shares_a_down_stem_that_only_sticks_out_below_it() -> None:
    top = BoundingEllipse(((50, 40), (16, 12), 0), empty)
    bottom = BoundingEllipse(((50, 52), (16, 12), 0), empty)
    # All that is left of the chord's stem is the stub below the lowest head.
    stub = RotatedBoundingBox(((42, 64), (2, 10), 0), empty)

    matched = combine_noteheads_with_stems([top, bottom], [stub])

    assert [note.stem_direction for note in matched] == [
        StemDirection.DOWN,
        StemDirection.DOWN,
    ]


def test_a_second_takes_the_stem_drawn_between_its_two_noteheads() -> None:
    left = BoundingEllipse(((40, 51), (16, 12), 0), empty)
    right = BoundingEllipse(((56, 45), (16, 12), 0), empty)
    down_stem = RotatedBoundingBox(((47, 63), (2, 24), 0), empty)

    matched = combine_noteheads_with_stems([left, right], [down_stem])

    assert {note.stem_direction for note in matched} == {StemDirection.DOWN}


def test_the_upper_voice_of_a_column_keeps_only_its_own_up_stem() -> None:
    """Two voices in one column: the down stem below is not the upper head's.

    The stem hangs from the lower head, and the ink the segmentation gives it
    reaches up alongside the upper head -- which is what makes this the hard
    case, since a stem drawn *beside* a head is normally its own or its chord's.
    It is the shape that puts a singer in the wrong voice: the head reports both
    directions where the page prints one.
    """
    upper = BoundingEllipse(((50, 40), (16, 12), 0), empty)
    lower = BoundingEllipse(((50, 52), (16, 12), 0), empty)
    up_stem = RotatedBoundingBox(((58, 22), (2, 36), 0), empty)
    down_stem = RotatedBoundingBox(((42, 63), (2, 36), 0), empty)

    matched = combine_noteheads_with_stems([upper, lower], [up_stem, down_stem])

    directions = {note.notehead.center[1]: note.stem_directions for note in matched}
    assert directions[40] == [StemDirection.UP]
    assert directions[52] == [StemDirection.DOWN]


def test_a_voice_beside_a_chord_does_not_take_its_stem() -> None:
    top = BoundingEllipse(((50, 40), (16, 12), 0), empty)
    bottom = BoundingEllipse(((50, 52), (16, 12), 0), empty)
    beside = BoundingEllipse(((68, 40), (16, 12), 0), empty)
    stub = RotatedBoundingBox(((42, 64), (2, 10), 0), empty)
    its_own = RotatedBoundingBox(((76, 20), (2, 30), 0), empty)

    matched = combine_noteheads_with_stems([top, bottom, beside], [stub, its_own])

    directions = {note.notehead.center[0]: note.stem_direction for note in matched}
    assert directions[68] == StemDirection.UP
    assert directions[50] == StemDirection.DOWN


def a_page_with_a_stem_crossing_a_staff_line() -> np.ndarray:
    """A scan: one notehead, its up stem, and a staff line drawn through both."""
    page = np.full((80, 80), 255, np.uint8)
    cv2.ellipse(page, (40, 60), (8, 6), 0, 0, 360, 0, -1)  # the notehead
    page[20:53, 47:50] = 0  # its stem, on the head's right
    page[46:50, :] = 0  # a staff line, right across the stem near its foot
    return page


def test_a_stem_crossing_a_staff_line_still_reaches_its_notehead() -> None:
    page = a_page_with_a_stem_crossing_a_staff_line()
    head = BoundingEllipse(((40, 60), (16, 12), 0), empty)

    matched = combine_noteheads_with_stems([head], [], page)

    assert matched[0].stem_direction == StemDirection.UP


def test_the_staff_line_is_still_no_stem_of_its_own() -> None:
    page = np.full((80, 80), 255, np.uint8)
    cv2.ellipse(page, (40, 60), (8, 6), 0, 0, 360, 0, -1)
    page[46:50, :] = 0  # a staff line and nothing else
    head = BoundingEllipse(((40, 60), (16, 12), 0), empty)

    matched = combine_noteheads_with_stems([head], [], page)

    assert matched[0].stem_direction is None


def test_mending_a_stroke_invents_no_ink_the_scan_has_not_got() -> None:
    page = a_page_with_a_stem_crossing_a_staff_line()

    mended = bridged_ink(page)

    assert not ((mended > 0) & (page >= 180)).any()


def test_two_strokes_with_paper_between_them_are_not_one_stroke() -> None:
    page = np.full((80, 80), 255, np.uint8)
    page[10:30, 40:43] = 0
    page[36:56, 40:43] = 0  # a second stroke, with paper in between

    mended = bridged_ink(page)

    assert not mended[30:36, 41].any()


def test_a_stem_the_scan_already_gives_is_not_read_again_through_the_mend() -> None:
    """Mending is a last resort: a head being read already is left alone."""
    page = a_page_with_a_stem_crossing_a_staff_line()
    head = BoundingEllipse(((40, 60), (16, 12), 0), empty)
    without_mending = stems_of_notehead(head, [], vertical_ink(page), [head], None)
    with_mending = stems_of_notehead(head, [], vertical_ink(page), [head], bridged_ink(page))

    assert not without_mending  # the pieces are too short and too far
    assert len(with_mending) == 1

    segmentation_found_it = RotatedBoundingBox(((48, 40), (2, 30), 0), empty)
    both = stems_of_notehead(
        head, [segmentation_found_it], vertical_ink(page), [head], bridged_ink(page)
    )

    assert both == stems_of_notehead(head, [segmentation_found_it], vertical_ink(page), [head])
