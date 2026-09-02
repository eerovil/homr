import numpy as np

from homr.bounding_boxes import BoundingEllipse, RotatedBoundingBox
from homr.model import StemDirection
from homr.note_detection import combine_noteheads_with_stems, split_notehead_ellipse


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
