import numpy as np

from homr.bounding_boxes import BoundingEllipse, RotatedBoundingBox
from homr.model import StemDirection
from homr.note_detection import combine_noteheads_with_stems


empty = np.array([])


def test_stem_matching_skips_short_or_horizontal_fragments() -> None:
    notehead = BoundingEllipse(((50, 50), (16, 12), 0), empty)
    horizontal_fragment = RotatedBoundingBox(((50, 56), (19, 3), 0), empty)
    real_up_stem = RotatedBoundingBox(((50, 22), (2, 50), 0), empty)

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
