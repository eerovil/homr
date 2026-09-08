# ruff: noqa: S101

"""The recovery gate must not disagree with final stem attachment."""

import cv2
import numpy as np
import pytest

from homr.bounding_boxes import BoundingEllipse
from homr.model import StemDirection
from homr.note_detection import (
    _reach_towards_head,
    combine_noteheads_with_stems,
    source_stem_candidates,
)


def crossing(direction: StemDirection) -> tuple[np.ndarray, tuple[int, int]]:
    page = np.full((80, 80), 255, dtype=np.uint8)
    cv2.ellipse(page, (40, 60), (8, 6), 0, 0, 360, 0, -1)
    page[20:53, 47:50] = 0
    page[46:50, :] = 0
    if direction == StemDirection.DOWN:
        return cv2.flip(page, -1), (39, 19)
    return page, (40, 60)


@pytest.mark.parametrize("height", [12, 14, 16])
@pytest.mark.parametrize("direction", [StemDirection.UP, StemDirection.DOWN])
def test_staff_line_recovery_is_stable_under_small_head_size_changes(
    height: int, direction: StemDirection
) -> None:
    # Same printed ink. A slightly taller predicted head used to make the
    # coarse run-search gate say "attached" while is_attached rejected it.
    # That bypassed bridging and erased a plainly printed stem.
    page, center = crossing(direction)
    head = BoundingEllipse((center, (16, height), 0), np.empty((0, 2)))
    result = combine_noteheads_with_stems([head], [], page)
    assert result[0].stem_directions == [direction]


@pytest.mark.parametrize("direction", [StemDirection.UP, StemDirection.DOWN])
def test_paper_gaps_are_not_repaired_as_staff_line_crossings(direction: StemDirection) -> None:
    page, center = crossing(direction)
    if direction == StemDirection.UP:
        page[46:50, 47:50] = 255
    else:
        page[30:34, 30:33] = 255
    head = BoundingEllipse((center, (16, 16), 0), np.empty((0, 2)))
    result = combine_noteheads_with_stems([head], [], page)
    assert result[0].stem_direction is None


@pytest.mark.parametrize("direction", [StemDirection.UP, StemDirection.DOWN])
def test_equally_long_recovery_runs_prefer_final_attachment(
    direction: StemDirection,
) -> None:
    ink: np.ndarray = np.zeros((80, 80), dtype=np.uint8)
    far_column, near_column = (48, 49) if direction == StemDirection.UP else (49, 48)
    ink[34:45, far_column] = 1  # Visited first; coarse search accepts the gap.
    ink[43:54, near_column] = 1  # Same raw length, but actually attached.
    center = (40, 60)
    if direction == StemDirection.DOWN:
        ink = cv2.flip(ink, -1)
        center = (39, 19)
    head = BoundingEllipse((center, (16, 14), 0), np.empty((0, 2)))
    assert source_stem_candidates(head, ink) == []
    # No bridge pixels are added. Tie-breaking must choose existing evidence,
    # not extend the disconnected run through paper or relax the length floor.
    candidates = source_stem_candidates(head, ink, ink.copy())
    assert len(candidates) == 1
    assert candidates[0].size[1] == 11


@pytest.mark.parametrize("towards", [-1, 1])
@pytest.mark.parametrize("endpoint_has_ink", [False, True])
def test_bridge_requires_a_source_backed_starting_endpoint(
    towards: int,
    endpoint_has_ink: bool,
) -> None:
    bridge = np.zeros((12, 1), dtype=np.uint8)
    near, edge = 5, 5 + towards * 3
    for distance in range(1, 4):
        bridge[near + towards * distance, 0] = 1
    bridge[near, 0] = endpoint_has_ink
    reached = _reach_towards_head(bridge, 0, near, edge, 3)
    assert reached == (edge if endpoint_has_ink else near)
