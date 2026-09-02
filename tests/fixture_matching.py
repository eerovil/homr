"""Compare homr's detected stem directions against a fixture's reference score.

The reference gives printed noteheads per staff in engraving tenths; homr gives
detected noteheads in scan pixels.  The two are lined up a printed moment at a
time -- every notehead sounding together is one column -- because that is the
level at which the two agree.  Within a column they do not: when two voices
collide on one beat the reference puts the up-stem voice on the left and the
scans put it on the right, so a check that trusted left-to-right order inside a
moment would report the engraver's convention as a detector error.  Anything
the alignment cannot pair is reported as missing or extra rather than dropped.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

MAX_POSITION_ERROR = 1.5
BLOCKED = 1e6
FIT_ROUNDS = 4
# A collision offset puts two voices of one moment about a notehead apart.
COLUMN_GAP = 1.4


@dataclass
class Head:
    """One printed notehead: a place on a staff and the stems drawn on it."""

    x: float
    position: float
    stems: set[str] = field(default_factory=set)
    label: str = ""
    # Where the notehead is in the scan, for a detected one.
    scan: tuple[float, float] | None = None


@dataclass
class Column:
    """The noteheads of one printed moment."""

    x: float
    heads: list[Head]


def reference_columns(notes: list[dict]) -> list[Column]:
    """Group reference notes by the moment they sound, merging unison heads."""
    moments: dict[tuple, dict[tuple[float, int], Head]] = {}
    for note in notes:
        heads = moments.setdefault(note["moment"], {})
        # Two voices share a printed notehead only when the reference draws
        # them at one place: same moment, same staff position, same x.  Two
        # voices merely sounding the same pitch at once are drawn side by side
        # and are two heads.
        key = (note["position"], round(note["x"]))
        head = heads.get(key)
        if head is None:
            head = Head(
                x=note["x"],
                position=note["position"],
                label=f"m{note['measure']} {note['step']}{note['octave']}",
            )
            heads[key] = head
        if note["stem"]:
            head.stems.add(note["stem"])
    columns = [
        Column(x=min(head.x for head in heads.values()), heads=list(heads.values()))
        for _, heads in sorted(moments.items())
    ]
    columns.sort(key=lambda column: column.x)
    return columns


def detected_columns(notes: list[dict]) -> list[Column]:
    """Group detected noteheads into the printed moments they belong to."""
    if not notes:
        return []
    ordered = sorted(notes, key=lambda note: note["x"])
    gap = COLUMN_GAP * sorted(note["w"] for note in notes)[len(notes) // 2]
    groups: list[list[dict]] = [[ordered[0]]]
    for note in ordered[1:]:
        if note["x"] - groups[-1][-1]["x"] > gap:
            groups.append([])
        groups[-1].append(note)
    return [
        Column(
            x=min(note["x"] for note in group),
            heads=[
                Head(
                    x=note["x"],
                    position=float(note["position"]),
                    stems=set(note["stems"]),
                    scan=(note["x"], note["y"]),
                )
                for note in group
            ],
        )
        for group in groups
    ]


class Warp:
    """A monotone map from reference tenths to scan pixels.

    A scan is not a scaled copy of the reference engraving: the same music is
    spaced differently bar by bar, so one scale and offset for a whole system
    leaves real matches tens of pixels out.  The map is therefore refitted
    through the columns it has matched so far, interpolating between them and
    extrapolating with the overall slope outside.
    """

    def __init__(self, pairs: list[tuple[float, float]]):
        anchors: dict[float, list[float]] = {}
        for reference, detected in pairs:
            anchors.setdefault(reference, []).append(detected)
        self.anchors = sorted(
            (reference, sum(values) / len(values)) for reference, values in anchors.items()
        )
        first, last = self.anchors[0], self.anchors[-1]
        span = last[0] - first[0]
        self.slope = (last[1] - first[1]) / span if span else 1.0

    def __call__(self, x: float) -> float:
        if x <= self.anchors[0][0]:
            return self.anchors[0][1] + self.slope * (x - self.anchors[0][0])
        if x >= self.anchors[-1][0]:
            return self.anchors[-1][1] + self.slope * (x - self.anchors[-1][0])
        for (left_x, left), (right_x, right) in zip(self.anchors, self.anchors[1:], strict=False):
            if left_x <= x <= right_x:
                share = (x - left_x) / (right_x - left_x) if right_x > left_x else 0.0
                return left + share * (right - left)
        return self.anchors[-1][1]


def assign(costs: list[list[float]]) -> list[int]:
    """Least-cost one-to-one assignment of rows to columns (Jonker-Volgenant)."""
    if not costs or not costs[0]:
        return []
    rows, columns = len(costs), len(costs[0])
    assert rows <= columns
    potential_row = [0.0] * (rows + 1)
    potential_column = [0.0] * (columns + 1)
    match = [-1] * (columns + 1)
    for row in range(1, rows + 1):
        match[0] = row - 1
        column = 0
        minima = [float("inf")] * (columns + 1)
        previous = [-1] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column] = True
            current, delta, next_column = match[column], float("inf"), 0
            for candidate in range(1, columns + 1):
                if used[candidate]:
                    continue
                value = (
                    costs[current][candidate - 1]
                    - potential_row[current + 1]
                    - potential_column[candidate]
                )
                if value < minima[candidate]:
                    minima[candidate], previous[candidate] = value, column
                if minima[candidate] < delta:
                    delta, next_column = minima[candidate], candidate
            for candidate in range(columns + 1):
                if used[candidate]:
                    potential_row[match[candidate] + 1] += delta
                    potential_column[candidate] -= delta
                else:
                    minima[candidate] -= delta
            column = next_column
            if match[column] == -1:
                break
        while column:
            match[column], column = match[previous[column]], previous[column]
    result = [-1] * rows
    for column in range(1, columns + 1):
        if match[column] != -1:
            result[match[column]] = column - 1
    return result


def _pair_up(costs: list[list[float]]) -> list[tuple[int | None, int | None]]:
    """Pair rows with columns by least cost, leaving blocked ones unpaired."""
    rows = len(costs)
    columns = len(costs[0]) if rows else 0
    size = max(rows, columns)
    if size == 0:
        return []
    square = [[BLOCKED] * size for _ in range(size)]
    for row in range(rows):
        for column in range(columns):
            square[row][column] = costs[row][column]
    chosen = assign(square)
    pairs: list[tuple[int | None, int | None]] = []
    taken = set()
    for row in range(rows):
        column = chosen[row]
        if column < columns and costs[row][column] < BLOCKED:
            taken.add(column)
            pairs.append((row, column))
        else:
            pairs.append((row, None))
    pairs.extend((None, column) for column in range(columns) if column not in taken)
    return pairs


def _column_heads(columns: list[Column]) -> list[Head]:
    """Every notehead, told the place of the moment it belongs to."""
    return [
        Head(
            x=column.x,
            position=head.position,
            stems=head.stems,
            label=head.label,
            scan=head.scan,
        )
        for column in columns
        for head in column.heads
    ]


def match(reference: list[Column], detected: list[Column]) -> list[tuple[Head | None, Head | None]]:
    """Pair every reference notehead with a detected one, or with nothing."""
    heads, others = _column_heads(reference), _column_heads(detected)
    if not heads or not others:
        return [(head, None) for head in heads] + [(None, other) for other in others]
    span = max(other.x for other in others) - min(other.x for other in others)
    reference_span = max(head.x for head in heads) - min(head.x for head in heads)
    scale = span / reference_span if reference_span else 1.0
    offset = min(other.x for other in others) - scale * min(head.x for head in heads)
    warp: Callable[[float], float] = lambda x: scale * x + offset  # noqa: E731
    pairs: list[tuple[int | None, int | None]] = []
    for index in range(FIT_ROUNDS):
        # Start loose enough to survive the reference's different bar spacing,
        # then tighten once the anchors have absorbed it.
        tolerance = max(span, 1.0) * (0.1 if index == 0 else 0.03)
        costs = [
            [_cost(head, other, warp, tolerance) for other in others] for head in heads
        ]
        pairs = _pair_up(costs)
        matched = [
            (heads[row].x, others[column].x)
            for row, column in pairs
            if row is not None and column is not None
        ]
        if len(matched) < 2:
            break
        warp = Warp(matched)
    return [
        (
            heads[row] if row is not None else None,
            others[column] if column is not None else None,
        )
        for row, column in pairs
    ]


def _cost(
    head: Head, other: Head, warp: "Warp | Callable[[float], float]", tolerance: float
) -> float:
    """What it costs to call one detected notehead this reference one.

    Where geometry alone leaves a choice -- two voices meeting on one line in
    one moment -- the pairing that agrees is preferred, so only a moment that
    cannot be read correctly at all is reported.
    """
    position_error = abs(head.position - other.position)
    distance = abs(warp(head.x) - other.x)
    if position_error > MAX_POSITION_ERROR or distance > tolerance:
        return BLOCKED
    return (
        0.5 * distance / tolerance
        + 0.4 * position_error / MAX_POSITION_ERROR
        + (0.0 if head.stems == other.stems else 0.1)
    )


@dataclass
class StaffResult:
    index: int
    correct: int = 0
    wrong: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[str]:
        return self.wrong + self.missing + self.extra


def _stems(head: Head) -> list[str]:
    return sorted(head.stems) or ["none"]


def check_staff(index: int, reference: list[dict], detected: list[dict]) -> StaffResult:
    result = StaffResult(index=index)
    for head, found in match(reference_columns(reference), detected_columns(detected)):
        if head is None:
            assert found is not None
            result.extra.append(
                f"staff {index}: extra notehead at x={found.x} position"
                f" {found.position} with stems {_stems(found)}"
            )
        elif found is None:
            result.missing.append(
                f"staff {index}: no detected notehead for {head.label} at x={head.x}"
            )
        elif found.stems != head.stems:
            result.wrong.append(
                f"staff {index}: {head.label} at x={found.x} expected"
                f" {_stems(head)} but detected {_stems(found)}"
            )
        else:
            result.correct += 1
    return result


def check_fixture(reference: list[dict], detected: list[list[dict]]) -> list[StaffResult]:
    results = []
    for index in range(max(len(reference), len(detected))):
        reference_notes = reference[index]["notes"] if index < len(reference) else []
        detected_notes = detected[index] if index < len(detected) else []
        results.append(check_staff(index + 1, reference_notes, detected_notes))
    return results
