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
# Staff positions count up from the bottom line, so the middle line is 5.
MIDDLE_LINE = 5


@dataclass
class Head:
    """One printed notehead: a place on a staff and the stems drawn on it."""

    x: float
    position: float
    stems: set[str] = field(default_factory=set)
    voices: set[str] = field(default_factory=set)
    label: str = ""
    # The bar and beat this notehead sounds on, for a reference head.
    moment: tuple | None = None
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
                moment=note["moment"],
            )
            heads[key] = head
        if note["stem"]:
            head.stems.add(note["stem"])
        head.voices.add(note["voice"])
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
            voices=head.voices,
            label=head.label,
            moment=head.moment,
            scan=head.scan,
        )
        for column in columns
        for head in column.heads
    ]


def _heads_in_column(reference: Column, detected: Column) -> list[tuple[Head | None, Head | None]]:
    """Pair the noteheads of one printed moment, by where they sit on the staff.

    Within a moment the two sides do not agree on left-to-right order -- when
    two voices collide the reference puts the up-stem voice on the left and the
    scan puts it on the right -- so only the staff position is trusted here.
    """
    costs = [
        [
            (
                BLOCKED
                if abs(head.position - other.position) > MAX_POSITION_ERROR
                else abs(head.position - other.position) / MAX_POSITION_ERROR
                + (0.0 if head.stems == other.stems else 0.1)
            )
            for other in detected.heads
        ]
        for head in reference.heads
    ]
    if not costs or not costs[0]:
        return [(head, None) for head in reference.heads] + [
            (None, other) for other in detected.heads
        ]
    return [
        (
            reference.heads[row] if row is not None else None,
            detected.heads[column] if column is not None else None,
        )
        for row, column in _pair_up(costs)
    ]


def _agreement(reference: Column, detected: Column) -> int:
    """How many of the two moments' noteheads sit at the same staff position."""
    return sum(1 for head, other in _heads_in_column(reference, detected) if head and other)


def align_columns(
    reference: list[Column], detected: list[Column]
) -> list[tuple[int | None, int | None]]:
    """Line the two sequences of printed moments up, keeping their order.

    Both sides read left to right and neither reorders the music, so this is a
    sequence alignment and not a geometry problem.  It used to be one: the
    columns were flattened to noteheads and paired by warping reference tenths
    into scan pixels, seeded with a single scale and offset for the whole
    system.  But a scan is not a scaled copy of the engraving -- MuseScore
    spaces a bar by what is in it and the printer spaced it by what fits the
    page -- so on a system where the two disagree by more than the seed's
    tolerance, a whole stretch failed to pair, and every note in it was reported
    both missing and extra.  On Laulun aika's second system that turned one
    wrong pitch and one lost notehead into fifteen failures, all of them
    pointing at a detector that had in fact found the notes.

    What is minimised is noteheads left unexplained, so a moment the scan lost
    entirely costs its own heads and does not drag its neighbours out of step.
    """
    rows, columns = len(reference), len(detected)
    best = [[0.0] * (columns + 1) for _ in range(rows + 1)]
    came: list[list[str]] = [[""] * (columns + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        best[row][0] = best[row - 1][0] + len(reference[row - 1].heads)
        came[row][0] = "reference"
    for column in range(1, columns + 1):
        best[0][column] = best[0][column - 1] + len(detected[column - 1].heads)
        came[0][column] = "detected"
    # A tie-break only: where two alignments explain the same noteheads, prefer
    # the one that also puts the moments in comparable places across the system.
    def place(columns_: list[Column], index: int) -> float:
        span = columns_[-1].x - columns_[0].x
        return (columns_[index].x - columns_[0].x) / span if span else 0.0

    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            here, there = reference[row - 1], detected[column - 1]
            agreed = _agreement(here, there)
            paired = BLOCKED
            if agreed:
                drift = abs(place(reference, row - 1) - place(detected, column - 1))
                paired = (
                    best[row - 1][column - 1]
                    + len(here.heads)
                    + len(there.heads)
                    - 2 * agreed
                    + 0.25 * drift
                )
            skip_reference = best[row - 1][column] + len(here.heads)
            skip_detected = best[row][column - 1] + len(there.heads)
            best[row][column] = min(paired, skip_reference, skip_detected)
            came[row][column] = (
                "paired"
                if best[row][column] == paired
                else "reference"
                if best[row][column] == skip_reference
                else "detected"
            )

    steps: list[tuple[int | None, int | None]] = []
    row, column = rows, columns
    while row or column:
        move = came[row][column]
        if move == "paired":
            row, column = row - 1, column - 1
            steps.append((row, column))
        elif move == "reference":
            row -= 1
            steps.append((row, None))
        else:
            column -= 1
            steps.append((None, column))
    return list(reversed(steps))


def match(reference: list[Column], detected: list[Column]) -> list[tuple[Head | None, Head | None]]:
    """Pair every reference notehead with a detected one, or with nothing."""
    pairs: list[tuple[Head | None, Head | None]] = []
    for row, column in align_columns(reference, detected):
        if row is not None and column is not None:
            pairs.extend(_heads_in_column(reference[row], detected[column]))
        elif row is not None:
            pairs.extend((head, None) for head in reference[row].heads)
        else:
            pairs.extend((None, other) for other in detected[column].heads)
    return pairs


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


def voice_failures(index: int, reference: list[dict], detected: list[dict]) -> list[str]:
    """Check the stem-to-voice rule against the printed voices, bar by bar.

    Stems are worth detecting because they say which voice a note belongs to,
    so the fixtures should say whether they do.  This applies the rule homr
    applies -- a stem decides a voice only in a bar where two notes sound
    together with their stems drawn opposite ways -- and compares the answer
    against the voice the reference prints.
    """
    pairs = [
        (head, other)
        for head, other in match(reference_columns(reference), detected_columns(detected))
        if head is not None and other is not None
    ]
    bars: dict[object, list[tuple[Head, Head]]] = {}
    for head, other in pairs:
        assert head.moment is not None
        bars.setdefault(head.moment[0], []).append((head, other))
    failures = []
    for bar, found in sorted(bars.items(), key=lambda item: str(item[0])):
        moments: dict[object, set[str]] = {}
        contradicts = False
        shared = False
        for head, other in found:
            assert head.moment is not None
            if len(other.stems) > 1:
                # One printed head carrying both stems is two voices meeting.
                shared = True
                continue
            if len(other.stems) != 1:
                continue
            moments.setdefault(head.moment, set()).update(other.stems)
            # A lone voice stems by height; a stem against the note's height
            # means the staff is carrying a second one.
            if "up" in other.stems and other.position > MIDDLE_LINE:
                contradicts = True
            if "down" in other.stems and other.position < MIDDLE_LINE:
                contradicts = True
        together = (
            shared or contradicts or any(len(stems) > 1 for stems in moments.values())
        )
        printed = {voice for head, _ in found for voice in head.voices}
        if not together:
            if len(printed) > 1:
                failures.append(
                    f"staff {index} bar {bar + 1}: the page prints voices"
                    f" {sorted(printed)} but nothing in the detected stems says"
                    " so, and no voice would be decided"
                )
            continue
        for head, other in found:
            if len(other.stems) != 1:
                continue
            expected = "1" if "up" in other.stems else "2"
            if head.voices != {expected}:
                failures.append(
                    f"staff {index} bar {bar + 1}: {head.label} is printed in voice"
                    f" {sorted(head.voices)} but its {sorted(other.stems)[0]} stem"
                    f" would put it in voice {expected}"
                )
    return failures
