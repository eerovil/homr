"""Musical reconstruction between decoder tokens and MusicXML serialization.

This module owns the decisions that turn a flat decoder token stream into the
musical structure the serializer consumes: sounding moments, tuplet grouping,
bar-arithmetic repair, inferred meter changes, and the timing metadata derived
from those reconstructed bars. It deliberately contains no XML construction.

Keeping this phase explicit means a MusicXML serializer can be changed or
replaced without silently changing what homr believes the music is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol, SymbolDuration, sort_token_chords


@dataclass(frozen=True)
class ReconstructionChange:
    """One deliberate change made after decoding and before serialization."""

    kind: str
    bar: int
    group: int
    symbol: int | None
    staff: str | None
    pitch: str | None
    before: str | None
    after: str
    reason: str


class SymbolChord:
    def __init__(self, symbols: list[EncodedSymbol], tuplet_mark: str = "") -> None:
        self.symbols = symbols
        self.tuplet_mark = tuplet_mark

    def __str__(self) -> str:
        return str.join("&", [str(s) for s in self.symbols])

    def __repr__(self) -> str:
        return str(self)

    def is_barline(self) -> bool:
        if len(self.symbols) == 0:
            return False
        first_rhythm = self.symbols[0].rhythm
        return "barline" in first_rhythm or "repeat" in first_rhythm

    def get_duration(self) -> Fraction:
        notes_rests = [
            s.get_duration().fraction for s in self.symbols if s.rhythm.startswith(("note", "rest"))
        ]
        if len(notes_rests) == 0:
            return Fraction(0)
        return min(notes_rests)

    def is_only_rests(self) -> bool:
        """Nothing here sounds -- every note or rest of this moment is a rest."""
        sounding = [s for s in self.symbols if s.rhythm.startswith(("note", "rest"))]
        return len(sounding) > 0 and all(s.rhythm.startswith("rest") for s in sounding)

    def into_positions(self) -> list[SymbolChord]:
        upper = []
        lower = []
        lower_is_only_rest = True
        for symbol in self.symbols:
            if symbol.position == "upper":
                upper.append(symbol)
            else:
                lower.append(symbol)
                lower_is_only_rest = lower_is_only_rest and symbol.rhythm.startswith("rest")
        chords = (
            SymbolChord(upper, self.tuplet_mark),
            SymbolChord(lower, self.tuplet_mark),
        )
        if lower_is_only_rest:
            chords = (chords[1], chords[0])
        return [chord for chord in chords if len(chord.symbols) > 0]


def find_common_division(durations: list[Fraction]) -> int:
    """
    Find the smallest division (denominator) so that all durations
    can be expressed as integer multiples.
    """

    def lcm(a: int, b: int) -> int:
        return abs(a * b) // math.gcd(a, b)

    denominators = [d.denominator for d in durations if d > 0]
    if not denominators:
        return 1
    common = denominators[0]
    for d in denominators[1:]:
        common = lcm(common, d)
    return common


def _bar_boundaries(voice: list[SymbolChord]) -> list[tuple[int, int]]:
    """Where each bar starts and ends in the stream, the last one open-ended."""
    bars: list[tuple[int, int]] = []
    start = 0
    for index, chord in enumerate(voice):
        if chord.is_barline():
            bars.append((start, index))
            start = index + 1
    if start < len(voice):
        bars.append((start, len(voice)))
    return bars


def _staff_lengths(voice: list[SymbolChord], span: tuple[int, int]) -> dict[str, Fraction]:
    """How long each staff of this bar measures, each on its own cursor.

    A moment costs a staff the shortest of *its* notes there, which is what
    `MeasureCursors` does when it writes the bar out, so a staff holding a whole
    note against four quarters measures a whole.
    """
    by_position: dict[str, Fraction] = {}
    for chord in voice[span[0] : span[1]]:
        sounding: dict[str, list[Fraction]] = {}
        for symbol in chord.symbols:
            if not symbol.rhythm.startswith(("note", "rest")):
                continue
            sounding.setdefault(symbol.position, []).append(symbol.get_duration().fraction)
        for position, durations in sounding.items():
            by_position[position] = by_position.get(position, Fraction(0)) + min(durations)
    return by_position


def _corroborated_length(voice: list[SymbolChord], span: tuple[int, int]) -> Fraction | None:
    """How long this bar is, when every staff of the system agrees about it.

    Agreement is the whole guard: a bar homr misread is short in the staff it
    lost a note from and right in the other, and a meter must not be invented off
    one staff's arithmetic. A single-staff system has nothing to agree with and
    so never carries one.
    """
    by_position = _staff_lengths(voice, span)
    if len(by_position) < 2:
        return None
    lengths = set(by_position.values())
    return lengths.pop() if len(lengths) == 1 else None


def repair_bar_arithmetic(
    voice: list[SymbolChord], changes: list[ReconstructionChange] | None = None
) -> list[SymbolChord]:
    """Take the decoder's second answer where its first one does not fit the bar.

    A misread note value is not silent: the bar it is in stops adding up. On
    `hanget-soi` bar 2 the page prints a dotted quarter, a 16th rest and a 16th
    against a half chord; the model read the dotted quarter as a **half** at
    88.2%, so that staff measures two and a half quarters of a bar every other
    piece of evidence says is two, and the 16th the page prints at beat 1.75 is
    written past the end of the bar and lost.

    The value the page prints is not out of reach: the decoder scores every
    rhythm token at every step, and `note_4.` is sitting there at 0.9%. What was
    missing is a reason to prefer it, and the arithmetic is one -- it is the only
    candidate that makes the bar add up. So this walks the bars that do not, and
    where **exactly one** of the decoder's own alternatives fixes one, takes it.

    Everything else here is a refusal, because a rule that repairs an ambiguous
    bar is guessing at the music rather than reading it:

    - **Never off one staff.** The target length is corroborated twice over: the
      bars of this span that every staff agrees about say what a bar is, and in
      the bar being repaired every staff but one must already measure it. A
      system printing one staff is never repaired -- the same rule
      `_corroborated_length` follows, and for the same reason.
    - **Never the bar that opens a span**, which is where an anacrusis is.
    - **Never a bar this cannot measure** -- a tuplet, a grace note or a
      multi-measure rest anywhere in it, since its length would be arithmetic
      this does not do.
    - **Never a note the page may have drawn as part of a chord.** Two notes of
      one moment drawn with the same stem are one chord and share a value;
      shortening one of them would take a printed chord apart. Only a note
      standing alone on its stem can have a value of its own, and a note whose
      stem nothing was matched to has no evidence either way.
    - **Never more than one way, on the arithmetic alone.** Two alternatives
      that both make the bar add up are two readings of the page, and the sum of
      a staff cannot choose between them -- it knows how much music is in the
      bar and nothing about where any of it falls. Where more than one fits, the
      **moments** are asked (`_moments_agree`), and one is taken only if exactly
      one of them leaves the staves dating every moment they share alike. That
      is a second piece of evidence and not a tie-break: nothing here compares
      two candidates' probabilities, and a bar where the moments cannot choose
      either is still refused.
    - **Never let the moments decide a bar holding a rest** (`_holds_a_rest`).
      A rest may not be the silence of the stream it stands in, so a moment
      holding one is a column the tokens are known not to line up.
    - **Never a different kind of symbol.** A note stays a note and a rest a
      rest: this is correcting a value, not deciding that something else was
      printed.
    """
    bars = _bar_boundaries(voice)
    repairs = {}
    for bar_number, (span, target) in enumerate(
        zip(bars, _bar_targets(voice, bars), strict=True), start=1
    ):
        if target is None:
            continue
        repair = _repair_for_bar(voice, span, target)
        if repair is not None:
            chord_index, symbol_index, rhythm, why = repair
            repairs[(chord_index, symbol_index)] = (rhythm, why, bar_number)
    if not repairs:
        return voice
    out = []
    for chord_index, chord in enumerate(voice):
        symbols = list(chord.symbols)
        for symbol_index, symbol in enumerate(symbols):
            accepted_repair = repairs.get((chord_index, symbol_index))
            if accepted_repair is None:
                continue
            rhythm, why, bar_number = accepted_repair
            if changes is not None:
                changes.append(
                    ReconstructionChange(
                        kind="rhythm_repair",
                        bar=bar_number,
                        group=chord_index,
                        symbol=symbol_index,
                        staff=symbol.position,
                        pitch=symbol.pitch,
                        before=symbol.rhythm,
                        after=rhythm,
                        reason=why,
                    )
                )
            eprint(
                f"Bar arithmetic: reading {symbol.pitch} as {rhythm} rather than "
                f"{symbol.rhythm}, {why}"
            )
            symbols[symbol_index] = symbol.change_rhythm(rhythm)
        out.append(SymbolChord(symbols, chord.tuplet_mark))
    return out


def _bar_targets(voice: list[SymbolChord], bars: list[tuple[int, int]]) -> list[Fraction | None]:
    """What each bar of the stream ought to measure, where anything says so.

    Read off the bars themselves rather than off a time signature, because a
    crop of one printed system usually declares none: the length that the most
    consecutive bars **every staff agrees about** come to is what a bar is here.
    At least two such bars have to agree, or a single clean bar beside a pickup
    would be enough to call the pickup wrong.

    Taken per span between signatures, so a page that changes meter is measured
    against the meter it changed to, and never for the bar that opens a span.
    """
    spans: list[list[int]] = []
    current: list[int] = []
    for index, (start, end) in enumerate(bars):
        opens = any(
            chord.symbols and chord.symbols[0].rhythm.startswith("timeSignature")
            for chord in voice[start:end]
        )
        if opens and current:
            spans.append(current)
            current = []
        current.append(index)
    if current:
        spans.append(current)

    targets: list[Fraction | None] = [None] * len(bars)
    for span in spans:
        agreed = [
            length
            for length in (_corroborated_length(voice, bars[index]) for index in span)
            if length is not None
        ]
        if not agreed:
            continue
        target = prevailing_length(agreed)
        if agreed.count(target) < _REPAIR_WITNESS_BARS:
            continue
        for index in span[1:]:
            targets[index] = target
    return targets


_REPAIR_WITNESS_BARS = 2


def _repair_for_bar(
    voice: list[SymbolChord], span: tuple[int, int], target: Fraction
) -> tuple[int, int, str, str] | None:
    """The one alternative that makes this bar add up, where there is exactly one.

    Where more than one does, the moments are asked as well: see
    `_moments_agree`. That is a second reading of the page rather than a
    tie-break, and it only ever speaks where the arithmetic has already refused.
    """
    lengths = _staff_lengths(voice, span)
    if len(lengths) < 2:
        return None
    adrift = [position for position, length in lengths.items() if length != target]
    if len(adrift) != 1:
        return None
    position = adrift[0]
    for chord in voice[span[0] : span[1]]:
        for symbol in chord.symbols:
            if symbol.rhythm.startswith(("note", "rest")) and not _plain_value(symbol.rhythm):
                return None
    found = []
    for chord_index in range(span[0], span[1]):
        chord = voice[chord_index]
        for symbol_index, symbol in enumerate(chord.symbols):
            if symbol.position != position or not symbol.rhythm.startswith(("note", "rest")):
                continue
            if not _stands_alone(chord, symbol):
                continue
            for candidate in _rhythm_alternatives(symbol):
                swapped = _length_with(voice, span, position, chord_index, symbol_index, candidate)
                if swapped == target:
                    found.append((chord_index, symbol_index, candidate))
    if len(found) == 1:
        return (*found[0], "the only alternative that makes its bar add up")
    if not found or _holds_a_rest(voice, span):
        return None
    agreeing = [candidate for candidate in found if _moments_agree(voice, span, *candidate)]
    if len(agreeing) != 1:
        return None
    return (
        *agreeing[0],
        f"the only one of {len(found)} that make its bar add up which also leaves "
        "the staves agreeing about when each shared moment sounds",
    )


def _holds_a_rest(voice: list[SymbolChord], span: tuple[int, int]) -> bool:
    """Whether this bar has a rest in it, which stops the moments deciding.

    homr's token language has no voice -- upstream say so themselves in
    liebharc/homr#126 -- so a printed rest and the notes of the voice engraved
    **beside** it come out in one stream. A rest is therefore the least
    trustworthy thing in the bar: it may not be the silence of the stream it
    stands in, which is the whole reason `disown_silence` exists, and a staff
    whose length is mostly a rest's is a length about the rest.

    That is not a reason to distrust the arithmetic -- it is why `hanget-soi`
    bar 2's staves disagree at a moment under the reading the page prints, and
    that repair is right and stands. It is a reason not to let the **moments**
    decide, since a moment holding a rest is a column the tokens have already
    been shown not to line up. On `virta-scratch-s7`'s last bar they picked a
    candidate confidently and wrongly, shortening a dotted quarter the page
    prints in a bar of two rests, in a piece whose bars are genuinely uneven --
    the one place on the corpus where this cost notes.
    """
    return any(
        symbol.rhythm.startswith("rest")
        for chord in voice[span[0] : span[1]]
        for symbol in chord.symbols
    )


def _moments_agree(
    voice: list[SymbolChord],
    span: tuple[int, int],
    at_chord: int,
    at_symbol: int,
    rhythm: str,
) -> bool:
    """Whether the staves would still date every moment they share alike.

    The arithmetic sums a staff and knows nothing about *when* inside the bar
    anything falls, so two candidates that each take the same amount off the
    same staff look identical to it. On `hanget-soi` bar 3 the bass is a
    sixteenth short and four alternatives close it, two of them a genuine
    reading of that voice: the page's own -- the first of two beamed eighths
    read as a sixteenth -- and the second of them read as a dotted eighth
    instead. Both make the staff measure two quarters, and #203 refused the bar
    rather than choose, which was right on the evidence it had.

    There is other evidence, and homr has already read it. The tokens are read
    across the page, so the symbols of one moment are printed above one another
    and **sound together** -- the claim `disown_silence` already rests on. The
    treble of that bar adds up, and it dates the moment the bass's last note
    stands in at beat 1.5. Only the page's own reading puts it there; the dotted
    eighth puts it at 1.25, in a column the other staff says is 1.5. So the
    moments pick one, and they pick the one printed.

    **It is asked as a discriminator and never as a precondition**, which is the
    part that has to stay true. A moment is where homr thinks two symbols line
    up, and it is only approximately that: in bar 2 of the same fixture a
    sixteenth rest printed at beat 1.5 shares a moment with a bass quarter
    printed at 1, so the staves disagree there under the reading the page
    prints. Vetoing on that would take back #203's repair. Good enough to
    choose between readings, then, and not good enough to refuse the only one.
    """
    cursors: dict[str, Fraction] = {}
    for chord_index in range(span[0], span[1]):
        sounding: dict[str, list[Fraction]] = {}
        for symbol_index, symbol in enumerate(voice[chord_index].symbols):
            if not symbol.rhythm.startswith(("note", "rest")):
                continue
            swapped = chord_index == at_chord and symbol_index == at_symbol
            duration = (EncodedSymbol(rhythm) if swapped else symbol).get_duration().fraction
            sounding.setdefault(symbol.position, []).append(duration)
        standing = {cursors.get(position, Fraction(0)) for position in sounding}
        if len(standing) > 1:
            return False
        for position, durations in sounding.items():
            cursors[position] = cursors.get(position, Fraction(0)) + min(durations)
    return True


def _plain_value(rhythm: str) -> bool:
    """A note or rest written as a plain value: no tuplet, grace note or multirest."""
    kern = rhythm.split("_", 1)[1] if "_" in rhythm else ""
    if not kern or "G" in kern or kern.endswith("m"):
        return False
    return not EncodedSymbol(rhythm).is_tuplet()


def _stands_alone(chord: SymbolChord, symbol: EncodedSymbol) -> bool:
    """Whether this note can have a value of its own, or is a head of a chord.

    Two notes of one moment on one staff drawn with the same stem are one chord
    and are written as one. A note whose stem nothing was matched to is no
    evidence, so it stands alone only when nothing shares its moment.
    """
    others = [
        other
        for other in chord.symbols
        if other is not symbol
        and other.position == symbol.position
        and other.rhythm.startswith(("note", "rest"))
    ]
    if not others:
        return True
    if symbol.stem_direction is None:
        return False
    return all(other.stem_direction != symbol.stem_direction for other in others)


def _rhythm_alternatives(symbol: EncodedSymbol) -> list[str]:
    """The decoder's other readings of this symbol, of the same kind as the one it took."""
    confidence = symbol.confidence or {}
    alternatives = confidence.get("rhythm", {}).get("alternatives", [])
    kind = "note" if symbol.rhythm.startswith("note") else "rest"
    return [
        alternative["value"]
        for alternative in alternatives
        if alternative["value"] != symbol.rhythm
        and alternative["value"].startswith(kind)
        and _plain_value(alternative["value"])
    ]


def _length_with(
    voice: list[SymbolChord],
    span: tuple[int, int],
    position: str,
    at_chord: int,
    at_symbol: int,
    rhythm: str,
) -> Fraction:
    """What that staff would measure with this one symbol read as `rhythm` instead."""
    total = Fraction(0)
    for chord_index in range(span[0], span[1]):
        durations = []
        for symbol_index, symbol in enumerate(voice[chord_index].symbols):
            if symbol.position != position or not symbol.rhythm.startswith(("note", "rest")):
                continue
            if chord_index == at_chord and symbol_index == at_symbol:
                durations.append(EncodedSymbol(rhythm).get_duration().fraction)
            else:
                durations.append(symbol.get_duration().fraction)
        if durations:
            total += min(durations)
    return total


def infer_meter_changes(
    voice: list[SymbolChord], changes: list[ReconstructionChange] | None = None
) -> list[SymbolChord]:
    """Write the time signature at a bar that plainly changed meter and read none.

    The vocabulary holds only denominators, so a change of numerator alone -- 3/4
    to 5/4, which is what a page does when it adds a beat -- is not something the
    model can emit even when it reads the bar perfectly. On the `sammon-ryosto`
    fixture both bars of the opening span are measured exactly right, 3 quarters
    and 5, and both came out declared the same, because one number was being
    fitted to a span that holds two.

    So a bar whose length every staff agrees on, and which is not the length the
    signature in force declares, gets that signature written at it. Three things
    it will not do, each because the alternative is worse than the fault it fixes:

    - **Never the first bar of a span.** That bar is where a printed signature
      stands, and an anacrusis is exactly a first bar shorter than its meter. A
      pickup would otherwise be read as the meter and the meter as a change.
    - **Never off one staff.** See `_corroborated_length`.
    - **Never a denominator.** The one in force is carried, because a numerator
      is what a bar length can settle and a denominator is a spelling the same
      bar length has more than one of.
    """
    fallback = find_division_and_time_signature_nominator(voice)[1]
    declared = list(find_nominator_per_time_signature(voice, fallback))
    bars = _bar_boundaries(voice)
    if not declared or len(bars) < 2:
        return voice

    denominator = "4"
    span_no = -1
    opens_span = True
    inserted: dict[int, SymbolChord] = {}
    for bar_number, (start, end) in enumerate(bars, start=1):
        signatures = [
            chord.symbols[0]
            for chord in voice[start:end]
            if chord.symbols and chord.symbols[0].rhythm.startswith("timeSignature")
        ]
        if signatures:
            denominator = signatures[-1].rhythm.split("/")[1]
            span_no += 1
            opens_span = True
        if opens_span or span_no < 0 or span_no >= len(declared):
            opens_span = False
            continue
        length = _corroborated_length(voice, (start, end))
        if length is None or length == declared[span_no]:
            continue
        before = f"{int(declared[span_no] * int(denominator))}/{denominator}"
        after = f"{int(length * int(denominator))}/{denominator}"
        reason = "every detected staff agrees on the bar length"
        if changes is not None:
            changes.append(
                ReconstructionChange(
                    kind="meter_inference",
                    bar=bar_number,
                    group=start,
                    symbol=None,
                    staff=None,
                    pitch=None,
                    before=before,
                    after=after,
                    reason=reason,
                )
            )
        eprint(
            f"Bar length {length} contradicts the {declared[span_no]} in force and every "
            f"staff agrees on it: writing a {after}"
        )
        inserted[start] = SymbolChord([EncodedSymbol(f"timeSignature/{denominator}")])
        declared[span_no] = length

    if not inserted:
        return voice
    out: list[SymbolChord] = []
    for index, chord in enumerate(voice):
        if index in inserted:
            out.append(inserted[index])
        out.append(chord)
    return out


def find_nominator_per_time_signature(
    voice: list[SymbolChord], fallback: Fraction
) -> list[Fraction]:
    """A numerator for each stretch between time signature changes.

    The model reads only the **denominator** of a time signature; the numerator
    is inferred from how long the bars actually are. Inferring it once for the
    whole voice makes a changing meter impossible to express — every signature
    then gets the same numerator and differs only in its denominator, so a part
    that goes 3/4, 5/4, 5/2 can have at most one of them right.

    Measured on a real system of Sammon ryösto: one median gave 7/4, then no
    change at all where the page changes to 5/4, then 3/2 where the page says
    5/2. Not one of the three.

    So the median is taken over each span between signatures instead. A span
    with no complete bar in it falls back to the voice-wide figure rather than
    inventing something out of nothing.

    This is still inference and not reading. A span whose bars homr measured
    wrongly gets a numerator that is wrong in the same way — what it removes is
    the case that could not be right however well the page was read.
    """
    spans: list[list[Fraction]] = []
    current: list[Fraction] = []
    in_measure = Fraction(0)
    started = False
    for chord in voice:
        if chord.symbols and chord.symbols[0].rhythm.startswith("timeSignature"):
            if started:
                if in_measure > Fraction(0):
                    current.append(in_measure)
                    in_measure = Fraction(0)
                spans.append(current)
                current = []
            started = True
            continue
        if chord.is_barline() and in_measure > Fraction(0):
            current.append(in_measure)
            in_measure = Fraction(0)
        else:
            duration = chord.get_duration()
            if duration > Fraction(0):
                in_measure += duration
    if in_measure > Fraction(0):
        current.append(in_measure)
    if started:
        spans.append(current)
    return [prevailing_length(span) if span else fallback for span in spans]


def prevailing_length(lengths: list[Fraction]) -> Fraction:
    """The length the most consecutive bars agree on; on a tie, the earliest run.

    Not the median, which cannot see order and is wrong at both ends of it. An
    anacrusis makes the opening bar short, and a span of `1/4, 4/4, 4/4, 4/4` has
    a median of a whole -- right by luck, since the odd bar sits at an end. A span
    of `3/4, 5/4` has a median of a whole and neither bar is a whole: a run says
    3/4, which is the bar the printed signature actually opens, and leaves the
    other to `infer_meter_changes` rather than averaging the two into a meter the
    page never prints.

    A run rather than a mode because a meter is what consecutive bars agree on. A
    span alternating 3/4 and 4/4 has no majority and no prevailing meter either;
    taking the first run says so by being wrong in one place instead of
    everywhere.
    """
    best, longest = lengths[0], 0
    current, run = None, 0
    for length in lengths:
        run = run + 1 if length == current else 1
        current = length
        if run > longest:
            best, longest = length, run
    return best


def find_division_and_time_signature_nominator(voice: list[SymbolChord]) -> tuple[int, Fraction]:
    durations = [Fraction(1, 4)]
    duration_in_measure = Fraction(0)
    measure_duration = []
    for chord in voice:
        if chord.is_barline() and duration_in_measure > Fraction(0):
            measure_duration.append(duration_in_measure)
            duration_in_measure = Fraction(0)
        else:
            duration = chord.get_duration()
            if duration > Fraction(0):
                durations.append(duration)
                duration_in_measure += duration

    if duration_in_measure > Fraction(0):
        measure_duration.append(duration_in_measure)

    if len(measure_duration) == 0:
        return find_common_division(durations), Fraction(1)

    nominator: Fraction = np.median(measure_duration)  # type: ignore

    return find_common_division(durations), nominator


def group_into_chords(voice: list[EncodedSymbol]) -> list[SymbolChord]:
    return [SymbolChord(s) for s in sort_token_chords(voice)]


class TupletParser:
    @staticmethod
    def parse(groups: list[SymbolChord]) -> list[SymbolChord]:
        for measure_groups in TupletParser.split_into_measures(groups):
            saved_marks = [group.tuplet_mark for group in measure_groups]
            if TupletParser.add_tuplets(measure_groups):
                continue
            for group, mark in zip(measure_groups, saved_marks, strict=True):
                group.tuplet_mark = mark
        return groups

    @staticmethod
    def get_tuplet_duration(group: SymbolChord) -> SymbolDuration | None:
        for symbol in group.symbols:
            if symbol.rhythm.startswith(("note", "rest")):
                duration = symbol.get_duration()
                if duration.normal_notes != duration.actual_notes:
                    return duration
        return None

    @staticmethod
    def split_into_measures(groups: list[SymbolChord]) -> list[list[SymbolChord]]:
        measures: list[list[SymbolChord]] = []
        current_measure: list[SymbolChord] = []
        for group in groups:
            current_measure.append(group)
            if group.is_barline():
                measures.append(current_measure)
                current_measure = []
        if current_measure:
            measures.append(current_measure)
        return measures

    @staticmethod
    def add_tuplets(groups: list[SymbolChord]) -> bool:
        cursor = 0
        while cursor < len(groups):
            duration = TupletParser.get_tuplet_duration(groups[cursor])

            if duration is None:
                cursor += 1
                continue

            start = cursor
            tuplet_format = (duration.actual_notes, duration.normal_notes)
            tuplet_size = duration.actual_notes

            while cursor - start < tuplet_size:
                if cursor >= len(groups):
                    return False
                current_duration = TupletParser.get_tuplet_duration(groups[cursor])
                if current_duration is None:
                    return False
                current_format = (current_duration.actual_notes, current_duration.normal_notes)
                if current_format != tuplet_format:
                    return False
                cursor += 1

            groups[start].tuplet_mark = "start"
            groups[cursor - 1].tuplet_mark = "stop"

        return True


def add_tuplet_start_stop(groups: list[SymbolChord]) -> list[SymbolChord]:
    return TupletParser.parse(groups)


@dataclass(frozen=True)
class ReconstructedVoice:
    """The complete token-level musical structure ready for serialization."""

    groups: list[SymbolChord]
    division: int
    nominator: Fraction
    nominators: list[Fraction]
    changes: tuple[ReconstructionChange, ...]


def reconstruct_voice(voice: list[EncodedSymbol]) -> ReconstructedVoice:
    """Apply homr's musical reconstruction passes in their established order."""
    changes: list[ReconstructionChange] = []
    groups = group_into_chords(voice)
    groups = repair_bar_arithmetic(groups, changes)
    groups = add_tuplet_start_stop(groups)
    groups = infer_meter_changes(groups, changes)
    division, nominator = find_division_and_time_signature_nominator(groups)
    return ReconstructedVoice(
        groups=groups,
        division=division,
        nominator=nominator,
        nominators=find_nominator_per_time_signature(groups, nominator),
        changes=tuple(changes),
    )
