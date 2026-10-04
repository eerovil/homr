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
import re
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol, SymbolDuration, sort_token_chords

#: The order a moment's positions are written in. Anything unexpected is read as the
#: lower staff, which is what the old upper/lower split did.
_POSITION_ORDER = ("upper", "upper2", "lower", "lower2")


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
        """One chord per position (upper, upper2, lower, lower2), in that order.

        A position holding only rests goes first: the last chord is the one which
        advances time, so a rest written last would push the notes beside it later.
        """
        buckets: dict[str, list[EncodedSymbol]] = defaultdict(list)
        for symbol in self.symbols:
            position = symbol.position if symbol.position in _POSITION_ORDER else "lower"
            buckets[position].append(symbol)
        chords = [
            SymbolChord(buckets[position], self.tuplet_mark)
            for position in _POSITION_ORDER
            if buckets[position]
        ]
        chords.sort(
            key=lambda chord: all(s.rhythm.startswith("rest") for s in chord.symbols), reverse=True
        )
        return chords


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

    A moment costs a staff the shortest of *its* notes there, so a staff holding a
    whole note against four quarters measures a whole. (The writer used to keep a
    cursor per staff the same way; since eerovil/musescore-choir-plugins#220 it runs
    upstream's one clock for the part, and this stays the per-staff measure the
    meter is inferred from.)
    """
    by_position: dict[str, Fraction] = {}
    for chord in voice[span[0] : span[1]]:
        sounding: dict[str, list[Fraction]] = {}
        for symbol in chord.symbols:
            if not symbol.rhythm.startswith(("note", "rest")):
                continue
            sounding.setdefault(symbol.position, []).append(symbol.get_duration().fraction)
        for position, durations in sounding.items():
            timed = [duration for duration in durations if duration > 0]
            if timed:
                by_position[position] = by_position.get(position, Fraction(0)) + min(timed)
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


def advance_to_next_group(
    group: SymbolChord, clock: Fraction, sounding: list[Fraction]
) -> Fraction:
    """How far the next group starts after this one, updating the sounding notes in place.

    Tokens say which notes start together, not when each group starts: a group starts
    when the earliest still-sounding note ends. The writer keeps time with this, and so
    does `repair_tuplet_overlaps`, so the two cannot disagree about when a note sounds.
    """
    durations = [
        s.get_duration().fraction for s in group.symbols if s.rhythm.startswith(("note", "rest"))
    ]
    timed = [d for d in durations if d > 0]  # grace notes have no duration and take no time
    if not timed:
        return Fraction(0)
    sounding.extend(clock + d for d in timed)
    return min(end for end in sounding if end > clock) - clock


def _onsets(
    voice: list[SymbolChord],
    span: tuple[int, int],
    swap: tuple[int, int, str] | None = None,
) -> list[Fraction]:
    """When each moment of this bar starts on the writer's clock, one symbol optionally reread.

    One entry more than the bar has moments: the last is where the bar ends.
    """
    clock = Fraction(0)
    sounding: list[Fraction] = []
    onsets = []
    for chord_index in range(span[0], span[1]):
        chord = voice[chord_index]
        if swap is not None and swap[0] == chord_index:
            symbols = list(chord.symbols)
            symbols[swap[1]] = symbols[swap[1]].change_rhythm(swap[2])
            chord = SymbolChord(symbols, chord.tuplet_mark)
        onsets.append(clock)
        clock += advance_to_next_group(chord, clock, sounding)
        sounding[:] = [end for end in sounding if end > clock]
    onsets.append(clock)
    return onsets


def _timed(symbol: EncodedSymbol) -> bool:
    return symbol.rhythm.startswith(("note", "rest")) and symbol.get_duration().fraction > 0


def _tuplet_twin(rhythm: str, alternative: str) -> bool:
    """Whether two readings differ only in whether the note is a tuplet's."""
    if rhythm == alternative:
        return False
    return (
        EncodedSymbol(alternative).remove_tuplet().rhythm == rhythm
        or EncodedSymbol(rhythm).remove_tuplet().rhythm == alternative
    )


def repair_tuplet_overlaps(
    voice: list[SymbolChord], changes: list[ReconstructionChange] | None = None
) -> list[SymbolChord]:
    """Read a note as a tuplet's (or not) when that is what lets its own voice go on.

    One voice cannot sound a new note while its last one is still sounding. On
    Sangerhilsen bars 21 and 37 (eerovil/musescore-choir-plugins#240) both voices of
    each staff sing a triplet of eighths in unison; the decoder read three of its
    four heads at the first moment as triplet eighths and the fourth as a plain
    eighth. That eighth was still sounding when its own voice's second triplet note
    began, the writer's clock started the third note late to make room, and the bar
    came out 9/8 long under 4/4 -- in every part, once the voices were split.

    So a note still sounding when the next note of its own voice starts is read the
    other way round -- as its tuplet value, or its tuplet value as plain -- when all
    three hold:

    - **the decoder offered that reading** among its own alternatives -- or the
      same notehead, read into the other voice of its staff, carries it
      (`_same_head_values`);
    - **the same moment already holds that value**, read by another head: the
      triplet is on the page, this head merely lost it;
    - **it ends the note exactly when its voice next sounds**, on the writer's own
      clock re-run with the change -- or, for its voice's last note in the bar,
      exactly when the bar ends (eerovil/musescore-choir-plugins#245: Legenda
      bar 2's last triplet eighth ran an eighth past its barline).

    Nothing else about the note changes, and no other value is tried: a voice that
    overlaps itself for any other reason -- two voices sharing one position label,
    which happens -- is left alone rather than shortened to fit.
    """
    repairs: dict[tuple[int, int], tuple[str, int]] = {}
    for bar_number, span in enumerate(_bar_boundaries(voice), start=1):
        onsets = _onsets(voice, span)
        moments = list(range(span[0], span[1]))
        # The first moment where some voice is out of time. Before it the clock
        # can be trusted; at and after it a value read off the clock may only be
        # fitting another note's mistake, so `_inside_its_triplet` waits for it.
        trouble_from: int | None = None
        for offset, chord_index in enumerate(moments):
            chord = voice[chord_index]
            for symbol_index, symbol in enumerate(chord.symbols):
                if not _timed(symbol):
                    continue
                following = next(
                    (
                        later
                        for later in range(offset + 1, len(moments))
                        if any(
                            _timed(other) and other.position == symbol.position
                            for other in voice[moments[later]].symbols
                        )
                    ),
                    # Its voice's last note in the bar: it must end with the bar.
                    len(moments),
                )
                ends_now = onsets[offset] + symbol.get_duration().fraction
                overlaps = ends_now > onsets[following]
                # A gap before its voice's next note, with nothing of its own filling it.
                leaves_gap = ends_now < onsets[following] and following < len(moments)
                if not overlaps and not leaves_gap:
                    continue
                if leaves_gap and _partner_sings_between(
                    voice, moments[offset + 1 : following], symbol.position
                ):
                    # The other voice of the staff sings in the gap: this voice
                    # most likely lost its copy of that note, not its own length.
                    continue
                clock_trusted = trouble_from is None or trouble_from == offset
                if trouble_from is None:
                    trouble_from = offset
                beside = {
                    other.rhythm for other in chord.symbols if other is not symbol and _timed(other)
                }
                fits = []
                for candidate in _all_alternatives(symbol) + _same_head_values(chord, symbol):
                    if candidate in fits:
                        continue
                    # A gap is only ever closed by the stricter reading below: the
                    # twin rule dates the note off a clock it moves itself, which an
                    # overlap can afford and a gap cannot.
                    twin_beside = (
                        overlaps and _tuplet_twin(symbol.rhythm, candidate) and candidate in beside
                    )
                    if not twin_beside:
                        if not clock_trusted or not _inside_its_triplet(chord, symbol, candidate):
                            continue
                        # Read only off the moments as they stand: shortening a note
                        # also moves the clock, and a value that "fits" only by
                        # dragging its voice's next note along with it fits nothing.
                        lasts = EncodedSymbol(candidate).get_duration().fraction
                        if onsets[offset] + lasts != onsets[following]:
                            continue
                    swapped = _onsets(voice, span, (chord_index, symbol_index, candidate))
                    ends = swapped[offset] + EncodedSymbol(candidate).get_duration().fraction
                    if ends == swapped[following]:
                        fits.append(candidate)
                if len(fits) == 1:
                    repairs[(chord_index, symbol_index)] = (fits[0], bar_number)
    if not repairs:
        return voice
    out = []
    for chord_index, chord in enumerate(voice):
        symbols = list(chord.symbols)
        for symbol_index, symbol in enumerate(symbols):
            repair = repairs.get((chord_index, symbol_index))
            if repair is None:
                continue
            rhythm, bar_number = repair
            why = "it was still sounding when its own voice next sounds"
            if changes is not None:
                changes.append(
                    ReconstructionChange(
                        kind="tuplet_repair",
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
            eprint(f"Tuplet: reading {symbol.pitch} as {rhythm} rather than {symbol.rhythm}, {why}")
            symbols[symbol_index] = symbol.change_rhythm(rhythm)
        out.append(SymbolChord(symbols, chord.tuplet_mark))
    return out


def _partner_sings_between(voice: list[SymbolChord], between: list[int], position: str) -> bool:
    partner = position[:-1] if position.endswith("2") else position + "2"
    return any(
        other.position == partner and other.rhythm.startswith("note") and _timed(other)
        for chord_index in between
        for other in voice[chord_index].symbols
    )


def _inside_its_triplet(chord: SymbolChord, symbol: EncodedSymbol, candidate: str) -> bool:
    """Whether `candidate` is a tuplet value of the kind already sounding in this moment.

    On Legenda system 2 (eerovil/musescore-choir-plugins#245) every voice of the bar
    sings triplets, but a head read as a plain quarter shares its moment only with
    triplet eighths, so its own twin is nowhere beside it. That the moment is inside
    a triplet of the same kind is the evidence instead -- read together with the
    decoder offering the value and the note then ending exactly where its voice
    goes on.
    """
    if not candidate.startswith(symbol.rhythm.split("_", 1)[0] + "_"):
        return False
    if "." in candidate or not _plain_value(EncodedSymbol(candidate).remove_tuplet().rhythm):
        return False  # a dotted triplet value is not something to guess at
    wanted = EncodedSymbol(candidate).get_duration()
    if not EncodedSymbol(candidate).is_tuplet():
        return False
    for other in chord.symbols:
        if other is symbol or not _timed(other) or not other.is_tuplet():
            continue
        theirs = other.get_duration()
        if (theirs.actual_notes, theirs.normal_notes) == (wanted.actual_notes, wanted.normal_notes):
            return True
    return False


def _staff_of(position: str) -> str:
    return position[:-1] if position.endswith("2") else position


def _same_head_values(chord: SymbolChord, symbol: EncodedSymbol) -> list[str]:
    """The values this head carries in the other voice of its own staff.

    A unison head drawn with two stems is written into both voices (#153), so the
    other copy is the same printed notehead read a second time. On Legenda bar 2
    (eerovil/musescore-choir-plugins#245) the lower voice read a beamed triplet
    eighth as `note_12` and the upper voice read the very same head as `note_8`,
    without `note_12` among its own alternatives at all. The head beside it saying
    so is the evidence the decoder's own list would have been.
    """
    return [
        other.rhythm
        for other in chord.symbols
        if other is not symbol
        and _timed(other)
        and other.pitch == symbol.pitch
        and other.position != symbol.position
        and _staff_of(other.position) == _staff_of(symbol.position)
    ]


def repair_tuplet_overlaps_until_settled(
    voice: list[SymbolChord], changes: list[ReconstructionChange] | None = None
) -> list[SymbolChord]:
    """Run `repair_tuplet_overlaps` again while it still finds something.

    Each pass dates the moments on the clock as it was read, so a run of misread
    triplet notes is repaired one note per pass: the second note only ends exactly
    where its voice goes on once the first has stopped overlapping it.
    """
    for _ in range(_MAX_REPAIR_PASSES):
        found: list[ReconstructionChange] = []
        voice = repair_tuplet_overlaps(voice, found)
        if changes is not None:
            changes.extend(found)
        if not found:
            break
    return voice


_MAX_REPAIR_PASSES = 256


def _all_alternatives(symbol: EncodedSymbol) -> list[str]:
    """Every other reading the decoder ranked under this symbol."""
    confidence = symbol.confidence or {}
    alternatives = confidence.get("rhythm", {}).get("alternatives", [])
    return [alternative["value"] for alternative in alternatives]


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
    - **Never a bar this cannot measure** -- a tuplet or a multi-measure rest
      anywhere in it, since its length would be arithmetic this does not do. A
      grace note takes no time and is measured as taking none; one read alone in
      front of a moment its own voice is missing from may be read as that voice's
      note there (`_grace_joins`), and then sounds with that moment.
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
    carried: list[EncodedSymbol] = []
    for chord_index, chord in enumerate(voice):
        symbols = carried + list(chord.symbols)
        offset = len(carried)
        carried = []
        joins = False
        for symbol_index, symbol in enumerate(symbols[offset:], start=offset):
            accepted_repair = repairs.get((chord_index, symbol_index - offset))
            if accepted_repair is None:
                continue
            joins = _is_grace(symbol)
            rhythm, why, bar_number = accepted_repair
            if changes is not None:
                changes.append(
                    ReconstructionChange(
                        kind="rhythm_repair",
                        bar=bar_number,
                        group=chord_index,
                        symbol=symbol_index - offset,
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
        if joins:
            # A grace note read as the note it is sounds with the moment after it.
            carried = symbols
            continue
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
    if _shared_rests(voice, span, position) == target - lengths[position]:
        return None
    for chord in voice[span[0] : span[1]]:
        for symbol in chord.symbols:
            if (
                symbol.rhythm.startswith(("note", "rest"))
                and not _plain_value(symbol.rhythm)
                and not _is_grace(symbol)
            ):
                return None
    found = []
    for chord_index in range(span[0], span[1]):
        chord = voice[chord_index]
        for symbol_index, symbol in enumerate(chord.symbols):
            if symbol.position != position or not symbol.rhythm.startswith(("note", "rest")):
                continue
            if not _stands_alone(chord, symbol):
                continue
            if _is_grace(symbol) and _grace_joins(voice, span, chord_index) is None:
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


def _shared_rests(voice: list[SymbolChord], span: tuple[int, int], position: str) -> Fraction:
    """How long the other voice of this staff rests where this voice has nothing.

    A choir page prints a rest both voices of a staff keep once, and the model files
    it under one of them. The other voice is then short by exactly that rest, and
    lengthening one of its notes to fill it would hold a note through a printed
    rest: on Sangerhilsen bar 20 (eerovil/musescore-choir-plugins#240) the lower
    voice's tied eighth became a half and sang on through the bar's two rests. A
    voice short by exactly what its partner rests here is not misread; it rests.
    """
    staff = position[:-1] if position.endswith("2") else position
    partner = staff if position.endswith("2") else staff + "2"
    total = Fraction(0)
    for chord in voice[span[0] : span[1]]:
        timed = [s for s in chord.symbols if s.rhythm.startswith(("note", "rest"))]
        if any(s.position == position for s in timed):
            continue
        theirs = [s for s in timed if s.position == partner]
        if theirs and all(s.rhythm.startswith("rest") for s in theirs):
            total += min(s.get_duration().fraction for s in theirs)
    return total


def _holds_a_rest(voice: list[SymbolChord], span: tuple[int, int]) -> bool:
    """Whether this bar has a rest in it, which stops the moments deciding.

    homr's token language has no voice -- upstream say so themselves in
    liebharc/homr#126 -- so a printed rest and the notes of the voice engraved
    **beside** it come out in one stream. A rest is therefore the least
    trustworthy thing in the bar: it may not be the silence of the stream it
    stands in, and a staff
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
    and **sound together**. The
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


def _is_grace(symbol: EncodedSymbol) -> bool:
    return symbol.rhythm.startswith("note") and "G" in symbol.rhythm.split("_", 1)[1]


def _grace_joins(voice: list[SymbolChord], span: tuple[int, int], chord_index: int) -> int | None:
    """The moment a grace note read alone would sound in, were it a real note.

    A grace note leads into the next note of its own voice, so a "grace note" whose
    voice has nothing in the moment after it leads into nothing: it is that voice's
    note *of* that moment, read small. On Sangerhilsen bars 15 and 31
    (eerovil/musescore-choir-plugins#240) the upper voice's accented quarter on beat
    four came out `note_4G`, alone, in front of a moment holding every other voice
    but not its own -- and the voice measured three beats of four.

    None when the grace note shares its moment with anything, or the next moment is
    not in this bar, or its own voice does sound there: then it is an ordinary
    grace note and stays one.
    """
    chord = voice[chord_index]
    if len(chord.symbols) != 1 or chord_index + 1 >= span[1]:
        return None
    position = chord.symbols[0].position
    following = voice[chord_index + 1]
    if following.is_barline() or not any(
        symbol.rhythm.startswith(("note", "rest")) for symbol in following.symbols
    ):
        return None
    if any(symbol.position == position for symbol in following.symbols):
        return None
    return chord_index + 1


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
        timed = [duration for duration in durations if duration > 0]
        if timed:
            total += min(timed)
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
            # Every note's own value, not only the chord's shortest: a value the
            # division cannot express is written as zero divisions.
            for symbol in chord.symbols:
                if symbol.rhythm.startswith(("note", "rest")):
                    frac = symbol.get_duration().fraction
                    if frac > Fraction(0):
                        durations.append(frac)
            duration = chord.get_duration()
            if duration > Fraction(0):
                duration_in_measure += duration

    if duration_in_measure > Fraction(0):
        measure_duration.append(duration_in_measure)

    if len(measure_duration) == 0:
        return find_common_division(durations), Fraction(1)

    nominator: Fraction = np.median(measure_duration)  # type: ignore

    return find_common_division(durations), nominator


def group_into_chords(voice: list[EncodedSymbol]) -> list[SymbolChord]:
    return [SymbolChord(s) for s in sort_token_chords(voice)]


def fill_unison_copies(
    voice: list[SymbolChord], changes: list[ReconstructionChange] | None = None
) -> list[SymbolChord]:
    """Give a voice that doubles its staff-mate note for note the notes it skipped.

    Either voice of the staff may be the copy: the one with fewer notes, all of
    them standing with the same pitch in the other.

    Where the two voices of a staff sing in unison, every head is read into both
    of them. On Legenda system 2, bar 1 (eerovil/musescore-choir-plugins#245) the
    decoder wrote the second voice's copy of every head in the bar but one -- the
    middle of three beamed triplet eighths -- so that voice had a hole MuseScore
    filled with a sixteenth rest the page does not print.

    A voice is a copy here only when, in this bar, it holds no rest, has at least
    two notes, and every one of them stands in a moment where the other voice of
    its staff has the same pitch. Then each note of the other voice standing in a
    moment the copy is missing from is written into the copy as well. Values are
    not touched: where the two disagree, `repair_tuplet_overlaps` settles it.
    """
    out = list(voice)
    for bar_number, span in enumerate(_bar_boundaries(voice), start=1):
        moments = _voice_moments(voice, span)
        for copy_position in list(moments):
            lead = copy_position[:-1] if copy_position.endswith("2") else copy_position + "2"
            if lead not in moments or len(moments[copy_position]) >= len(moments[lead]):
                continue
            copies = moments[copy_position]
            if len(copies) < _MIN_UNISON_NOTES:
                continue
            copied = {chord_index for chord_index, _ in copies}
            in_unison = True
            for chord_index, indices in copies:
                symbols = voice[chord_index].symbols
                pitches = {
                    s.pitch for s in symbols if s.position == lead and s.rhythm.startswith("note")
                }
                for i in indices:
                    if symbols[i].rhythm.startswith("rest") or symbols[i].pitch not in pitches:
                        in_unison = False
            if not in_unison:
                continue
            for chord_index, indices in moments[lead]:
                if chord_index in copied:
                    continue
                symbols = list(out[chord_index].symbols)
                for i in indices:
                    leading = symbols[i]
                    if not leading.rhythm.startswith("note"):
                        continue
                    twin = leading.change_rhythm(leading.rhythm)
                    twin.position = copy_position
                    symbols.append(twin)
                    why = "its voice doubles the other note for note and skipped this one"
                    if changes is not None:
                        changes.append(
                            ReconstructionChange(
                                kind="unison_fill",
                                bar=bar_number,
                                group=chord_index,
                                symbol=len(symbols) - 1,
                                staff=copy_position,
                                pitch=leading.pitch,
                                before=None,
                                after=leading.rhythm,
                                reason=why,
                            )
                        )
                    eprint(f"Unison: writing {leading.pitch} into {copy_position} as well, {why}")
                out[chord_index] = SymbolChord(symbols, out[chord_index].tuplet_mark)
    return out


_MIN_UNISON_NOTES = 2


def _plain_length(length: Fraction) -> bool:
    """Whether a length is one plain note values can spell: its denominator a power of two."""
    denominator = length.denominator
    return denominator & (denominator - 1) == 0


def _triplet_twin_of(rhythm: str) -> str | None:
    """The triplet value of a plain undotted note or rest (`note_8` -> `note_12`)."""
    match = re.fullmatch(r"(note|rest)_(\d+)", rhythm)
    if not match:
        return None
    base = int(match[2])
    if base & (base - 1) or base < 2:  # noqa: PLR2004 -- a half is the longest triplet value
        return None
    return f"{match[1]}_{base * 3 // 2}"


def _voice_moments(
    voice: list[SymbolChord], span: tuple[int, int]
) -> dict[str, list[tuple[int, list[int]]]]:
    """Per position, the moments it sounds in: (chord index, its timed symbol indices)."""
    moments: dict[str, list[tuple[int, list[int]]]] = {}
    for chord_index in range(span[0], span[1]):
        by_position: dict[str, list[int]] = {}
        for symbol_index, symbol in enumerate(voice[chord_index].symbols):
            if _timed(symbol) and not _is_grace(symbol):
                by_position.setdefault(symbol.position, []).append(symbol_index)
        for position, indices in by_position.items():
            moments.setdefault(position, []).append((chord_index, indices))
    return moments


def _moment_value(voice: list[SymbolChord], chord_index: int, indices: list[int]) -> EncodedSymbol:
    """The symbol whose value this position's moment lasts: its shortest."""
    symbols = [voice[chord_index].symbols[i] for i in indices]
    return min(symbols, key=lambda s: s.get_duration().fraction)


def _offers_triplet(voice: list[SymbolChord], chord_index: int, indices: list[int]) -> bool:
    """Whether every head of this moment is plain and offers its triplet value."""
    for symbol_index in indices:
        symbol = voice[chord_index].symbols[symbol_index]
        twin = _triplet_twin_of(symbol.rhythm)
        if twin is None or twin not in _all_alternatives(symbol):
            return False
    return True


def _apply_triplets(
    voice: list[SymbolChord],
    moments: list[tuple[int, list[int]]],
    bar_number: int,
    why: str,
    changes: list[ReconstructionChange] | None,
) -> list[SymbolChord]:
    out = list(voice)
    for chord_index, indices in moments:
        symbols = list(out[chord_index].symbols)
        for symbol_index in indices:
            symbol = symbols[symbol_index]
            twin = _triplet_twin_of(symbol.rhythm)
            if twin is None:
                continue
            if changes is not None:
                changes.append(
                    ReconstructionChange(
                        kind="tuplet_repair",
                        bar=bar_number,
                        group=chord_index,
                        symbol=symbol_index,
                        staff=symbol.position,
                        pitch=symbol.pitch,
                        before=symbol.rhythm,
                        after=twin,
                        reason=why,
                    )
                )
            eprint(f"Tuplet: reading {symbol.pitch} as {twin} rather than {symbol.rhythm}, {why}")
            symbols[symbol_index] = symbol.change_rhythm(twin)
        out[chord_index] = SymbolChord(symbols, out[chord_index].tuplet_mark)
    return out


def complete_open_triplets(
    voice: list[SymbolChord], changes: list[ReconstructionChange] | None = None
) -> list[SymbolChord]:
    """Finish a triplet the decoder started and then read on as plain notes.

    A triplet note cannot stand on its own: the bracket it is printed under closes
    only when its notes together last a plain value. On Legenda bar 2
    (eerovil/musescore-choir-plugins#245) the page prints a quarter and an eighth
    under one triplet bracket; the decoder read the quarter as `note_6` and the
    eighth as a plain `note_8`, so the triplet was left two thirds of a beat long
    and every note after it in the bar came a twelfth late.

    So while a voice's triplet is open, its following plain notes are read as
    their triplet values until the triplet closes -- but only when the decoder
    offered each of those values itself, and only when the triplet does close
    inside the bar. A triplet that would still be open at the barline, or that
    meets a note offering no triplet value, is left exactly as it was read.
    """
    bars = _bar_boundaries(voice)
    for bar_number, span in enumerate(bars, start=1):
        for moments in _voice_moments(voice, span).values():
            open_total = Fraction(0)
            pending: list[tuple[int, list[int]]] = []
            for chord_index, indices in moments:
                value = _moment_value(voice, chord_index, indices)
                if value.is_tuplet():
                    if pending:
                        break  # a run of guesses meeting a second triplet: not ours to join
                    open_total += value.get_duration().fraction
                    continue
                if _plain_length(open_total):
                    open_total = Fraction(0)
                    continue
                if not _offers_triplet(voice, chord_index, indices):
                    break
                twin = EncodedSymbol(_triplet_twin_of(value.rhythm) or value.rhythm)
                pending.append((chord_index, indices))
                open_total += twin.get_duration().fraction
                if _plain_length(open_total):
                    voice = _apply_triplets(
                        voice, pending, bar_number, "it closes the triplet open before it", changes
                    )
                    pending, open_total = [], Fraction(0)
    return voice


def triplets_onto_the_beat(
    voice: list[SymbolChord], changes: list[ReconstructionChange] | None = None
) -> list[SymbolChord]:
    """Read one run of plain notes as a triplet where that alone puts the bar on the beat.

    Under a time signature in quarters a bar is a whole number of quarters, but the
    model only reads the denominator and a triplet it misses entirely leaves no
    trace in any one note. On Legenda bar 2 (eerovil/musescore-choir-plugins#245)
    a second quarter-and-eighth triplet was read as a plain quarter and a plain
    eighth, so the voice measured nine eighths: an eighth off any beat.

    A run of notes read as a triplet is shortened by a third of what it spells, so
    a voice an amount off the beat is put back on it by the runs spelling three
    times that amount. One is read as a triplet when all of this holds:

    - a time signature is in force and this is not the bar that opens its span;
    - the voice, on its own cursor, is not a whole number of beats long;
    - **exactly one** run of consecutive plain notes in that voice spells three
      times the overrun, closes as a triplet, and offers its triplet values among
      the decoder's own alternatives, note by note.

    Two runs that would each do it are two readings of the page, and the bar is
    left alone.
    """
    bars = _bar_boundaries(voice)
    denominator: int | None = None
    opens_span = False
    for bar_number, span in enumerate(bars, start=1):
        for chord in voice[span[0] : span[1]]:
            if chord.symbols and chord.symbols[0].rhythm.startswith("timeSignature"):
                denominator = int(chord.symbols[0].rhythm.split("/")[1])
                opens_span = True
        if denominator is None or opens_span:
            opens_span = False
            continue
        beat = Fraction(1, denominator)
        lengths = _staff_lengths(voice, span)
        for position, moments in _voice_moments(voice, span).items():
            overrun = lengths.get(position, Fraction(0)) % beat
            if overrun == 0:
                continue
            found = []
            for first in range(len(moments)):
                written = Fraction(0)
                for last in range(first, len(moments)):
                    chord_index, indices = moments[last]
                    if not _offers_triplet(voice, chord_index, indices):
                        break
                    written += _moment_value(voice, chord_index, indices).get_duration().fraction
                    if written / 3 == overrun:
                        found.append(moments[first : last + 1])
                    if written / 3 >= overrun:
                        break
            if len(found) == 1:
                voice = _apply_triplets(
                    voice,
                    found[0],
                    bar_number,
                    "it is the one run whose triplet puts its bar on the beat",
                    changes,
                )
    return voice


def triplets_to_match_the_other_staff(
    voice: list[SymbolChord], changes: list[ReconstructionChange] | None = None
) -> list[SymbolChord]:
    """Read a voice's plain notes as triplets where the other staff says how long the bar is.

    On Legenda system 3, bar 7 (eerovil/musescore-choir-plugins#245) both bass
    voices read every triplet and measure a whole note, while the decoder read the
    tenors' bar -- the same three-plus-three rhythm printed above them -- as plain
    quarters and eighths after its first triplet, and hardly offered a triplet
    value for any of them. The tenor bar came out eleven eighths long.

    A bar is one length on every staff, so where every voice of the other staff
    agrees on it, a voice that runs over it has misread something. Reading a run
    of plain notes as triplets shortens it by a third of what the run spells, so
    the run that fits spells three times the overrun. One is read as triplets when:

    - every voice of the other staff measures the same length, on its own cursor;
    - this voice measures more than that;
    - **exactly one** run of consecutive plain undotted notes in this voice spells
      three times the overrun, and read as triplets it closes on a plain length.

    The decoder need not have offered the values: the other staff's bar is the
    evidence, and two runs that would each fit leave the bar alone.
    """
    out = voice
    for bar_number, span in enumerate(_bar_boundaries(voice), start=1):
        lengths = _staff_lengths(out, span)
        for position, moments in _voice_moments(out, span).items():
            staff = _staff_of(position)
            others = {length for pos, length in lengths.items() if _staff_of(pos) != staff}
            if len(others) != 1:
                continue
            target = others.pop()
            overrun = lengths.get(position, Fraction(0)) - target
            if overrun <= 0:
                continue
            found = []
            for first in range(len(moments)):
                written = Fraction(0)
                for last in range(first, len(moments)):
                    chord_index, indices = moments[last]
                    if not all(
                        _triplet_twin_of(out[chord_index].symbols[i].rhythm) for i in indices
                    ):
                        break
                    written += _moment_value(out, chord_index, indices).get_duration().fraction
                    if written / 3 == overrun and _plain_length(written * 2 / 3):
                        found.append(moments[first : last + 1])
                    if written / 3 >= overrun:
                        break
            if len(found) == 1:
                out = _apply_triplets(
                    out,
                    found[0],
                    bar_number,
                    "the other staff measures the bar and only this run as triplets fits it",
                    changes,
                )
    return out


def retime_onto_steady_voices(
    voice: list[SymbolChord], changes: list[ReconstructionChange] | None = None
) -> list[SymbolChord]:
    """Regroup a bar's moments by what each voice's own values say.

    The decoder says which symbols sound together, and that is only approximately
    true. On Legenda system 3, bar 7 (eerovil/musescore-choir-plugins#245) it put
    each tenor eighth into the moment *after* the bass eighth printed under it, so
    once the tenors' values were right the tenor still sounded a twelfth late and
    the writer's one clock, stepping at every tenor-only moment, pushed both
    staves past the barline.

    Each voice is timed by its own values from the moment it first sounds. The
    bar is regrouped by those times only when the reading then agrees with itself
    everywhere:

    - every voice ends at the same time;
    - every note of every voice starts where some voice of **another staff** has
      a note or rest starting too -- the staves line up, moment for moment;
    - the bar holds nothing but timed notes and rests, on at least two staves.

    Otherwise the bar stays as the decoder grouped it.
    """
    out: list[SymbolChord] = []
    previous = 0
    for bar_number, span in enumerate(_bar_boundaries(voice), start=1):
        out.extend(voice[previous : span[0]])
        out.extend(_retimed_bar(voice, span, bar_number, changes) or voice[span[0] : span[1]])
        previous = span[1]
    out.extend(voice[previous:])
    return out


_MIN_STAVES_TO_RETIME = 2


def _retimed_bar(
    voice: list[SymbolChord],
    span: tuple[int, int],
    bar_number: int,
    changes: list[ReconstructionChange] | None,
) -> list[SymbolChord] | None:
    chords = voice[span[0] : span[1]]
    if not chords or any(
        not (_timed(symbol) and not _is_grace(symbol))
        for chord in chords
        for symbol in chord.symbols
    ):
        return None
    onsets = _onsets(voice, span)
    timeline: dict[str, list[Fraction]] = {}
    first: dict[str, Fraction] = {}
    for offset, chord in enumerate(chords):
        by_position: dict[str, Fraction] = {}
        for symbol in chord.symbols:
            length = symbol.get_duration().fraction
            by_position[symbol.position] = min(by_position.get(symbol.position, length), length)
        for position, length in by_position.items():
            timeline.setdefault(position, []).append(length)
            first.setdefault(position, onsets[offset])
    staves = {_staff_of(position) for position in timeline}
    if len(staves) < _MIN_STAVES_TO_RETIME:
        return None
    moved: dict[str, list[Fraction]] = {}
    ends = set()
    for position, lengths in timeline.items():
        cursor = first[position]
        times = []
        for length in lengths:
            times.append(cursor)
            cursor += length
        moved[position] = times
        ends.add(cursor)
    if len(ends) != 1:
        return None
    for position, times in moved.items():
        across = {
            at
            for other, other_times in moved.items()
            if _staff_of(other) != _staff_of(position)
            for at in other_times
        }
        if not set(times) <= across:
            return None
    placed: list[tuple[Fraction, int, EncodedSymbol]] = []
    seen: dict[str, int] = {}
    order = 0
    unchanged = True
    for offset, chord in enumerate(chords):
        counted: set[str] = set()
        for symbol in chord.symbols:
            if symbol.position not in counted:
                seen[symbol.position] = seen.get(symbol.position, -1) + 1
                counted.add(symbol.position)
            at = moved[symbol.position][seen[symbol.position]]
            unchanged = unchanged and at == onsets[offset]
            placed.append((at, order, symbol))
            order += 1
    if unchanged:
        return None
    placed.sort(key=lambda item: (item[0], item[1]))
    rebuilt: list[SymbolChord] = []
    current: Fraction | None = None
    for at, _, symbol in placed:
        if at != current:
            rebuilt.append(SymbolChord([]))
            current = at
        rebuilt[-1].symbols.append(symbol)
    why = "timed by their own values, every voice ends together and meets the other staff"
    for position in sorted(moved):
        if changes is not None:
            changes.append(
                ReconstructionChange(
                    kind="retime",
                    bar=bar_number,
                    group=span[0],
                    symbol=None,
                    staff=position,
                    pitch=None,
                    before=None,
                    after="retimed",
                    reason=why,
                )
            )
    eprint(f"Retime: regrouping bar {bar_number} by each voice's values, {why}")
    return rebuilt


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
    groups = complete_open_triplets(groups, changes)
    groups = triplets_onto_the_beat(groups, changes)
    groups = fill_unison_copies(groups, changes)
    groups = repair_tuplet_overlaps_until_settled(groups, changes)
    groups = triplets_to_match_the_other_staff(groups, changes)
    groups = repair_tuplet_overlaps_until_settled(groups, changes)
    groups = retime_onto_steady_voices(groups, changes)
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
