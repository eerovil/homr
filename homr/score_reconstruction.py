"""Musical reconstruction between decoder tokens and MusicXML serialization.

This module owns the decisions that turn a flat decoder token stream into the
musical structure the serializer consumes: sounding moments, tuplet grouping,
bar-arithmetic repair, inferred meter changes, and the timing metadata derived
from those reconstructed bars. It deliberately contains no XML construction.

Keeping this phase explicit means a MusicXML serializer can be changed or
replaced without silently changing what homr believes the music is.
"""

from __future__ import annotations

import copy
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from homr.simple_logging import eprint
from homr.transformer.vocabulary import (
    EncodedSymbol,
    SymbolDuration,
    printed_meter,
    sort_token_chords,
)

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


def _signature_in(voice: list[SymbolChord], span: tuple[int, int]) -> EncodedSymbol | None:
    signatures = [
        chord.symbols[0]
        for chord in voice[span[0] : span[1]]
        if chord.symbols and chord.symbols[0].rhythm.startswith("timeSignature")
    ]
    return signatures[-1] if signatures else None


def _accepted_printed_length(
    signature: EncodedSymbol, voice: list[SymbolChord], bars: list[tuple[int, int]]
) -> Fraction | None:
    """The bar length the page's digits give this span, when the evidence allows it.

    The digits (`time_signature_reader`) and the bars the notes add up to are two
    readings of one fact, and neither is trusted alone where they disagree:

    - the digits count only when every staff that printed them read the same, and
      the model's own denominator token says the same denominator;
    - when any staff of any bar in the span adds up to the printed length, the
      notes agree, and the digits stand;
    - when none does, the digits still stand if more than one staff printed them
      -- two readings of the page against a decoder that misread the rhythms the
      same way in every bar, which is Legenda system 10's bar 22
      (eerovil/musescore-choir-plugins#267);
    - one staff's digits against at least two bars every staff agrees on at some
      other length lose: that is the notes outvoting a single glance at the page.
    """
    printed = printed_meter(signature)
    if printed is None:
        return None
    numerator, denominator = printed
    if signature.rhythm != f"timeSignature/{denominator}":
        return None
    length = Fraction(numerator, denominator)
    if any(length in _staff_lengths(voice, span).values() for span in bars):
        return length
    if len(signature.printed_meters) >= 2:  # noqa: PLR2004
        return length
    agreed = [_corroborated_length(voice, span) for span in bars]
    if any(other is not None and agreed.count(other) >= 2 for other in agreed):  # noqa: PLR2004
        eprint(
            f"Printed {numerator}/{denominator} is one staff's reading against the bars: not used"
        )
        return None
    return length


def printed_bar_lengths(voice: list[SymbolChord]) -> list[Fraction | None]:
    """Per bar, the length the printed time signature in force gives it, where accepted."""
    bars = _bar_boundaries(voice)
    lengths: list[Fraction | None] = [None] * len(bars)
    for first, last in _meter_spans(voice, bars):
        signature = _signature_in(voice, bars[first])
        if signature is None:
            continue
        length = _accepted_printed_length(signature, voice, bars[first:last])
        lengths[first:last] = [length] * (last - first)
    return lengths


def _printed_per_signature(voice: list[SymbolChord]) -> list[Fraction | None]:
    """`printed_bar_lengths` taken once per time signature, in stream order."""
    per_bar = printed_bar_lengths(voice)
    out: list[Fraction | None] = []
    for bar, span in enumerate(_bar_boundaries(voice)):
        for chord in voice[span[0] : span[1]]:
            if chord.symbols and chord.symbols[0].rhythm.startswith("timeSignature"):
                out.append(per_bar[bar])
    return out


class _Ending(Fraction):
    """When a sounding note ends, and the voice (`position`) it sounds in.

    The callers keep the sounding notes as a list of end times and drop the
    ended ones between groups; carrying the voice on the end time itself keeps
    that bookkeeping as it is, while the silent-voice rule below can still tell
    which voice a note held over from an earlier group belongs to.
    """

    position: str

    def __new__(cls, value: Fraction, position: str) -> _Ending:
        ending = super().__new__(cls, value)
        ending.position = position
        return ending


def advance_to_next_group(
    group: SymbolChord,
    clock: Fraction,
    sounding: list[Fraction],
    following: SymbolChord | None = None,
) -> Fraction:
    """How far the next group starts after this one, updating the sounding notes in place.

    Tokens say which notes start together, not when each group starts: a group starts
    when the earliest still-sounding note ends. The writer keeps time with this, and so
    does `repair_tuplet_overlaps`, so the two cannot disagree about when a note sounds.

    Given the group that ``following`` it, the next group also cannot start before
    each of *its* voices is free: a voice's next symbol does not start inside its own
    note. On Illan viimeinen tango s3 (eerovil/musescore-choir-plugins#274) three
    voices hold a dotted eighth and go on, and the fourth, misread as a plain eighth,
    is silent after it; the earliest ending alone started everyone's next note a
    sixteenth early.
    """
    # Grace notes have no duration and take no time.
    timed = [
        _Ending(clock + s.get_duration().fraction, s.position)
        for s in group.symbols
        if s.rhythm.startswith(("note", "rest")) and s.get_duration().fraction > 0
    ]
    if not timed:
        return Fraction(0)
    sounding.extend(timed)
    still = [end for end in sounding if end > clock]
    earliest = min(still)
    advance = earliest - clock
    if following is None:
        return advance
    # Each voice's latest note: the one it starts here, else one held over from an
    # earlier group. A note held over in a voice that starts another here is not
    # that voice's any more; the reading has it overlap itself, and that is left
    # to the earliest ending alone, as before.
    ends: dict[str, Fraction] = {}
    for end in timed:
        ends[end.position] = min(ends.get(end.position, end), end)
    for held in still:
        position = getattr(held, "position", None)
        if position is not None and all(position != new.position for new in timed):
            ends[position] = min(ends.get(position, held), held)
    going_on = {
        symbol.position
        for symbol in following.symbols
        if symbol.rhythm.startswith(("note", "rest")) and symbol.get_duration().fraction > 0
    }
    # Only where every voice that ends first falls silent: a voice that goes on
    # starts its next symbol where its note ends, and that is the evidence the
    # earliest ending gives; a voice that stops gives none.
    enders = {position for position, end in ends.items() if end == earliest}
    continuing = [ends[position] for position in going_on if position in ends]
    if enders and not enders & going_on and continuing:
        return max(advance, min(continuing) - clock)
    return advance


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
        following = voice[chord_index + 1] if chord_index + 1 < span[1] else None
        clock += advance_to_next_group(chord, clock, sounding, following)
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
    voice: list[SymbolChord],
    changes: list[ReconstructionChange] | None = None,
    restore_decoded: bool = False,
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

    A gap is closed one more way, for a note `voices_from_opposite_stems` took out
    of a chord (eerovil/musescore-choir-plugins#265): its value was read for the
    chord, so its own plain or triplet twin is tried when the decoder offered it
    and it ends the note exactly when its voice next sounds. The other voice
    singing in the gap does not stop this one -- that voice is the chord's other
    half, now singing its own value.

    With `restore_decoded`, which `reconstruct_voice` asks for once every other
    pass has run, that is all it does: a note an earlier pass re-valued, which
    still leaves a gap, gets back the value the decoder read when that is the one
    that closes it. Only then, because before `retime_onto_steady_voices` a gap
    can be a note grouped a moment late rather than a value read wrong (Legenda
    system 3, bar 7). On Legenda system 9, bar 18 `complete_open_triplets` read the
    second tenor's plain quarter as a triplet quarter, closing a "triplet" that
    was really the tail of one the decoder had read plain, and once
    `triplets_onto_the_beat` had mended those notes the quarter stood a third of
    a beat short of the next.

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
                # A note taken out of a chord by its stem kept the chord's value,
                # which may be its staff-mate's: the partner singing in its gap is
                # then the reason for the gap, not a copy this voice lost.
                own_value_in_doubt = leaves_gap and symbol.split_from_chord
                # A note an earlier repair gave another value, and that now leaves
                # a gap: the value the decoder read may be the one that fits.
                decoded = _decoded_rhythm(symbol)
                repaired_into_gap = (
                    restore_decoded and leaves_gap and decoded not in (None, symbol.rhythm)
                )
                if restore_decoded and not repaired_into_gap:
                    continue
                if (
                    leaves_gap
                    and not own_value_in_doubt
                    and _partner_sings_between(
                        voice, moments[offset + 1 : following], symbol.position
                    )
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
                    own_twin = (
                        clock_trusted
                        and _tuplet_twin(symbol.rhythm, candidate)
                        and (
                            (own_value_in_doubt and candidate in _all_alternatives(symbol))
                            or (repaired_into_gap and candidate == decoded)
                        )
                    )
                    if own_twin:
                        lasts = EncodedSymbol(candidate).get_duration().fraction
                        if onsets[offset] + lasts != onsets[following]:
                            continue
                    elif not twin_beside:
                        if not clock_trusted or not _inside_its_triplet(chord, symbol, candidate):
                            continue
                        # The head beside it vouches only for this note's own twin;
                        # any other value has to be one the decoder offered for it.
                        # On Legenda system 10, bar 22 the other copy of a unison F
                        # was itself misread (a triplet eighth for a triplet quarter).
                        if candidate not in _all_alternatives(symbol) and not _tuplet_twin(
                            symbol.rhythm, candidate
                        ):
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


def _decoded_rhythm(symbol: EncodedSymbol) -> str | None:
    """The value the decoder itself read for this symbol, before any repair."""
    value = (symbol.confidence or {}).get("rhythm", {}).get("value")
    return value if isinstance(value, str) else None


def _all_alternatives(symbol: EncodedSymbol) -> list[str]:
    """Every other reading the decoder ranked under this symbol."""
    confidence = symbol.confidence or {}
    alternatives = confidence.get("rhythm", {}).get("alternatives", [])
    return [alternative["value"] for alternative in alternatives]


def repair_bar_arithmetic(
    voice: list[SymbolChord],
    changes: list[ReconstructionChange] | None = None,
    system_targets: list[Fraction | None] | None = None,
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
    others = (
        system_targets
        if system_targets is not None and len(system_targets) == len(bars)
        else [None] * len(bars)
    )
    opening = next(iter(_musical_bars(voice, bars)), None)
    for bar_number, (span, own, theirs) in enumerate(
        zip(bars, _bar_targets(voice, bars), others, strict=True), start=1
    ):
        # A staff whose own voices cannot say what a bar is -- one voice, or a
        # second voice in a single bar -- may take it from the other staves of
        # the system, when every one of them agrees (`system_bar_targets`).
        # The opening bar may be a pickup; the other staves cannot say it is not.
        borrowed = None if bar_number - 1 == opening else theirs
        target = own if own is not None else borrowed
        if target is None:
            continue
        repair = _repair_for_bar(voice, span, target, alone=own is None)
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
    printed = printed_bar_lengths(voice)
    for span in spans:
        if printed[span[0]] is not None:
            # The page says it: every bar of the span, the opening one too, except
            # the very first bar of the system, which may be a pickup.
            for index in span:
                if index > 0:
                    targets[index] = printed[span[0]]
            continue
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


#: Two notes of one voice standing closer than this share of the voice's
#: ordinary spacing in the bar are one printed head read twice.
_SAME_HEAD_SHARE = 0.4


def _one_head_apart(a: EncodedSymbol, b: EncodedSymbol) -> bool:
    """Whether two notes stand on the same staff position or neighbouring ones."""
    steps = "CDEFGAB"

    def position(symbol: EncodedSymbol) -> int | None:
        pitch = symbol.pitch or ""
        if len(pitch) < 2 or pitch[0] not in steps or not pitch[1:].isdigit():
            return None
        return int(pitch[1:]) * 7 + steps.index(pitch[0])

    left, right = position(a), position(b)
    return left is not None and right is not None and abs(left - right) <= 1


def _probability(symbol: EncodedSymbol) -> float | None:
    """The decoder's probability for the rhythm it read, if it gave one."""
    rhythm = (symbol.confidence or {}).get("rhythm") if symbol.confidence else None
    value = rhythm.get("probability") if isinstance(rhythm, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def drop_double_reads(
    voice: list[SymbolChord],
    changes: list[ReconstructionChange] | None = None,
    system_targets: list[Fraction | None] | None = None,
) -> list[SymbolChord]:
    """Take out a note the decoder read twice off one printed head.

    Two notes one after another in the same voice cannot stand at the same
    place on the page, nor the second to the left of the first. On Kantajani s8
    (eerovil/musescore-choir-plugins#274) the first sopranos' sixth eighth was
    read twice, 7px apart, and the second sopranos gained an E4 placed after the
    F4 it was decoded before; each bar came out an eighth over its 5/4.

    A note is taken out only when all of these hold: it and its neighbour in the
    voice stand at one place or out of order on the page, the bar measures
    longer than its target by exactly that note's value, and it is the less
    certain of the two by the decoder's own probability. Exactly one such note in
    the bar, or nothing is done.
    """
    bars = _bar_boundaries(voice)
    others = (
        system_targets
        if system_targets is not None and len(system_targets) == len(bars)
        else [None] * len(bars)
    )
    drop: set[tuple[int, int]] = set()
    printed = printed_bar_lengths(voice)
    for bar_number, (span, own, theirs) in enumerate(
        zip(bars, _bar_targets(voice, bars), others, strict=True), start=1
    ):
        # The opening bar may be a pickup, but a pickup is short; a bar *longer*
        # than the other staves, or than the meter printed at its head, is not
        # one, and only a bar that is too long is acted on here.
        target = own if own is not None else theirs
        if target is None and bar_number == 1:
            target = printed[0]
        if target is None:
            continue
        lengths = _staff_lengths(voice, span)
        found = []
        for position, length in lengths.items():
            over = length - target
            if over <= 0:
                continue
            notes = [
                (chord_index, symbol_index, symbol)
                for chord_index in range(span[0], span[1])
                for symbol_index, symbol in enumerate(voice[chord_index].symbols)
                if symbol.position == position
                and symbol.rhythm.startswith("note")
                and not _is_grace(symbol)
                and symbol.image_coordinates is not None
                and _stands_alone(voice[chord_index], symbol)
            ]
            xs = [float(symbol.image_coordinates[0]) for _, _, symbol in notes]  # type: ignore[index]
            steps = sorted(abs(b - a) for a, b in zip(xs, xs[1:], strict=False))
            if len(steps) < 2:
                continue
            ordinary = steps[len(steps) // 2]
            for (left, right), (xa, xb) in zip(
                zip(notes, notes[1:], strict=False), zip(xs, xs[1:], strict=False), strict=True
            ):
                if xb - xa >= _SAME_HEAD_SHARE * ordinary:
                    continue
                if not _one_head_apart(left[2], right[2]):
                    # A head read twice comes back at its pitch or a step off it.
                    # Finlandia s8's baritone ends E3 A3 with the A3 placed 2px
                    # right of the E3 -- the decoder's position for a bar's last
                    # note is not always where it is printed -- and a fourth
                    # apart they are two heads.
                    continue
                certainty = (_probability(left[2]), _probability(right[2]))
                if None in certainty or certainty[0] == certainty[1]:
                    # "The less certain of the two" has to mean something: with a
                    # probability missing, or the two equal, there is no such note.
                    continue
                weaker = left if certainty[0] < certainty[1] else right  # type: ignore[operator]
                if weaker[2].get_duration().fraction == over:
                    found.append((bar_number, weaker))
        if len(found) == 1:
            _, (chord_index, symbol_index, symbol) = found[0]
            drop.add((chord_index, symbol_index))
            eprint(f"Bar {bar_number}: {symbol.pitch} read twice off one head; dropped")
            if changes is not None:
                changes.append(
                    ReconstructionChange(
                        kind="double_read",
                        bar=bar_number,
                        group=chord_index,
                        symbol=symbol_index,
                        staff=symbol.position,
                        pitch=symbol.pitch,
                        before=symbol.rhythm,
                        after="dropped",
                        reason="read twice off one printed head; the bar was over by its value",
                    )
                )
    if not drop:
        return voice
    out = []
    for chord_index, chord in enumerate(voice):
        kept = [s for i, s in enumerate(chord.symbols) if (chord_index, i) not in drop]
        if kept:
            out.append(SymbolChord(kept, chord.tuplet_mark))
    return out


def _musical_bars(groups: list[SymbolChord], bars: list[tuple[int, int]]) -> list[int]:
    """Which of the bars hold a note or rest: a clef and key before a start-repeat
    are counted as a bar of their own and hold nothing."""
    return [
        index
        for index, span in enumerate(bars)
        if any(
            symbol.rhythm.startswith(("note", "rest"))
            for chord in groups[span[0] : span[1]]
            for symbol in chord.symbols
        )
    ]


def system_bar_targets(voices: list[list[EncodedSymbol]]) -> list[list[Fraction | None]]:
    """What each bar of each staff ought to measure, by the other staves of the system.

    A staff read on its own cannot always say what a bar is: on Finlandia s8
    (eerovil/musescore-choir-plugins#274) the baritone's first eighth was read as
    a quarter, that bar ran an eighth over, and the staff had no second voice to
    corroborate a length -- while the three other staves measured four quarters
    in every bar. For each staff and bar this gives the length **every** other
    staff measures there, all of its voices agreeing, when at least two other
    staves have the bar and at least two bars of the system agree like that.
    Bars are lined up by the music they hold, so a staff whose clef and key
    stand before a start-repeat (an empty bar to the reconstruction) still lines
    up; staves holding a different number of bars give nothing. The opening bar
    gets a target too, and the repairs decide what it may be used for: it may be
    a pickup, which is short and never long.
    """
    grouped = [group_into_chords(voice) for voice in voices]
    bars = [_bar_boundaries(groups) for groups in grouped]
    musical = [_musical_bars(groups, b) for groups, b in zip(grouped, bars, strict=True)]
    empty: list[list[Fraction | None]] = [[None] * len(b) for b in bars]
    if len(voices) < 3 or len({len(m) for m in musical}) != 1:
        return empty
    agreed: list[list[Fraction | None]] = []
    printed: list[list[Fraction | None]] = []
    for groups, spans, indices in zip(grouped, bars, musical, strict=True):
        per_bar: list[Fraction | None] = []
        for index in indices:
            lengths = set(_staff_lengths(groups, spans[index]).values())
            per_bar.append(lengths.pop() if len(lengths) == 1 else None)
        agreed.append(per_bar)
        on_page = printed_bar_lengths(groups)
        printed.append([on_page[index] for index in indices])
    count = len(musical[0])
    out = empty
    for staff in range(len(voices)):
        targets: list[Fraction | None] = [None] * count
        for bar in range(count):
            theirs = {agreed[other][bar] for other in range(len(voices)) if other != staff}
            if len(theirs) == 1 and None not in theirs:
                targets[bar] = theirs.pop()
                continue
            # Where the staves disagree, a time signature read off any staff of
            # the system holds for all of them: Kantajani s8's two soprano staves
            # both read an eighth too many, and its altos' 5/4 was read off the page.
            read = {printed[other][bar] for other in range(len(voices))} - {None}
            if len(read) == 1:
                targets[bar] = read.pop()
        if sum(1 for target in targets[1:] if target is not None) >= _REPAIR_WITNESS_BARS:
            for bar, index in enumerate(musical[staff]):
                out[staff][index] = targets[bar]
    return out


def _repair_for_bar(
    voice: list[SymbolChord], span: tuple[int, int], target: Fraction, alone: bool = False
) -> tuple[int, int, str, str] | None:
    """The one alternative that makes this bar add up, where there is exactly one.

    Where more than one does, the moments are asked as well: see
    `_moments_agree`. That is a second reading of the page rather than a
    tie-break, and it only ever speaks where the arithmetic has already refused.
    """
    lengths = _staff_lengths(voice, span)
    # Corroborated by the other staves of the system, one voice is enough.
    if len(lengths) < (1 if alone else 2):
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
    # A bar under digits the page prints and the notes allow is a misread bar,
    # not a change of meter.
    printed = printed_bar_lengths(voice)

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
        if printed[bar_number - 1] is not None:
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
    # Measured on the writer's own clock (`advance_to_next_group`), not by adding
    # up each moment's shortest note: where voices interleave -- an eighth-then-
    # quarter triplet against a quarter-then-eighth one, Legenda system 8
    # (eerovil/musescore-choir-plugins#245) -- that sum runs past the barline
    # and the bar is labelled 5/4 when every voice in it is 4/4.
    sounding: list[Fraction] = []
    started = False
    for chord_index, chord in enumerate(voice):
        if chord.symbols and chord.symbols[0].rhythm.startswith("timeSignature"):
            if started:
                if in_measure > Fraction(0):
                    current.append(in_measure)
                    in_measure, sounding = Fraction(0), []
                spans.append(current)
                current = []
            started = True
            continue
        if chord.is_barline() and in_measure > Fraction(0):
            current.append(in_measure)
            in_measure, sounding = Fraction(0), []
        else:
            following = voice[chord_index + 1] if chord_index + 1 < len(voice) else None
            in_measure += advance_to_next_group(chord, in_measure, sounding, following)
            sounding[:] = [end for end in sounding if end > in_measure]
    if in_measure > Fraction(0):
        current.append(in_measure)
    if started:
        spans.append(current)
    measured = [prevailing_length(span) if span else fallback for span in spans]
    printed = _printed_per_signature(voice)
    if len(printed) != len(measured):
        return measured
    return [
        page if page is not None else bars for page, bars in zip(printed, measured, strict=True)
    ]


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
    # On the writer's clock, as `find_nominator_per_time_signature` measures.
    sounding: list[Fraction] = []
    for chord_index, chord in enumerate(voice):
        if chord.is_barline() and duration_in_measure > Fraction(0):
            measure_duration.append(duration_in_measure)
            duration_in_measure, sounding = Fraction(0), []
        else:
            # Every note's own value, not only the chord's shortest: a value the
            # division cannot express is written as zero divisions.
            for symbol in chord.symbols:
                if symbol.rhythm.startswith(("note", "rest")):
                    frac = symbol.get_duration().fraction
                    if frac > Fraction(0):
                        durations.append(frac)
            following = voice[chord_index + 1] if chord_index + 1 < len(voice) else None
            duration_in_measure += advance_to_next_group(
                chord, duration_in_measure, sounding, following
            )
            sounding[:] = [end for end in sounding if end > duration_in_measure]

    if duration_in_measure > Fraction(0):
        measure_duration.append(duration_in_measure)

    if len(measure_duration) == 0:
        return find_common_division(durations), Fraction(1)

    nominator: Fraction = np.median(measure_duration)  # type: ignore

    return find_common_division(durations), nominator


def voices_from_opposite_stems(
    voice: list[SymbolChord], changes: list[ReconstructionChange] | None = None
) -> list[SymbolChord]:
    """Move a note to the voice its stem says, where its own voice holds a second stem.

    On Legenda system 9, bar 18 (eerovil/musescore-choir-plugins#265) the basses
    cross on beat 3: the second voice's G is printed above the first voice's E,
    stem down, and the E stem up. The decoder read the two heads as one chord of
    the first voice and left the second voice nothing there, so its G went into
    the wrong part, the second voice ended a beat short, and the bar was written
    with every later note of both voices early.

    A chord's heads share one stem, so two notes of one voice in one moment whose
    heads the segmentation found with **opposite** stems are not a chord: they are
    the staff's two voices meeting. The model's voice mark otherwise outranks a
    stem (#225); this is the one shape where the mark is refuted by the moment
    itself, the same argument `stem_voice_hints._pair` makes for a unison.

    Moved only when every note of that voice in the moment has a stem of its own
    (`up` or `down`, not shared, not missing), both directions are there, and the
    staff's other voice has no note in the moment. The note whose stem names the
    other voice -- down for a first voice, up for a second -- goes there. Values
    are not touched here; every note of the chord is marked `split_from_chord`,
    because the one value the decoder read for it describes at most one of them,
    and `repair_tuplet_overlaps` may then give another its own.
    """
    out = list(voice)
    for bar_number, span in enumerate(_bar_boundaries(voice), start=1):
        for chord_index in range(span[0], span[1]):
            symbols = list(out[chord_index].symbols)
            moved = False
            for staff in ("upper", "lower"):
                for position, other, foreign in (
                    (staff, staff + "2", "down"),
                    (staff + "2", staff, "up"),
                ):
                    notes = [
                        i
                        for i, symbol in enumerate(symbols)
                        if symbol.position == position and symbol.rhythm.startswith("note")
                    ]
                    if any(
                        symbol.position == other and symbol.rhythm.startswith("note")
                        for symbol in symbols
                    ):
                        continue
                    directions = [symbols[i].stem_direction for i in notes]
                    if set(directions) != {"up", "down"}:
                        continue
                    for i in notes:
                        symbol = copy.copy(symbols[i])
                        symbol.split_from_chord = True
                        if symbol.stem_direction == foreign:
                            symbol.position = other
                            why = f"its stem is {foreign} and its voice holds the other stem here"
                            if changes is not None:
                                changes.append(
                                    ReconstructionChange(
                                        kind="voice_from_stem",
                                        bar=bar_number,
                                        group=chord_index,
                                        symbol=i,
                                        staff=position,
                                        pitch=symbol.pitch,
                                        before=position,
                                        after=other,
                                        reason=why,
                                    )
                                )
                            eprint(
                                f"Voice: moving {symbol.pitch} from {position} to {other}, {why}"
                            )
                        symbols[i] = symbol
                        moved = True
            if moved:
                out[chord_index] = SymbolChord(symbols, out[chord_index].tuplet_mark)
    return out


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
            onsets = _onsets(voice, span)
            shared_only = all(
                voice[chord_index].symbols[i].stem_direction == _SHARED_STEMS
                for chord_index, indices in copies
                for i in indices
            )
            # When each of the copy's notes stops sounding: a moment the copy is
            # still holding a note through has nothing missing from it.
            held = [
                (
                    onsets[chord_index - span[0]],
                    onsets[chord_index - span[0]]
                    + max(voice[chord_index].symbols[i].get_duration().fraction for i in indices),
                )
                for chord_index, indices in copies
            ]
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
                at = onsets[chord_index - span[0]]
                if shared_only and any(
                    voice[chord_index].symbols[i].stem_direction in ("up", "down") for i in indices
                ):
                    # Finlandia s8 (eerovil/musescore-choir-plugins#274): the copy's
                    # own heads are drawn with two stems, shared by both voices; a
                    # head drawn with one stem is one voice's, not a skipped copy.
                    continue
                if any(start < at < end for start, end in held):
                    # Illan viimeinen tango s7, s8 (eerovil/musescore-choir-plugins#274):
                    # the copy holds an eighth through the lead's second sixteenth,
                    # and that sixteenth was written into it as an extra note.
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
#: The stem direction stem_voice_hints writes on a head drawn with two stems.
_SHARED_STEMS = "both"


def _plain_length(length: Fraction) -> bool:
    """Whether a length is one plain note values can spell: its denominator a power of two."""
    denominator = length.denominator
    return denominator & (denominator - 1) == 0


def _triplet_twin_of(rhythm: str) -> str | None:
    """The triplet value of a plain undotted note or rest (`note_8` -> `note_12`).

    Quarters and shorter only. A triplet of half notes is real music, but read
    off a plain half it is a claim about a whole bar, and on Legenda system 9
    (eerovil/musescore-choir-plugins#245) closing a triplet with one turned the
    held half note at the end of a bar into a triplet half and a stray rest.
    """
    match = re.fullmatch(r"(note|rest)_(\d+)", rhythm)
    if not match:
        return None
    base = int(match[2])
    if base & (base - 1) or base < _SHORTEST_TRIPLET_BASE:
        return None
    return f"{match[1]}_{base * 3 // 2}"


_SHORTEST_TRIPLET_BASE = 4


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
                # A signature read off the page after a barline opens a full bar
                # of the new meter; only the system's first bar may be a pickup.
                opens_span = bar_number == 1 or printed_meter(chord.symbols[0]) is None
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


def _triplets_close(
    voice: list[SymbolChord],
    moments: list[tuple[int, list[int]]],
    run: list[tuple[int, list[int]]],
) -> bool:
    """Whether every triplet in this voice closes once `run` is read as triplets.

    A run may finish a triplet the decoder already read -- a quarter left plain
    after a triplet eighth's partner, say -- so closure is judged over the whole
    voice: each stretch of consecutive tuplet values must add up to a plain length.
    """
    converted = {chord_index for chord_index, _ in run}
    total = Fraction(0)
    for chord_index, indices in moments:
        value = _moment_value(voice, chord_index, indices)
        if chord_index in converted:
            value = EncodedSymbol(_triplet_twin_of(value.rhythm) or value.rhythm)
        if value.is_tuplet():
            total += value.get_duration().fraction
            continue
        if not _plain_length(total):
            return False
        total = Fraction(0)
    return _plain_length(total)


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
      three times the overrun, and read as triplets it leaves every triplet of
      the voice closing on a plain length (`_triplets_close`).

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
                    run = moments[first : last + 1]
                    if written / 3 == overrun and _triplets_close(out, moments, run):
                        found.append(run)
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


def _rhythm_options(symbol: EncodedSymbol, as_decoded: bool = False) -> list[str]:
    """The value as read, and its plain/triplet twin when it has one.

    With `as_decoded`, the value the decoder itself read and its twin are offered
    too, where an earlier pass changed it: on Legenda system 11's bar 24
    (eerovil/musescore-choir-plugins#245) the passes before this one had turned
    the basses' first note into a triplet eighth, whose twin is a plain eighth,
    and the page's triplet quarter -- the twin of the quarter the decoder read --
    was no longer among the choices. Only where the page prints the bar length:
    there the length cannot be bent to fit the extra choices.
    """
    options = _plain_and_twin(symbol)
    decoded = _decoded_rhythm(symbol) if as_decoded else None
    if decoded is not None and decoded != symbol.rhythm and decoded.startswith(("note", "rest")):
        options += [o for o in _plain_and_twin(symbol.change_rhythm(decoded)) if o not in options]
    return options


def _plain_and_twin(symbol: EncodedSymbol) -> list[str]:
    options = [symbol.rhythm]
    twin = _triplet_twin_of(symbol.rhythm)
    if twin is None and symbol.is_tuplet() and "." not in symbol.rhythm:
        plain = symbol.remove_tuplet().rhythm
        if _plain_value(plain) and _triplet_twin_of(plain) == symbol.rhythm:
            twin = plain
    if twin is not None:
        options.append(twin)
    return options


_MAX_BAR_CHOICES = 1 << 14


def solve_bar_rhythms(
    voice: list[SymbolChord],
    changes: list[ReconstructionChange] | None = None,
    bar_length: Fraction | None = None,
) -> list[SymbolChord]:
    """Read a bar's plain/triplet values all at once, where only one reading is consistent.

    The note-by-note rules each need some part of the bar to be right already.
    On Legenda system 5 (eerovil/musescore-choir-plugins#245) nothing is: bar 10
    came out as plain quarters and eighths in every voice where the page prints
    triplets throughout, and in bar 11 the decoder split one printed moment in
    two and read a plain quarter as a triplet one beside a real triplet.

    So for a bar the voices disagree about, every note is tried both as read and
    as its plain/triplet twin, and a reading is kept only when:

    - every voice ends at the same time, the bar length;
    - every note of every voice starts where a note or rest of **another staff**
      can start too -- the staves line up, moment for moment;
    - of those, it changes the fewest of the values the decoder read, and it is
      the **only** such reading for each voice;
    - a voice that already ends with the bar as read is not re-read at all, and
      no voice has more than one note lengthened: one that could only fill the
      bar by stretching several is short of notes, and stays as read.

    The bar length is the one length **every** such bar between two time
    signatures can be read at, and there must be at least two of them: a single
    bar can always be made to agree with itself. Where more than one length fits them all, the bar
    length the music before this system was in (`bar_length`, passed in by the
    caller -- it reads one printed system at a time) chooses; Legenda system 8
    fits both 4/4 and 6/4, and the system before it is in 4/4. Where no single
    length is shared, or more than one reading survives, the bar is left as it
    was.
    """
    bars = _bar_boundaries(voice)
    printed = printed_bar_lengths(voice)
    beats = _beat_per_bar(voice, bars)
    found = [
        _bar_solutions(voice, span, length, beat)
        for span, length, beat in zip(bars, printed, beats, strict=True)
    ]
    lengths: list[Fraction | None] = [None] * len(bars)
    for first, last in _meter_spans(voice, bars):
        if printed[first] is not None:
            # The printed digits, weighed against the notes, settle it -- for a
            # span of one bar too, which bars alone never can.
            lengths[first:last] = [printed[first]] * (last - first)
            continue
        # The music before this system says nothing about bars after a time
        # signature printed in it: Legenda system 10 goes 3/4, 4/4, 3/4.
        hint = bar_length if first == 0 and not _declares_meter(voice, bars[0]) else None
        span_found = found[first:last]
        # The one length every bar here can be read at; a single bar alone proves nothing.
        shared = [set(sols) for sols in span_found if sols is not None]
        common = set.intersection(*shared) if shared else set()
        if len(shared) < _MIN_BARS_FOR_LENGTH and hint is None:
            common = set()
        if len(common) > 1 and hint in common:
            # More than one length fits every bar here; the music before this system decides.
            common = {hint}
        if len(common) == 1:
            lengths[first:last] = [next(iter(common))] * (last - first)
    rebuilt: list[SymbolChord] = []
    previous = 0
    for bar_number, (span, sols, length) in enumerate(
        zip(bars, found, lengths, strict=True), start=1
    ):
        rebuilt.extend(voice[previous : span[0]])
        chosen = None
        if sols is not None and length is not None and not _has_gap(voice, span):
            chosen = sols.get(length) or None  # {}: nothing to change
        if chosen is None:
            rebuilt.extend(voice[span[0] : span[1]])
        else:
            rebuilt.extend(_place_solution(voice, span, chosen, bar_number, changes))
        previous = span[1]
    rebuilt.extend(voice[previous:])
    return rebuilt


def _has_gap(voice: list[SymbolChord], span: tuple[int, int]) -> bool:
    """Whether a voice of this bar falls silent between two of its own moments.

    The decoder places every symbol in the moment it reads it at, so a voice
    that ends before its next moment starts is missing a symbol there -- on
    Illan viimeinen tango s1 (eerovil/musescore-choir-plugins#274) a unison head
    written once, for the other voice, and a sixteenth rest the decoder never
    wrote. That bar is short for want of a symbol, not because a value was
    misread, and rebuilding it one value after another closes the gap: every
    later note of the voice moves a sixteenth early.
    """
    onsets = _onsets(voice, span)
    for moments in _voice_moments(voice, span).values():
        cursor: Fraction | None = None
        for chord_index, indices in moments:
            at = onsets[chord_index - span[0]]
            if cursor is not None and at > cursor:
                return True
            cursor = at + max(
                voice[chord_index].symbols[i].get_duration().fraction for i in indices
            )
    return False


def _declares_meter(voice: list[SymbolChord], span: tuple[int, int]) -> bool:
    return any(
        chord.symbols and chord.symbols[0].rhythm.startswith("timeSignature")
        for chord in voice[span[0] : span[1]]
    )


def _meter_spans(voice: list[SymbolChord], bars: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Runs of bar indices between time signatures, each opening at one."""
    spans: list[tuple[int, int]] = []
    first = 0
    for index, span in enumerate(bars):
        if index > first and _declares_meter(voice, span):
            spans.append((first, index))
            first = index
    if bars:
        spans.append((first, len(bars)))
    return spans


def _bar_head(voice: list[SymbolChord], span: tuple[int, int]) -> int:
    """Where the bar's notes start: a clef, key or time signature in front stays put."""
    lead = span[0]
    while lead < span[1] and not any(_timed(s) for s in voice[lead].symbols):
        lead += 1
    return lead


Reading = tuple[tuple[str, ...], list[Fraction], Fraction]


def _bar_solutions(
    voice: list[SymbolChord],
    span: tuple[int, int],
    printed: Fraction | None = None,
    beat: Fraction | None = None,
) -> dict[Fraction, dict[str, Reading]] | None:
    """For each bar length, the one consistent reading of every voice, if there is one.

    None when the bar is already consistent or cannot be searched at all.

    `printed` is the bar length the page's time signature gives this bar, where
    one was read and the notes did not outvote it (`printed_bar_lengths`). At that
    length two guards are lifted, because they exist to stop the search inventing
    a length and the page has now said what it is: a unison copy is re-read with
    the voice it copies instead of being held as read, and a voice may have more
    than one triplet read back as plain. On Legenda system 10's bar 20
    (eerovil/musescore-choir-plugins#267) the basses' second voice is a unison copy
    of the first, held at its 7/8 as read, so no reading of the bar could reach
    the 3/4 the page prints; it now does. Bar 22 still has two equally good
    readings of one voice at 3/4 and is left as read: the rule that every note
    must start where one on the other staff can is what rejects the page's own
    reading there, and changing it is beyond this pass.
    """
    span = (_bar_head(voice, span), span[1])
    chords = voice[span[0] : span[1]]
    if not chords or any(
        not (_timed(symbol) and not _is_grace(symbol))
        for chord in chords
        for symbol in chord.symbols
    ):
        return None
    onsets = _onsets(voice, span)
    moments = _voice_moments(voice, span)
    if len({_staff_of(p) for p in moments}) < _MIN_STAVES_TO_SOLVE:
        return None
    starts = {p: onsets[m[0][0] - span[0]] for p, m in moments.items()}
    readings: dict[str, list[Reading]] = {}
    current: dict[str, tuple[str, ...]] = {}
    for position, steps in moments.items():
        options = [
            _rhythm_options(_moment_value(voice, chord_index, indices), printed is not None)
            for chord_index, indices in steps
        ]
        current[position] = tuple(o[0] for o in options)
        count = 1
        for option in options:
            count *= len(option)
        if count > _MAX_BAR_CHOICES:
            return None
        readings[position] = []
        for choice in _product(options):
            cursor = starts[position]
            times = []
            for rhythm in choice:
                times.append(cursor)
                cursor += EncodedSymbol(rhythm).get_duration().fraction
            readings[position].append((choice, times, cursor))
    as_read = {p: [r for r in readings[p] if r[0] == current[p]] for p in moments}
    as_read_ends = {rs[0][2] for rs in as_read.values() if rs}
    split = (
        beat is not None
        and as_read_ends == {printed}
        and any(rs and not _triplets_fill_beats(rs[0], beat) for rs in as_read.values())
    )
    if _consistent(as_read) and not split:
        chosen_as_read = {p: rs[0] for p, rs in as_read.items()}
        length = next(iter(chosen_as_read.values()))[2]
        if _total_unaligned(chosen_as_read) == 0:
            # Already right: nothing to change, but its length is evidence for the others.
            return {length: {}}
        refit = _refit_one_staff(readings, chosen_as_read, current)
        return {length: refit if refit is not None else {}}
    copies = _unison_copies(voice, moments)
    # Where the page prints the length, that is the one length tried: a voice
    # sharing a rest printed once for both may stop short of it, and a copy that
    # doubles its partner pitch for pitch may take the partner's readings
    # (`_doubled`) where its own values cannot reach it.
    doubled = _doubled(voice, moments, copies) if printed is not None else {}
    if printed is not None:
        ends = {printed}
    else:
        ends = set.intersection(*({r[2] for r in rs} for rs in readings.values()))
    solutions: dict[Fraction, dict[str, Reading]] = {}
    for end in sorted(ends):
        if not _plain_length(end):
            continue
        # A voice already ending there as read keeps its reading: only the voices
        # that do not fit the bar are re-read.
        at_end: dict[str, list[Reading]] = {}
        known = end == printed
        for p, rs in readings.items():
            fits = [r for r in rs if r[2] == end and (known or _lengthened(r, current[p]) <= 1)]
            whole_beats = (
                not known
                or beat is None
                or not as_read[p]
                or _triplets_fill_beats(as_read[p][0], beat)
            )
            if (as_read[p] and as_read[p][0][2] == end and whole_beats) or (
                p in copies and not known
            ):
                at_end[p] = as_read[p]
            elif fits:
                at_end[p] = fits
            elif known and p in doubled:
                at_end[p] = [r for r in readings[doubled[p]] if r[2] == end]
            elif as_read[p] and as_read[p][0][2] < end:
                # Short, and only reachable by stretching several notes: it has
                # lost notes, and stays as read with the hole showing.
                at_end[p] = as_read[p]
            else:
                at_end[p] = []
        if any(
            not rs or any(r[2] != end and not (r is as_read[p][0] and r[2] < end) for r in rs)
            for p, rs in at_end.items()
        ):
            continue  # some voice can neither end there nor be left short
        if known:
            if beat is not None:
                at_end = {
                    p: [r for r in rs if _triplets_fill_beats(r, beat)] or rs
                    for p, rs in at_end.items()
                }
            remaining: dict[str, list[Reading]] | None = _likeliest_readings(
                voice, moments, at_end, copies
            )
        else:
            remaining = _align(at_end)
            if remaining is not None:
                remaining = {p: _fewest_changes(rs, current[p]) for p, rs in remaining.items()}
        if remaining is not None and all(len(rs) == 1 for rs in remaining.values()):
            solutions[end] = {p: rs[0] for p, rs in remaining.items()}
    return solutions or None


def _place_solution(
    voice: list[SymbolChord],
    span: tuple[int, int],
    chosen: dict[str, Reading],
    bar_number: int,
    changes: list[ReconstructionChange] | None,
) -> list[SymbolChord]:
    """The bar rebuilt from a chosen reading: new values, moments regrouped by them."""
    lead = _bar_head(voice, span)
    head = voice[span[0] : lead]
    chords = voice[lead : span[1]]
    moments = _voice_moments(voice, (lead, span[1]))
    placed: list[tuple[Fraction, int, EncodedSymbol]] = []
    order = 0
    position_step: dict[str, int] = {}
    why = "it is the one reading of the bar in which every voice fits and lines up"
    for chord_offset, chord in enumerate(chords):
        counted: set[str] = set()
        for symbol in chord.symbols:
            position = symbol.position
            if position not in counted:
                position_step[position] = position_step.get(position, -1) + 1
                counted.add(position)
            step = position_step[position]
            choice, times, _ = chosen[position]
            value = _moment_value(voice, lead + chord_offset, moments[position][step][1])
            rhythm = choice[step]
            new_symbol = symbol
            if rhythm != value.rhythm and symbol.rhythm == value.rhythm:
                new_symbol = symbol.change_rhythm(rhythm)
                if changes is not None:
                    changes.append(
                        ReconstructionChange(
                            kind="tuplet_repair",
                            bar=bar_number,
                            group=lead + chord_offset,
                            symbol=None,
                            staff=position,
                            pitch=symbol.pitch,
                            before=symbol.rhythm,
                            after=rhythm,
                            reason=why,
                        )
                    )
                eprint(
                    f"Tuplet: reading {symbol.pitch} as {rhythm} rather than {symbol.rhythm}, {why}"
                )
            placed.append((times[step], order, new_symbol))
            order += 1
    placed.sort(key=lambda item: (item[0], item[1]))
    out: list[SymbolChord] = list(head)
    at: Fraction | None = None
    for when, _, symbol in placed:
        if when != at:
            out.append(SymbolChord([]))
            at = when
        out[-1].symbols.append(symbol)
    return out


_MIN_STAVES_TO_SOLVE = 2
_MIN_BARS_FOR_LENGTH = 2


def _lengthened(reading: Reading, current: tuple[str, ...]) -> int:
    """How many notes this reading makes longer than the decoder read them.

    One is a misread value -- B1's plain quarter read as a triplet one on
    Legenda system 5, bar 11. More than one, to fill a voice that ends short, is
    a voice that lost notes having its others stretched over the hole: on
    system 9, bar 18 (eerovil/musescore-choir-plugins#245) the decoder dropped
    B2's last two notes, and stretching three others "fixed" the bar.
    """
    return sum(
        1
        for new, old in zip(reading[0], current, strict=True)
        if EncodedSymbol(new).get_duration().fraction > EncodedSymbol(old).get_duration().fraction
    )


def _beat_per_bar(voice: list[SymbolChord], bars: list[tuple[int, int]]) -> list[Fraction | None]:
    """The beat each bar counts in: one over the denominator of the time signature in force."""
    beats: list[Fraction | None] = []
    beat: Fraction | None = None
    for span in bars:
        for chord in voice[span[0] : span[1]]:
            if chord.symbols and chord.symbols[0].rhythm.startswith("timeSignature"):
                beat = Fraction(1, int(chord.symbols[0].rhythm.split("/")[1]))
        beats.append(beat)
    return beats


def _triplets_fill_beats(reading: Reading, beat: Fraction) -> bool:
    """Whether every run of triplet notes in this reading starts and ends on a beat.

    A page prints a triplet bracket over whole beats: Legenda system 11's bar 24
    (eerovil/musescore-choir-plugins#245) was read in the basses as two triplet
    eighths and a triplet quarter from the barline, five twelfths of a bar that
    no bracket can print, where the page has a quarter-and-eighth triplet on each
    beat. A run of shorter triplets (sixteenths) need only fill its own third.
    """
    values, times, end = reading
    run_start: Fraction | None = None
    shortest = Fraction(0)
    for value, at in [*zip(values, times, strict=True), (None, end)]:
        symbol = EncodedSymbol(value) if value is not None else None
        if symbol is not None and symbol.is_tuplet():
            duration = symbol.get_duration().fraction
            if run_start is None:
                run_start, shortest = at, duration
            shortest = min(shortest, duration)
            continue
        if run_start is not None:
            unit = min(beat, 3 * shortest)
            if run_start % unit or at % unit:
                return False
            run_start = None
    return True


def _likeliest_readings(
    voice: list[SymbolChord],
    moments: dict[str, list[tuple[int, list[int]]]],
    at_end: dict[str, list[Reading]],
    copies: set[str],
) -> dict[str, list[Reading]]:
    """Per voice, the reading of a bar of printed length the decoder itself found likeliest.

    Where the page's digits say how long the bar is, lining the staves up is the
    wrong question: Legenda system 10's bar 20 (eerovil/musescore-choir-plugins#245)
    sets a quarter-and-eighth triplet in the basses against plain quarters and an
    eighth triplet above, so the page's own reading leaves two bass notes starting
    where no tenor does, and the reading that lines up best puts the basses' last
    triplet under the tenors' one, which the page does not print. The decoder
    ranked every value it could have read for each head; the reading whose values
    it ranked highest, all heads together, is the one kept. A value it did not
    rank at all counts as half its least likely one.

    A unison copy whose pitches follow its partner's moment for moment takes the
    partner's reading: bar 22's lower bass voice was read with a value whose twin
    is not the page's, so no reading of its own can reach the page.
    """
    chosen: dict[str, list[Reading]] = {}
    heads_of: dict[str, list[list[EncodedSymbol]]] = {}
    for position, readings in at_end.items():
        heads = [
            [
                voice[chord_index].symbols[i]
                for i in indices
                if voice[chord_index].symbols[i].rhythm
                == _moment_value(voice, chord_index, indices).rhythm
            ]
            for chord_index, indices in moments[position]
        ]
        heads_of[position] = heads
        scores = [_decoder_likelihood(heads, reading[0]) for reading in readings]
        best = max(scores)
        chosen[position] = [r for r, score in zip(readings, scores, strict=True) if score == best]
    for position in sorted(copies):
        partner = position[:-1] if position.endswith("2") else position + "2"
        if partner not in chosen or position not in chosen:
            continue
        if _pitches(voice, moments[position]) != _pitches(voice, moments[partner]):
            continue
        # One line read twice: of the readings either voice settled on, the one
        # the decoder found likeliest across both voices' heads is the line's.
        candidates = {r[0]: r for r in chosen[position] + chosen[partner]}
        scored = sorted(
            candidates.values(),
            key=lambda r: _decoder_likelihood(heads_of[position], r[0])
            + _decoder_likelihood(heads_of[partner], r[0]),
            reverse=True,
        )
        best = _decoder_likelihood(heads_of[position], scored[0][0]) + _decoder_likelihood(
            heads_of[partner], scored[0][0]
        )
        top = [
            r
            for r in scored
            if _decoder_likelihood(heads_of[position], r[0])
            + _decoder_likelihood(heads_of[partner], r[0])
            == best
        ]
        chosen[position] = list(top)
        chosen[partner] = list(top)
    return chosen


def _doubled(
    voice: list[SymbolChord],
    moments: dict[str, list[tuple[int, list[int]]]],
    copies: set[str],
) -> dict[str, str]:
    """The unison copies that sing their partner's pitches moment for moment, and that partner.

    Legenda system 11's bar 25 (eerovil/musescore-choir-plugins#245): the lower
    bass voice was read with a triplet eighth whose twin is not the page's
    triplet quarter, so no reading of its own fills the 4/4 the page prints,
    and the whole bar was left as read. The upper bass voice can, and the two
    are one line.
    """
    out = {}
    for position in copies:
        partner = position[:-1] if position.endswith("2") else position + "2"
        if partner in moments and _pitches(voice, moments[position]) == _pitches(
            voice, moments[partner]
        ):
            out[position] = partner
    return out


def _pitches(voice: list[SymbolChord], steps: list[tuple[int, list[int]]]) -> list[set[str]]:
    return [
        {voice[chord_index].symbols[i].pitch for i in indices} for chord_index, indices in steps
    ]


def _decoder_likelihood(heads: list[list[EncodedSymbol]], values: tuple[str, ...]) -> float:
    """How likely the decoder found these values for these moments, in log probability."""
    total = 0.0
    for moment, value in zip(heads, values, strict=True):
        for symbol in moment:
            ranked = {
                alternative["value"]: alternative["probability"]
                for alternative in (
                    (symbol.confidence or {}).get("rhythm", {}).get("alternatives", [])
                )
            }
            floor = min(ranked.values()) / 2 if ranked else _UNRANKED_PROBABILITY
            total += math.log(max(ranked.get(value, floor), _UNRANKED_PROBABILITY))
    return total


_UNRANKED_PROBABILITY = 1e-6


def _fewest_changes(readings: list[Reading], current: tuple[str, ...]) -> list[Reading]:
    """The readings that change the fewest of the values the decoder read."""
    counts = [sum(1 for a, b in zip(r[0], current, strict=True) if a != b) for r in readings]
    least = min(counts)
    return [r for r, count in zip(readings, counts, strict=True) if count == least]


def _product(options: list[list[str]]) -> list[tuple[str, ...]]:
    result: list[tuple[str, ...]] = [()]
    for option in options:
        result = [prefix + (value,) for prefix in result for value in option]
    return result


def _align(
    readings: dict[str, list[tuple[tuple[str, ...], list[Fraction], Fraction]]],
) -> dict[str, list[tuple[tuple[str, ...], list[Fraction], Fraction]]] | None:
    """Keep each voice's readings that leave the fewest notes out of line with another staff.

    A note may legitimately start where no other staff does -- the middle of a
    beamed triplet against a quarter-and-eighth one -- so this is a minimum, not
    a rule; what it settles is which of several readings lines up best.
    """
    while True:
        changed = False
        for position, rs in readings.items():
            staff = _staff_of(position)
            across = [others for other, others in readings.items() if _staff_of(other) != staff]
            settled = [others for others in across if len(others) == 1]
            # Lean on the other staff's voices that are already settled, if any.
            possible = {at for others in settled or across for r in others for at in r[1]}
            scores = [sum(1 for at in r[1] if at not in possible) for r in rs]
            best = min(scores)
            kept = [r for r, score in zip(rs, scores, strict=True) if score == best]
            if len(kept) != len(rs):
                readings[position] = kept
                changed = True
        if not changed:
            return readings


def _unison_copies(
    voice: list[SymbolChord], moments: dict[str, list[tuple[int, list[int]]]]
) -> set[str]:
    """Voices that mostly double the other voice of their staff, note for note.

    Such a voice that does not fit the bar has most likely lost a copy of a note
    -- Sangerhilsen system 6 (eerovil/musescore-choir-plugins#245): each tenor
    voice dropped a different note of one unison triplet -- and stretching its
    other notes to fill the hole would be inventing a rhythm.
    """
    copies = set()
    for position, steps in moments.items():
        partner = position[:-1] if position.endswith("2") else position + "2"
        shared = 0
        for chord_index, indices in steps:
            mine = {voice[chord_index].symbols[i].pitch for i in indices}
            theirs = {s.pitch for s in voice[chord_index].symbols if s.position == partner}
            if mine & theirs:
                shared += 1
        if steps and shared * 2 > len(steps):
            copies.add(position)
    return copies


def _total_unaligned(chosen: dict[str, Reading]) -> int:
    """How many notes start where no voice of another staff starts."""
    total = 0
    for position, reading in chosen.items():
        staff = _staff_of(position)
        across = {at for p, r in chosen.items() if _staff_of(p) != staff for at in r[1]}
        total += sum(1 for at in reading[1] if at not in across)
    return total


def _refit_one_staff(
    readings: dict[str, list[Reading]],
    as_read: dict[str, Reading],
    current: dict[str, tuple[str, ...]],
) -> dict[str, Reading] | None:
    """Re-read one staff against the other as read, when the bar adds up but does not line up.

    On Legenda system 4, bar 9 (eerovil/musescore-choir-plugins#245) every voice
    ends on the barline, but the tenors were read as four plain eighths and a
    triplet where the page prints a triplet and two eighths -- B1's rhythm, which
    the basses were read right in. Re-reading the tenors against the basses
    lines every tenor note up with a bass note; re-reading the basses against
    the tenors cannot do as well, because B2's triplets have middles no one
    shares either way.

    Each staff in turn is re-read voice by voice: the readings of the same bar
    length that leave the fewest of its notes out of line with the other staff
    as read, and of those the one that changes least -- which must be the only
    one, and which, if it changes anything, must sing exactly the rhythm of a
    voice on the other staff (`_sings_like`). The staff whose re-reading leaves
    the fewest notes out of line in the whole bar wins, then the one that
    changes least; a tie between the staves, or no improvement on the bar as
    read, leaves the bar alone.
    """
    end = next(iter(as_read.values()))[2]
    staves = sorted({_staff_of(p) for p in readings})
    outcomes = []
    for staff in staves:
        ref = {at for p, r in as_read.items() if _staff_of(p) != staff for at in r[1]}
        others = [r for p, r in as_read.items() if _staff_of(p) != staff]
        trial = dict(as_read)
        for position, rs in readings.items():
            if _staff_of(position) != staff:
                continue
            options = [r for r in rs if r[2] == end]
            if not options:
                trial = {}
                break
            scored = [
                (sum(1 for at in r[1] if at not in ref), _changes(r, current[position]), r)
                for r in options
            ]
            lowest = min((u, c) for u, c, _ in scored)
            winners = [r for u, c, r in scored if (u, c) == lowest]
            if len(winners) != 1:
                trial = {}
                break
            trial[position] = winners[0]
            if winners[0][0] != current[position] and not _sings_like(winners[0], others):
                trial = {}
                break
        if trial:
            changes = sum(_changes(trial[p], current[p]) for p in trial)
            outcomes.append((_total_unaligned(trial), changes, trial))
    if not outcomes:
        return None
    outcomes.sort(key=lambda o: (o[0], o[1]))
    best_unaligned, best_changes, refit = outcomes[0]
    if best_unaligned >= _total_unaligned(as_read) or best_changes == 0:
        return None
    if len(outcomes) > 1 and outcomes[1][:2] == (best_unaligned, best_changes):
        return None
    return refit


def _sings_like(reading: Reading, others: list[Reading]) -> bool:
    """Whether a re-read voice now sings exactly the rhythm of one on the other staff.

    The re-reading is only ever a claim that the voices move together -- the
    tenors singing B1's rhythm on Legenda system 4, bar 9. A re-reading that
    lines a voice up into a rhythm nobody sings is inventing one: on system 9,
    bar 18 (eerovil/musescore-choir-plugins#245) it would have flattened B2's
    printed triplets against the other voices' plain quarters.
    """
    mine = list(
        zip(
            reading[1],
            [EncodedSymbol(r).get_duration().fraction for r in reading[0]],
            strict=True,
        )
    )
    for other in others:
        theirs = list(
            zip(
                other[1],
                [EncodedSymbol(r).get_duration().fraction for r in other[0]],
                strict=True,
            )
        )
        if mine == theirs:
            return True
    return False


def _changes(reading: Reading, current: tuple[str, ...]) -> int:
    return sum(1 for a, b in zip(reading[0], current, strict=True) if a != b)


def _consistent(
    readings: dict[str, list[tuple[tuple[str, ...], list[Fraction], Fraction]]],
) -> bool:
    """Whether the bar as read already has every voice ending together."""
    return len({rs[0][2] for rs in readings.values() if rs}) == 1


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


def reconstruct_voice(
    voice: list[EncodedSymbol],
    bar_length: Fraction | None = None,
    system_targets: list[Fraction | None] | None = None,
) -> ReconstructedVoice:
    """Apply homr's musical reconstruction passes in their established order."""
    changes: list[ReconstructionChange] = []
    groups = group_into_chords(voice)
    groups = voices_from_opposite_stems(groups, changes)
    # Before the triplets: a unison copy's skipped note leaves a gap that would
    # otherwise read as a triplet still open (#265, Legenda system 9, bar 19).
    groups = fill_unison_copies(groups, changes)
    groups = complete_open_triplets(groups, changes)
    groups = triplets_onto_the_beat(groups, changes)
    groups = repair_tuplet_overlaps_until_settled(groups, changes)
    groups = triplets_to_match_the_other_staff(groups, changes)
    groups = repair_tuplet_overlaps_until_settled(groups, changes)
    groups = retime_onto_steady_voices(groups, changes)
    groups = repair_tuplet_overlaps(groups, changes, restore_decoded=True)
    groups = solve_bar_rhythms(groups, changes, bar_length)
    groups = drop_double_reads(groups, changes, system_targets)
    groups = repair_bar_arithmetic(groups, changes, system_targets)
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
