"""Read a fused grand staff again, one staff at a time, when the decoder doubts itself.

homr fuses two braced staffs into a single image before inference
(`Staff.merge`), so one transformer pass has to read both at once. On a crowded
bar that pass can collapse. On the `sammon-ryosto` fixture it wrote one notehead
per staff where the page prints six -- at rhythm probabilities of 0.31, 0.41 and
0.44, against a median of 0.89 everywhere else on the same page. Reading each
staff on its own recovered every printed head, at ordinary confidence.

That was written up once as a limit of the model: the notes "were never
tokenised, so nothing downstream dropped them". True of the fused pass, and only
of the fused pass -- the ink is legible, and the same weights read it.

So when a fused reading holds a note or rest the decoder was unsure of, the pair
is read again as two staffs, spliced back into the one stream a fused read would
have produced, and **whichever reading the decoder was surer of is kept**. The
threshold therefore decides only when to spend the two extra passes; it never
decides what the answer is. A false trigger costs time, not correctness -- which
matters, because the trigger is the part most likely to be wrong: it was chosen
against three fused pages, and `kolme-kakea`'s fused pair holds a barline at 0.46
while being note-perfect.

Splicing rather than un-fusing, because the output has to keep looking like the
page: a grand staff is one part on two staffs, and handing back two parts would
lose the brace to fix the notes.
"""

from __future__ import annotations

from fractions import Fraction

from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol, sort_token_chords

#: A note or rest read less surely than this makes a fused pair worth reading
#: again. Measured, not chosen: the faults on `sammon-ryosto` sit at 0.31-0.44,
#: `hanget-soi`'s least sure note at 0.53, and an ordinary symbol at ~0.89.
DOUBT_THRESHOLD = 0.5

_HEADINGS = ("clef", "keySignature", "timeSignature")


def _is_timed(symbol: EncodedSymbol) -> bool:
    return symbol.rhythm.startswith(("note", "rest"))


def _is_barline(symbol: EncodedSymbol) -> bool:
    return "barline" in symbol.rhythm or symbol.rhythm.startswith("repeat")


def _rhythm_probability(symbol: EncodedSymbol) -> float | None:
    confidence = symbol.confidence
    if not confidence:
        return None
    rhythm = confidence.get("rhythm")
    if not isinstance(rhythm, dict):
        return None
    value = rhythm.get("probability")
    return float(value) if isinstance(value, int | float) else None


def doubtful(symbols: list[EncodedSymbol], threshold: float = DOUBT_THRESHOLD) -> bool:
    """Did the decoder read a note or rest here less surely than it usually does?

    Notes and rests only. A barline or a clef read unsurely is not the symptom
    this exists for -- it is a lost notehead -- and counting one would fire on a
    fixture that is already right.
    """
    for symbol in symbols:
        if not _is_timed(symbol):
            continue
        probability = _rhythm_probability(symbol)
        if probability is not None and probability < threshold:
            return True
    return False


def surety(symbols: list[EncodedSymbol]) -> float | None:
    """How sure the decoder was of the notes and rests in a reading, on average.

    None when nothing here carries a confidence, which is not the same as zero:
    a reading we cannot judge must not win a comparison, and must not lose one
    either.
    """
    probabilities = [
        probability
        for symbol in symbols
        if _is_timed(symbol)
        for probability in [_rhythm_probability(symbol)]
        if probability is not None
    ]
    if not probabilities:
        return None
    return sum(probabilities) / len(probabilities)


def _as_lower(symbol: EncodedSymbol) -> EncodedSymbol:
    """The same symbol, read as the lower staff of a pair.

    A staff read on its own comes back entirely `upper`, since there is no other
    staff for it to be the top of. Symbols with no position at all -- a key or a
    time signature -- keep none.
    """
    if symbol.position != "upper":
        return symbol
    copy = EncodedSymbol(
        rhythm=symbol.rhythm,
        pitch=symbol.pitch,
        lift=symbol.lift,
        articulation=symbol.articulation,
        slur=symbol.slur,
        position="lower",
        coordinates=symbol.coordinates,
        confidence=symbol.confidence,
        stem_direction=symbol.stem_direction,
    )
    return copy


def _split_bars(
    symbols: list[EncodedSymbol],
) -> tuple[list[list[EncodedSymbol]], list[EncodedSymbol | None]]:
    """The bars of one staff's reading, and the barline that closed each."""
    bars: list[list[EncodedSymbol]] = []
    closers: list[EncodedSymbol | None] = []
    current: list[EncodedSymbol] = []
    for symbol in symbols:
        if _is_barline(symbol):
            bars.append(current)
            closers.append(symbol)
            current = []
        else:
            current.append(symbol)
    if current:
        bars.append(current)
        closers.append(None)
    return bars, closers


def _moments(bar: list[EncodedSymbol]) -> list[list[EncodedSymbol]]:
    return sort_token_chords(bar)


def _onsets(
    bar: list[EncodedSymbol],
) -> tuple[list[EncodedSymbol], list[tuple[Fraction, list[EncodedSymbol]]]]:
    """A bar as its headings plus its sounding moments, each at its own onset.

    A moment lasts as long as its shortest note, which is how the decoder's own
    stream is read back downstream (`SymbolChord.get_duration`), so measuring it
    the same way is what lets two staffs be lined up against each other.
    """
    headings: list[EncodedSymbol] = []
    timed: list[tuple[Fraction, list[EncodedSymbol]]] = []
    at = Fraction(0)
    for moment in _moments(bar):
        durations = [symbol.get_duration().fraction for symbol in moment if _is_timed(symbol)]
        if not durations:
            headings.extend(moment)
            continue
        timed.append((at, moment))
        at += min(durations)
    return headings, timed


def _splice_headings(
    upper: list[EncodedSymbol], lower: list[EncodedSymbol]
) -> list[list[EncodedSymbol]]:
    """The clefs as one moment, then each other heading once.

    A key or a time signature is a property of the printed system rather than of
    one staff of it, so where both staffs read one, the upper staff's is kept
    and the lower staff's dropped -- writing both would declare the key twice.
    """
    out: list[list[EncodedSymbol]] = []
    clefs = [symbol for symbol in upper if symbol.rhythm.startswith("clef")]
    clefs += [_as_lower(symbol) for symbol in lower if symbol.rhythm.startswith("clef")]
    if clefs:
        out.append(clefs)
    seen: set[str] = set()
    for symbol in [*upper, *(_as_lower(symbol) for symbol in lower)]:
        if symbol.rhythm.startswith("clef"):
            continue
        kind = next((name for name in _HEADINGS if symbol.rhythm.startswith(name)), symbol.rhythm)
        if kind in seen:
            continue
        seen.add(kind)
        out.append([symbol])
    return out


def _splice_bar(upper: list[EncodedSymbol], lower: list[EncodedSymbol]) -> list[EncodedSymbol]:
    upper_headings, upper_timed = _onsets(upper)
    lower_headings, lower_timed = _onsets(lower)

    moments: list[list[EncodedSymbol]] = _splice_headings(upper_headings, lower_headings)

    at_onset: dict[Fraction, list[EncodedSymbol]] = {}
    for onset, moment in upper_timed:
        at_onset.setdefault(onset, []).extend(moment)
    for onset, moment in lower_timed:
        at_onset.setdefault(onset, []).extend(_as_lower(symbol) for symbol in moment)
    moments.extend(at_onset[onset] for onset in sorted(at_onset))

    out: list[EncodedSymbol] = []
    for moment in moments:
        for index, symbol in enumerate(moment):
            if index:
                out.append(EncodedSymbol("chord"))
            out.append(symbol)
    return out


def splice(
    upper: list[EncodedSymbol], lower: list[EncodedSymbol]
) -> list[EncodedSymbol] | None:
    """Two separately read staffs as the one stream a fused read would have given.

    None when the two cannot be lined up -- an empty reading, or a disagreement
    about how many bars the system holds. There is no way to guess which staff
    lost or invented the barline, and a stream spliced across a misalignment
    would be worse than the fused reading it replaced.
    """
    if not upper or not lower:
        return None
    upper_bars, upper_closers = _split_bars(upper)
    lower_bars, lower_closers = _split_bars(lower)
    if len(upper_bars) != len(lower_bars):
        eprint(
            "Not splicing the re-read staffs:",
            len(upper_bars),
            "bars against",
            len(lower_bars),
        )
        return None

    out: list[EncodedSymbol] = []
    for bar_no, (upper_bar, lower_bar) in enumerate(zip(upper_bars, lower_bars, strict=True)):
        out.extend(_splice_bar(upper_bar, lower_bar))
        closer = upper_closers[bar_no] or lower_closers[bar_no]
        if closer is not None:
            out.append(closer)
    return out


def better_of(
    fused: list[EncodedSymbol], spliced: list[EncodedSymbol] | None
) -> tuple[list[EncodedSymbol], bool]:
    """The reading the decoder was surer of, and whether that replaced the fused one.

    Ties keep the fused reading: it is what homr does without this pass, and a
    re-read has to earn the replacement rather than merely match it.
    """
    if spliced is None:
        return fused, False
    was, now = surety(fused), surety(spliced)
    if was is None or now is None or now <= was:
        eprint(f"Keeping the fused reading (sureness {was} against {now})")
        return fused, False
    eprint(f"Keeping the re-read staffs (sureness {now} against {was})")
    return spliced, True
