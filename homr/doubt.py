"""Bars this reading is probably wrong about, marked so a person checks them.

eerovil/musescore-choir-plugins#245: Legenda system 11's bar 25 came out with a
triplet on the wrong two bass notes. Both readings fill the bar and the decoder
slightly prefers the wrong one, so no repair can tell -- and a person reading
the score would not notice either, so it would reach the practice track. The
owner's requirement is that no wrong bar goes unmarked; false alarms come
second.

Five rules, each catching what the others miss. The first three were measured on 38 printed
systems (Legenda, the five recognition fixtures and four songs): on the bars
with owner-checked references, all 9 wrong ones are marked, along with 39 of
the 117 right ones; on the songs whose references are less certain, 26 of 27.

- The decoder doubted it (`confidence_doubts`): two complete readings of a
  voice that fill the bar are nearly as likely; a pitch, accidental or voice
  line it was unsure of; or the two voices reading one shared notehead gave it
  different lengths.
- A second reading disagrees (`reading_doubts`): the same crop read again at
  80% of its size. A confident misreading is often not a stable one. This is
  what catches the errors the decoder was sure of.
- A note starts at a time no ordinary rhythm reaches (`odd_time_doubts`): a
  bar whose arithmetic went wrong upstream puts notes at 15/32 of a bar.
- Two notes struck together make an interval music does not use
  (`interval_doubts`): a doubly augmented or diminished one, which a misread
  accidental makes and a printed score does not.
- A voice falls silent, with no rest, where the other voice of its staff
  strikes two heads at once (`silent_beside_chord_doubts`): a note handed to
  the wrong voice.

Each marked bar gets a red `⚠` text at its head, on its own staff, naming what
was doubted. It is not on the page, which is why it is opt-in (`--mark-doubt`).
"""

import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from fractions import Fraction

from homr.arc_finder import ARC_DOUBT
from homr.slur_resolution import INFERRED
from homr.transformer.vocabulary import EncodedSymbol

#: What every mark starts with; the choir app recognises a mark by it.
MARK_PREFIX = "⚠ "

# Thresholds, from the measurement in the module docstring.
_GAP = 2.0  # log-likelihood between the best and second-best reading of a voice
_PITCH = 0.85
_LIFT = 0.85
_POSITION = 0.65
_ALTERNATIVES = 4  # how many of the decoder's ranked values a note may take
_UNRANKED = 1e-6

RHYTHM_CLOSE = "rhythm: another reading is nearly as likely"
PITCH_UNSURE = "pitch unsure"
ACCIDENTAL_UNSURE = "accidental unsure"
VOICE_UNSURE = "voice unsure"
COPIES_DISAGREE = "the two voices read a shared note differently"
SECOND_READING = "a second reading came out different"
SECOND_READING_FAILED = "the second reading failed"
ODD_TIME = "a note starts at an odd time"
ODD_INTERVAL = "an accidental makes an interval music does not use"
SILENT_BESIDE_CHORD = "a voice falls silent where the other holds two notes"
SLUR_INFERRED = "a slur end inferred, not read"
SLUR_PICTURE = "a slur on the page whose notes the picture left open"
TIE_PICTURE = "a tie on the page whose notes the picture left open"

# (part index, staff within the part from 1, bar from 1) -> reasons
Doubts = dict[tuple[int, int, int], set[str]]


def _duration(value: str) -> Fraction | None:
    match = re.fullmatch(r"(note|rest)_(\d+)(\.*)", value or "")
    if not match:
        return None
    duration = Fraction(1, int(match.group(2)) or 1)
    if match.group(3) == ".":
        duration *= Fraction(3, 2)
    elif match.group(3) == "..":
        duration *= Fraction(7, 4)
    return duration


def _staff(position: str) -> int:
    return 2 if position.startswith("lower") else 1


def _voice(position: str) -> str:
    return "voice 2" if position.endswith("2") else "voice 1"


def _confidence(symbol: EncodedSymbol, field: str) -> dict:
    return (symbol.confidence or {}).get(field) or {}


def _probability(symbol: EncodedSymbol, field: str) -> float:
    return float(_confidence(symbol, field).get("probability", 1.0))


def _options(heads: list[EncodedSymbol]) -> list[tuple[Fraction, float]]:
    """One moment of one voice: each value it could take, with its log-likelihood."""
    kind = heads[0].rhythm.split("_")[0]
    values = {heads[0].rhythm: None}
    for head in heads:
        for alternative in _confidence(head, "rhythm").get("alternatives", [])[:_ALTERNATIVES]:
            value = alternative["value"]
            if value.startswith(kind + "_") and _duration(value):
                values.setdefault(value, None)
    options = []
    for value in values:
        duration = _duration(value)
        if duration is None:
            continue
        total = 0.0
        for head in heads:
            ranked = {
                a["value"]: a["probability"]
                for a in _confidence(head, "rhythm").get("alternatives", [])
            }
            floor = min(ranked.values()) / 2 if ranked else _UNRANKED
            total += math.log(max(ranked.get(value, floor), _UNRANKED))
        options.append((duration, total))
    return options


def reading_gap(moments: list[list[EncodedSymbol]], length: Fraction) -> float | None:
    """How much likelier the best reading of a voice that fills the bar is than the next.

    None when no reading fills the bar; infinity when only one does.
    """
    states: dict[Fraction, list[float]] = {Fraction(0): [0.0]}
    for heads in moments:
        following: dict[Fraction, list[float]] = defaultdict(list)
        for elapsed, scores in states.items():
            for duration, score in _options(heads):
                if elapsed + duration > length:
                    continue
                following[elapsed + duration].extend(s + score for s in scores)
        states = {t: sorted(s, reverse=True)[:2] for t, s in following.items()}
    best = states.get(length)
    if not best:
        return None
    return best[0] - best[1] if len(best) > 1 else math.inf


def bar_lengths(xml: ET.Element) -> list[list[Fraction]]:
    """Per part, the length of each bar its time signatures give."""
    out = []
    for part in xml.findall("part"):
        current = Fraction(1)
        lengths = []
        for measure in part.findall("measure"):
            time = measure.find("attributes/time")
            if time is not None and time.findtext("beats") and time.findtext("beat-type"):
                current = Fraction(
                    int(time.findtext("beats") or 4), int(time.findtext("beat-type") or 4)
                )
            lengths.append(current)
        out.append(lengths)
    return out


def confidence_doubts(staffs: list[list[EncodedSymbol]], lengths: list[list[Fraction]]) -> Doubts:
    """The bars the decoder itself was unsure of."""
    doubts: Doubts = defaultdict(set)
    for part, symbols in enumerate(staffs):
        bar = 1
        groups: list[tuple[int, list[EncodedSymbol]]] = []
        linked = False
        for symbol in symbols:
            if symbol.rhythm == "barline":
                bar += 1
                linked = False
                continue
            if symbol.rhythm == "chord":
                linked = True
                continue
            timed = symbol.rhythm.startswith(("note", "rest")) and _duration(symbol.rhythm)
            if not timed:
                linked = False
                continue
            if linked and groups and groups[-1][0] == bar:
                groups[-1][1].append(symbol)
            else:
                groups.append((bar, [symbol]))
            linked = False
        voices: dict[tuple[int, str], list[list[EncodedSymbol]]] = defaultdict(list)
        for bar_number, group in groups:
            by_position: dict[str, list[EncodedSymbol]] = defaultdict(list)
            for symbol in group:
                by_position[symbol.position].append(symbol)
            for position, heads in by_position.items():
                voices[(bar_number, position)].append(heads)
            notes = [s for s in group if s.rhythm.startswith("note")]
            for symbol in notes:
                key = (part, _staff(symbol.position), bar_number)
                voice = _voice(symbol.position)
                if _probability(symbol, "pitch") < _PITCH:
                    doubts[key].add(f"{voice}: {PITCH_UNSURE}")
                if _probability(symbol, "lift") < _LIFT:
                    doubts[key].add(f"{voice}: {ACCIDENTAL_UNSURE}")
                if _probability(symbol, "position") < _POSITION:
                    doubts[key].add(VOICE_UNSURE)
            for i, first in enumerate(notes):
                for second in notes[i + 1 :]:
                    if (
                        first.pitch == second.pitch
                        and first.position != second.position
                        and _staff(first.position) == _staff(second.position)
                        and first.rhythm != second.rhythm
                    ):
                        doubts[(part, _staff(first.position), bar_number)].add(COPIES_DISAGREE)
        part_lengths = lengths[part] if part < len(lengths) else []
        for (bar_number, position), moments in voices.items():
            if bar_number > len(part_lengths):
                continue
            if not any(h.rhythm.startswith("note") for heads in moments for h in heads):
                continue
            gap = reading_gap(moments, part_lengths[bar_number - 1])
            if gap is not None and gap < _GAP:
                key = (part, _staff(position), bar_number)
                doubts[key].add(f"{_voice(position)}: {RHYTHM_CLOSE}")
    return doubts


def _staves_of(part: ET.Element) -> int:
    staves = part.find("measure/attributes/staves")
    return int(staves.text) if staves is not None and staves.text else 1


def _events(
    xml: ET.Element,
) -> tuple[dict[tuple[int, int], set[tuple]], dict[int, tuple[int, int]], int]:
    """Every note and rest by (score staff, bar): (onset, kind/pitch, duration).

    Score staves are numbered through the parts, so two readings that split the
    staves into parts differently still compare staff by staff. Also returns
    which (part, staff) each score staff is, and the number of bars.
    """
    events: dict[tuple[int, int], set[tuple]] = defaultdict(set)
    where: dict[int, tuple[int, int]] = {}
    base = 0
    bars = 0
    for part_index, part in enumerate(xml.findall("part")):
        staves = _staves_of(part)
        for staff in range(1, staves + 1):
            where[base + staff] = (part_index, staff)
        divisions = 1
        for bar, measure in enumerate(part.findall("measure"), start=1):
            bars = max(bars, bar)
            at = previous = Fraction(0)
            for element in measure:
                text = element.findtext("divisions")
                if element.tag == "attributes" and text:
                    divisions = int(text)
                if element.tag in ("backup", "forward"):
                    step = Fraction(int(element.findtext("duration") or 0), divisions * 4)
                    at += -step if element.tag == "backup" else step
                    continue
                if element.tag != "note":
                    continue
                duration = Fraction(int(element.findtext("duration") or 0), divisions * 4)
                onset = previous if element.find("chord") is not None else at
                if element.find("chord") is None:
                    previous, at = at, at + duration
                if element.find("grace") is not None or duration == 0:
                    continue
                staff = base + int(element.findtext("staff") or 1)
                if element.find("rest") is not None:
                    what: tuple = ("rest",)
                else:
                    pitch = element.find("pitch")
                    what = (
                        pitch.findtext("step") if pitch is not None else "",
                        pitch.findtext("octave") if pitch is not None else "",
                        int(float(pitch.findtext("alter") or 0)) if pitch is not None else 0,
                    )
                events[(staff, bar)].add((onset, what, duration))
        base += staves
    return events, where, bars


def reading_doubts(xml: ET.Element, second: ET.Element | None) -> Doubts:
    """The bars a second reading of the same crop read differently.

    Rests are left out of the comparison: where a rest sits between voices is
    not what a singer hears. With no second reading at all, every bar is in
    doubt -- an unread system is not a checked one.
    """
    events, where, bars = _events(xml)
    doubts: Doubts = defaultdict(set)
    if second is None:
        for part, inner in where.values():
            for bar in range(1, bars + 1):
                doubts[(part, inner, bar)].add(SECOND_READING_FAILED)
        return doubts
    other, other_where, other_bars = _events(second)
    same_shape = bars == other_bars and len(where) == len(other_where)
    for staff, (part, inner) in where.items():
        for bar in range(1, bars + 1):
            notes = {e for e in events.get((staff, bar), set()) if e[1] != ("rest",)}
            theirs = {e for e in other.get((staff, bar), set()) if e[1] != ("rest",)}
            if not same_shape or notes != theirs:
                doubts[(part, inner, bar)].add(SECOND_READING)
    return doubts


def _odd(onset: Fraction) -> bool:
    return (onset * 16).denominator != 1 and (onset * 24).denominator != 1


def odd_time_doubts(xml: ET.Element) -> Doubts:
    """The bars where a note or rest starts off every sixteenth and every triplet sixteenth."""
    events, where, _ = _events(xml)
    doubts: Doubts = defaultdict(set)
    for (staff, bar), found in events.items():
        if any(_odd(onset) for onset, _, _ in found):
            part, inner = where[staff]
            doubts[(part, inner, bar)].add(ODD_TIME)
    return doubts


_STEPS = "CDEFGAB"
_SEMITONES = (0, 2, 4, 5, 7, 9, 11)  # also the major/perfect size of each generic interval
_PERFECT = {0, 3, 4}  # unison, fourth and fifth, by steps modulo the octave


def _out_of_use(low: tuple, high: tuple) -> bool:
    """Whether two notes sounding together are a doubly augmented or diminished
    interval: B natural against F flat, say, where Bbb-Fb is a fifth."""
    steps = [_STEPS.index(note[0]) + 7 * int(note[1]) for note in (low, high)]
    pitch = [
        _SEMITONES[_STEPS.index(note[0])] + 12 * int(note[1]) + note[2] for note in (low, high)
    ]
    if steps[0] > steps[1]:
        steps.reverse()
        pitch.reverse()
    generic = steps[1] - steps[0]
    size = 12 * (generic // 7) + _SEMITONES[generic % 7]
    off = pitch[1] - pitch[0] - size
    allowed = (-1, 0, 1) if generic % 7 in _PERFECT else (-2, -1, 0, 1)
    return off not in allowed


def interval_doubts(xml: ET.Element) -> Doubts:
    """The bars where two notes struck together on one staff are an interval music
    does not use, which a misread accidental makes.

    Finlandia system 8 (eerovil/musescore-choir-plugins#274): the bass prints
    B double-flat under F flat, a fifth, and the decoder read the double flat as a
    natural -- B natural against F flat -- sure of it and the same on the second
    reading, so nothing else marked the bar. Across the 71 hand-checked systems
    of that card, the five recognition fixtures and every chord of the 48 songs'
    scores on the owner's host (1324 of them), no such interval is printed; in
    every homr reading of those songs this was the only one.
    """
    events, where, _ = _events(xml)
    doubts: Doubts = defaultdict(set)
    for (staff, bar), found in events.items():
        struck: dict[Fraction, list[tuple]] = defaultdict(list)
        for onset, what, _ in found:
            if what != ("rest",) and what[0]:
                struck[onset].append(what)
        if any(
            _out_of_use(notes[i], notes[j])
            for notes in struck.values()
            for i in range(len(notes))
            for j in range(i + 1, len(notes))
        ):
            part, inner = where[staff]
            doubts[(part, inner, bar)].add(ODD_INTERVAL)
    return doubts


def silent_beside_chord_doubts(xml: ET.Element) -> Doubts:
    """The bars where a voice falls silent -- no note, no rest -- at a moment the
    other voice of its staff strikes two heads at once.

    That is a note given to the wrong voice: on Finlandia system 10
    (eerovil/musescore-choir-plugins#274) the first tenor's A flat was written
    into the second tenor's voice as a chord on her G, and the first tenor had a
    hole on that beat. Printed music fills a voice's silence with a rest, so no
    hand-checked reference of that card or the recognition fixtures shows it. A
    `<forward>` is MusicXML's written silence for a voice and counts as covering
    it: only a hole nothing was written into is evidence.
    """
    doubts: Doubts = defaultdict(set)
    for part_index, part in enumerate(xml.findall("part")):
        divisions = 1
        for bar, measure in enumerate(part.findall("measure"), start=1):
            at = previous = Fraction(0)
            spans: dict[tuple[str, str], list[tuple[Fraction, Fraction]]] = defaultdict(list)
            heads: dict[tuple[str, str, Fraction], int] = defaultdict(int)
            last_staff, last_voice = "1", None
            for element in measure:
                text = element.findtext("divisions")
                if element.tag == "attributes" and text:
                    divisions = int(text)
                if element.tag in ("backup", "forward"):
                    step = Fraction(int(element.findtext("duration") or 0), divisions * 4)
                    if element.tag == "forward":
                        # A forward is a voice's own written silence, not a hole:
                        # it covers the voice it names, else the one written last.
                        voice = element.findtext("voice") or last_voice
                        if voice is not None:
                            spans[(element.findtext("staff") or last_staff, voice)].append(
                                (at, at + step)
                            )
                    at += -step if element.tag == "backup" else step
                    continue
                if element.tag != "note" or element.find("grace") is not None:
                    continue
                duration = Fraction(int(element.findtext("duration") or 0), divisions * 4)
                chord = element.find("chord") is not None
                onset = previous if chord else at
                if not chord:
                    previous, at = at, at + duration
                key = (element.findtext("staff") or "1", element.findtext("voice") or "1")
                last_staff, last_voice = key
                if not chord:
                    spans[key].append((onset, onset + duration))
                if element.find("pitch") is not None:
                    heads[(key[0], key[1], onset)] += 1
            for (staff, voice), covered in spans.items():
                start = min(a for a, _ in covered)
                end = max(b for _, b in covered)
                for (other_staff, other, onset), count in heads.items():
                    if (
                        other_staff == staff
                        and other != voice
                        and count >= 2  # noqa: PLR2004
                        and start < onset < end
                        and not any(a <= onset < b for a, b in covered)
                    ):
                        doubts[(part_index, int(staff), bar)].add(SILENT_BESIDE_CHORD)
    return doubts


def _inferred_slur_notes(xml: ET.Element) -> dict[tuple[int, int, int], list[ET.Element]]:
    """Notes carrying a slur stop inferred rather than read, by (part, staff, bar)."""
    found: dict[tuple[int, int, int], list[ET.Element]] = defaultdict(list)
    for part_index, part in enumerate(xml.findall("part")):
        for bar, measure in enumerate(part.findall("measure"), 1):
            for note in measure.findall("note"):
                if any(slur.get(INFERRED) for slur in note.iter("slur")):
                    found[(part_index, int(note.findtext("staff", "1")), bar)].append(note)
    return found


def _picture_arc_notes(xml: ET.Element) -> dict[tuple[int, int, int], dict[str, list[ET.Element]]]:
    """Notes the arc finder was unsure about, by (part, staff, bar) and reason."""
    found: dict[tuple[int, int, int], dict[str, list[ET.Element]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for part_index, part in enumerate(xml.findall("part")):
        for bar, measure in enumerate(part.findall("measure"), 1):
            for note in measure.findall("note"):
                key = (part_index, int(note.findtext("staff", "1")), bar)
                kinds = (note.get(ARC_DOUBT) or "").split(",")
                if "slur" in kinds:
                    found[key][SLUR_PICTURE].append(note)
                if "tie" in kinds:
                    found[key][TIE_PICTURE].append(note)
    return found


def picture_arc_doubts(xml: ET.Element) -> Doubts:
    """An arc the page shows but the picture could not place for certain is
    written as its best guess and marked (eerovil/musescore-choir-plugins#328)."""
    return {key: set(reasons) for key, reasons in _picture_arc_notes(xml).items()}


def slur_doubts(xml: ET.Element) -> Doubts:
    """A slur whose stop was inferred (`slur_resolution._close_on_tie`) is written
    and marked `slur?`: the owner chose a missed slur as the worse error, and an
    inferred one is never silent (eerovil/musescore-choir-plugins#318)."""
    return {key: {SLUR_INFERRED} for key in _inferred_slur_notes(xml)}


def find_doubts(
    staffs: list[list[EncodedSymbol]], xml: ET.Element, second: ET.Element | None
) -> Doubts:
    doubts: Doubts = defaultdict(set)
    for found in (
        confidence_doubts(staffs, bar_lengths(xml)),
        reading_doubts(xml, second),
        odd_time_doubts(xml),
        interval_doubts(xml),
        silent_beside_chord_doubts(xml),
        slur_doubts(xml),
        picture_arc_doubts(xml),
    ):
        for key, reasons in found.items():
            doubts[key] |= reasons
    return doubts


_ORDER = [
    RHYTHM_CLOSE,
    PITCH_UNSURE,
    ACCIDENTAL_UNSURE,
    VOICE_UNSURE,
    COPIES_DISAGREE,
    ODD_TIME,
    ODD_INTERVAL,
    SILENT_BESIDE_CHORD,
    SLUR_INFERRED,
    SLUR_PICTURE,
    TIE_PICTURE,
    SECOND_READING,
    SECOND_READING_FAILED,
]


def _sort_key(reason: str) -> tuple[int, str]:
    tail = reason.split(": ", 1)[-1]
    return (_ORDER.index(tail) if tail in _ORDER else len(_ORDER), reason)


#: What a person reads at the bar: one word per kind of doubt. The reasons above
#: stay as they are for the record (`find_doubts`); the score carries only this.
_WORD = {
    RHYTHM_CLOSE: "rhythm?",
    PITCH_UNSURE: "pitch?",
    ACCIDENTAL_UNSURE: "accidental?",
    VOICE_UNSURE: "voice?",
    COPIES_DISAGREE: "rhythm?",
    ODD_TIME: "rhythm?",
    ODD_INTERVAL: "accidental?",
    SILENT_BESIDE_CHORD: "voice?",
    SLUR_INFERRED: "slur?",
    SLUR_PICTURE: "slur?",
    TIE_PICTURE: "tie?",
    SECOND_READING: "notes?",
    SECOND_READING_FAILED: "unchecked?",
}
_WORD_ORDER = [
    "pitch?",
    "accidental?",
    "rhythm?",
    "voice?",
    "slur?",
    "tie?",
    "notes?",
    "unchecked?",
]


def mark_words(reasons: set[str]) -> list[str]:
    """The words a mark says, each once, in a fixed order."""
    words = {_WORD.get(reason.split(": ", 1)[-1], "check?") for reason in reasons}
    return sorted(
        words, key=lambda w: _WORD_ORDER.index(w) if w in _WORD_ORDER else len(_WORD_ORDER)
    )


def mark_text(reasons: set[str], words: set[str] | None = None) -> str:
    chosen = (
        mark_words(reasons)
        if words is None
        else sorted(
            words, key=lambda w: _WORD_ORDER.index(w) if w in _WORD_ORDER else len(_WORD_ORDER)
        )
    )
    return MARK_PREFIX + " ".join(chosen)


def mark_doubts(
    xml: ET.Element,
    doubts: Doubts,
    spots: "Spots | None" = None,
    words: "dict[tuple[int, int, int], set[str]] | None" = None,
) -> int:
    """Mark each doubted bar on its staff. Returns how many.

    Where the doubt names notes (`find_spots`), they are coloured red and the
    `⚠` word stands right above the first of them; otherwise the word stands at
    the head of the bar. Colouring is not undone by anything here: a person
    turns the notes back once the bar is put right.
    """
    parts = xml.findall("part")
    marked = 0
    for key, reasons in sorted(doubts.items()):
        part_index, staff, bar = key
        if not reasons or part_index >= len(parts):
            continue
        part = parts[part_index]
        measures = part.findall("measure")
        if bar > len(measures):
            continue
        measure = measures[bar - 1]
        direction = ET.Element("direction", placement="above")
        text = ET.SubElement(ET.SubElement(direction, "direction-type"), "words", color="#FF0000")
        text.text = mark_text(reasons, (words or {}).get(key))
        if _staves_of(part) > 1:
            ET.SubElement(direction, "staff").text = str(staff)
        children = list(measure)
        flagged = [note for note in (spots or {}).get(key, []) if note in children]
        for note in flagged:
            note.set("color", "#FF0000")
            head = note.find("notehead")
            if head is None:
                head = ET.Element("notehead")
                head.text = "normal"
                _insert_notehead(note, head)
            head.set("color", "#FF0000")
        if flagged:
            first = min(children.index(note) for note in flagged)
            # A chord's later notes carry <chord/>: the word goes before the
            # note that opens the chord, where the cursor stands at its onset.
            while first > 0 and children[first].find("chord") is not None:
                first -= 1
            index = first
        else:
            index = next(
                (i for i, child in enumerate(children) if child.tag not in ("attributes", "print")),
                len(children),
            )
        measure.insert(index, direction)
        marked += 1
    return marked


#: Where <notehead> goes among a note's children (MusicXML's own order).
_BEFORE_NOTEHEAD = {
    "grace",
    "cue",
    "chord",
    "pitch",
    "unpitched",
    "rest",
    "tie",
    "duration",
    "instrument",
    "footnote",
    "level",
    "voice",
    "type",
    "dot",
    "accidental",
    "time-modification",
    "stem",
}


def _insert_notehead(note: ET.Element, head: ET.Element) -> None:
    index = 0
    for i, child in enumerate(note):
        if isinstance(child.tag, str) and child.tag in _BEFORE_NOTEHEAD:
            index = i + 1
    note.insert(index, head)


# --- where in the bar --------------------------------------------------------

#: (part, staff, bar) -> the notes a doubt is about.
Spots = dict[tuple[int, int, int], list[ET.Element]]

_IMGPOS = re.compile(r"imgpos: (-?\d+), (-?\d+)")


def _notes_by_position(xml: ET.Element) -> dict[tuple[int, int], list[ET.Element]]:
    """The written notes, by the image position their symbol was decoded at."""
    found: dict[tuple[int, int], list[ET.Element]] = defaultdict(list)
    for note in xml.iter("note"):
        for child in note:
            if child.tag is ET.Comment:  # type: ignore[comparison-overlap]
                match = _IMGPOS.search(child.text or "")
                if match:
                    found[(int(match.group(1)), int(match.group(2)))].append(note)
    return found


def _written(
    symbol: EncodedSymbol, by_position: dict[tuple[int, int], list[ET.Element]]
) -> list[ET.Element]:
    coordinates = symbol.image_coordinates
    if coordinates is None:
        return []
    x, y = coordinates
    return by_position.get((round(x), round(y)), [])


def _located(xml: ET.Element) -> dict[tuple[int, int, int], list[tuple]]:
    """Every pitched note by (part, staff, bar): (onset, (step, octave, alter), duration, note)."""
    found: dict[tuple[int, int, int], list[tuple]] = defaultdict(list)
    for part_index, part in enumerate(xml.findall("part")):
        divisions = 1
        for bar, measure in enumerate(part.findall("measure"), start=1):
            at = previous = Fraction(0)
            for element in measure:
                text = element.findtext("divisions")
                if element.tag == "attributes" and text:
                    divisions = int(text)
                if element.tag in ("backup", "forward"):
                    step = Fraction(int(element.findtext("duration") or 0), divisions * 4)
                    at += -step if element.tag == "backup" else step
                    continue
                if element.tag != "note":
                    continue
                duration = Fraction(int(element.findtext("duration") or 0), divisions * 4)
                onset = previous if element.find("chord") is not None else at
                if element.find("chord") is None:
                    previous, at = at, at + duration
                pitch = element.find("pitch")
                if pitch is None or element.find("grace") is not None:
                    continue
                what = (
                    pitch.findtext("step") or "",
                    pitch.findtext("octave") or "",
                    int(float(pitch.findtext("alter") or 0)),
                )
                staff = int(element.findtext("staff") or 1)
                found[(part_index, staff, bar)].append((onset, what, duration, element))
    return found


def _two_best(moments: list[list[EncodedSymbol]], length: Fraction) -> tuple[list, list] | None:
    """The best and second-best values of each moment, of the readings that fill the bar."""
    states: dict[Fraction, list[tuple[float, tuple]]] = {Fraction(0): [(0.0, ())]}
    for heads in moments:
        following: dict[Fraction, list[tuple[float, tuple]]] = defaultdict(list)
        for elapsed, paths in states.items():
            for duration, score in _options(heads):
                if elapsed + duration > length:
                    continue
                following[elapsed + duration].extend(
                    (total + score, values + (duration,)) for total, values in paths
                )
        states = {t: sorted(p, key=lambda item: -item[0])[:2] for t, p in following.items()}
    best = states.get(length)
    if not best or len(best) < 2:  # noqa: PLR2004
        return None
    return list(best[0][1]), list(best[1][1])


def find_spots(
    staffs: list[list[EncodedSymbol]],
    xml: ET.Element,
    second: ET.Element | None,
    doubts: Doubts,
) -> tuple[Spots, dict[tuple[int, int, int], set[str]]]:
    """For each doubted bar, the notes the doubt is about, and the words to say.

    The notes are found again by the same evidence each rule used: a decoded
    symbol by the image position it was written with, a note of the score by
    where it stands in the bar. A second reading that differs only in pitch
    (the same moments, the same lengths) says "pitch?" rather than "notes?".
    """
    spots: Spots = defaultdict(list)
    words: dict[tuple[int, int, int], set[str]] = {}
    by_position = _notes_by_position(xml)
    located = _located(xml)

    def add(key: tuple[int, int, int], notes: list[ET.Element]) -> None:
        if key in doubts:
            for note in notes:
                if note.find("pitch") is not None and note not in spots[key]:
                    spots[key].append(note)

    lengths = bar_lengths(xml)
    for part, symbols in enumerate(staffs):
        bar = 1
        moments: dict[tuple[int, str], list[list[EncodedSymbol]]] = defaultdict(list)
        linked = False
        for symbol in symbols:
            if symbol.rhythm == "barline":
                bar += 1
                linked = False
                continue
            if symbol.rhythm == "chord":
                linked = True
                continue
            if not (symbol.rhythm.startswith(("note", "rest")) and _duration(symbol.rhythm)):
                linked = False
                continue
            key = (part, _staff(symbol.position), bar)
            if symbol.rhythm.startswith("note") and (
                _probability(symbol, "pitch") < _PITCH
                or _probability(symbol, "lift") < _LIFT
                or _probability(symbol, "position") < _POSITION
            ):
                add(key, _written(symbol, by_position))
            group = moments[(bar, symbol.position)]
            if linked and group:
                group[-1].append(symbol)
            else:
                group.append([symbol])
            linked = False
        part_lengths = lengths[part] if part < len(lengths) else []
        for (bar_number, position), voice in moments.items():
            key = (part, _staff(position), bar_number)
            if key not in doubts or bar_number > len(part_lengths):
                continue
            pair = _two_best(voice, part_lengths[bar_number - 1])
            if pair is None:
                continue
            best, other = pair
            for heads, one, two in zip(voice, best, other, strict=False):
                if one != two:
                    for head in heads:
                        add(key, _written(head, by_position))

    for key, notes in located.items():
        if key not in doubts:
            continue
        reasons = {reason.split(": ", 1)[-1] for reason in doubts[key]}
        struck: dict[Fraction, list[tuple]] = defaultdict(list)
        for entry in notes:
            struck[entry[0]].append(entry)
        if ODD_TIME in reasons:
            add(key, [entry[3] for entry in notes if _odd(entry[0])])
        if ODD_INTERVAL in reasons:
            for together in struck.values():
                for i, low in enumerate(together):
                    for high in together[i + 1 :]:
                        if _out_of_use(low[1], high[1]):
                            add(key, [low[3], high[3]])
        if SILENT_BESIDE_CHORD in reasons:
            for together in struck.values():
                voices: dict[str, list[ET.Element]] = defaultdict(list)
                for entry in together:
                    voices[entry[3].findtext("voice") or "1"].append(entry[3])
                for chord_notes in voices.values():
                    if len(chord_notes) >= 2:  # noqa: PLR2004
                        add(key, chord_notes)
        if COPIES_DISAGREE in reasons:
            for together in struck.values():
                for i, one in enumerate(together):
                    for two in together[i + 1 :]:
                        if one[1] == two[1] and one[2] != two[2]:
                            add(key, [one[3], two[3]])

    for key, inferred in _inferred_slur_notes(xml).items():
        add(key, inferred)
    for key, by_reason in _picture_arc_notes(xml).items():
        for unsure in by_reason.values():
            add(key, unsure)

    if second is not None:
        theirs = _located(second)
        for key, notes in located.items():
            if key not in doubts or not any(
                reason.endswith(SECOND_READING) for reason in doubts[key]
            ):
                continue
            read_again = {(entry[0], entry[1], entry[2]) for entry in theirs.get(key, [])}
            differing = [
                entry for entry in notes if (entry[0], entry[1], entry[2]) not in read_again
            ]
            add(key, [entry[3] for entry in differing])
            moments_there = {(entry[0], entry[2]) for entry in theirs.get(key, [])}
            if differing and all((entry[0], entry[2]) in moments_there for entry in differing):
                words[key] = set(mark_words(doubts[key])) - {"notes?"} | {"pitch?"}
    return spots, words
