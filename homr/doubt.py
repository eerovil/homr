"""Bars this reading is probably wrong about, marked so a person checks them.

eerovil/musescore-choir-plugins#245: Legenda system 11's bar 25 came out with a
triplet on the wrong two bass notes. Both readings fill the bar and the decoder
slightly prefers the wrong one, so no repair can tell -- and a person reading
the score would not notice either, so it would reach the practice track. The
owner's requirement is that no wrong bar goes unmarked; false alarms come
second.

Three rules, each catching what the others miss. Measured on 38 printed
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

Each marked bar gets a red `⚠` text at its head, on its own staff, naming what
was doubted. It is not on the page, which is why it is opt-in (`--mark-doubt`).
"""

import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from fractions import Fraction

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


def find_doubts(
    staffs: list[list[EncodedSymbol]], xml: ET.Element, second: ET.Element | None
) -> Doubts:
    doubts: Doubts = defaultdict(set)
    for found in (
        confidence_doubts(staffs, bar_lengths(xml)),
        reading_doubts(xml, second),
        odd_time_doubts(xml),
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
    SECOND_READING,
    SECOND_READING_FAILED,
]


def _sort_key(reason: str) -> tuple[int, str]:
    tail = reason.split(": ", 1)[-1]
    return (_ORDER.index(tail) if tail in _ORDER else len(_ORDER), reason)


def mark_text(reasons: set[str]) -> str:
    return MARK_PREFIX + "check against the page: " + "; ".join(sorted(reasons, key=_sort_key))


def mark_doubts(xml: ET.Element, doubts: Doubts) -> int:
    """Put a red `⚠` text at the head of each doubted bar, on its staff. Returns how many."""
    parts = xml.findall("part")
    marked = 0
    for (part_index, staff, bar), reasons in sorted(doubts.items()):
        if not reasons or part_index >= len(parts):
            continue
        part = parts[part_index]
        measures = part.findall("measure")
        if bar > len(measures):
            continue
        measure = measures[bar - 1]
        direction = ET.Element("direction", placement="above")
        words = ET.SubElement(ET.SubElement(direction, "direction-type"), "words", color="#FF0000")
        words.text = mark_text(reasons)
        if _staves_of(part) > 1:
            ET.SubElement(direction, "staff").text = str(staff)
        index = next(
            (i for i, child in enumerate(measure) if child.tag not in ("attributes", "print")),
            len(measure),
        )
        measure.insert(index, direction)
        marked += 1
    return marked
