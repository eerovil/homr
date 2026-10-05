"""The readings a person can choose between, for each bar homr marked as doubtful.

eerovil/musescore-choir-plugins#269: Legenda system 11's bar 25 has several
readings of the bass that fill the bar, and the decoder slightly prefers the
wrong one. No rule can tell them apart, but a person with the page crop beside
the options can, in seconds. So for each marked bar (`homr/doubt.py`) this
writes, per voice, the few likeliest **complete** readings -- each one fills the
bar -- ranked by the decoder's own likelihood, and the choir app offers them.

Only note lengths vary between readings: pitches, rests and the order of the
notes are as written. A reading is a list of rhythm values, one per moment of
the voice (a chord is one moment), in the vocabulary's own spelling
(`note_4`, `note_12`, `note_4.`).

The readings are worked out against the **written** MusicXML, not the decoder's
raw tokens, because reconstruction re-reads note values after decoding: the
moments are the voice as written, and each moment's alternatives are those the
decoder offered for the noteheads it matches by pitch. A moment no notehead
matches keeps the value it was written with.

They travel inside the MusicXML (`identification/miscellaneous`, field
`homr-bar-readings`), so a crop saved to disk carries its own options.
"""

import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from fractions import Fraction

from homr.doubt import (
    _ALTERNATIVES,
    _UNRANKED,
    Doubts,
    _confidence,
    _duration,
    _staff,
    _staves_of,
    bar_lengths,
)
from homr.transformer.vocabulary import EncodedSymbol

FIELD = "homr-bar-readings"
VERSION = 1

#: How many readings of a voice are offered at most.
TOP = 3

_ALTER = {"#": 1, "b": -1, "N": 0}


def value_of(duration: Fraction, kind: str = "note") -> str | None:
    """The vocabulary's spelling of a length: plain, dotted or double-dotted."""
    for dots, factor in (("", Fraction(1)), (".", Fraction(3, 2)), ("..", Fraction(7, 4))):
        base = duration / factor
        if base > 0 and base.numerator == 1:
            return f"{kind}_{base.denominator}{dots}"
    return None


def _pitch_of(note: ET.Element) -> dict | None:
    pitch = note.find("pitch")
    if pitch is None:
        return None
    return {
        "step": pitch.findtext("step") or "",
        "alter": int(float(pitch.findtext("alter") or 0)),
        "octave": int(pitch.findtext("octave") or 0),
    }


def _token_pitch(symbol: EncodedSymbol) -> tuple[str, int] | None:
    pitch = symbol.pitch or ""
    if len(pitch) < 2 or not pitch[-1].isdigit():
        return None
    return pitch[0], int(pitch[-1])


class _Moment:
    def __init__(self, onset: Fraction, kind: str, duration: Fraction) -> None:
        self.onset = onset
        self.kind = kind
        self.duration = duration
        self.pitches: list[dict] = []


def _voices(measure: ET.Element, staff: int, divisions: int) -> tuple[dict[str, list], int]:
    """The moments of each voice on one staff of a bar, following the cursor."""
    voices: dict[str, list[_Moment]] = defaultdict(list)
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
        chord = element.find("chord") is not None
        onset = previous if chord else at
        if not chord:
            previous, at = at, at + duration
        if element.find("grace") is not None or duration == 0:
            continue
        if int(element.findtext("staff") or 1) != staff:
            continue
        voice = element.findtext("voice") or "1"
        moments = voices[voice]
        pitch = _pitch_of(element)
        if chord and moments and moments[-1].onset == onset:
            if pitch is not None:
                moments[-1].pitches.append(pitch)
            continue
        moment = _Moment(onset, "rest" if element.find("rest") is not None else "note", duration)
        if pitch is not None:
            moment.pitches.append(pitch)
        moments.append(moment)
    return voices, divisions


def _heads_in_bar(symbols: list[EncodedSymbol], staff: int, bar: int) -> list[EncodedSymbol]:
    heads = []
    current = 1
    for symbol in symbols:
        if symbol.rhythm == "barline":
            current += 1
            continue
        if current != bar or not symbol.rhythm.startswith(("note", "rest")):
            continue
        if _duration(symbol.rhythm) and _staff(symbol.position) == staff:
            heads.append(symbol)
    return heads


def _heads_for(moment: _Moment, occurrence: int, heads: list[EncodedSymbol]) -> list[EncodedSymbol]:
    """The decoded noteheads a written moment came from.

    A voice's k-th moment at a pitch is taken to be the k-th notehead at that
    pitch in **each** decoded voice line of the staff, because a notehead two
    voices share was decoded once per voice, and either copy may be the one
    that offered the right length. A rest is matched the same way among rests.
    """
    targets = (
        [(p["step"], p["octave"]) for p in moment.pitches] if moment.kind == "note" else [None]
    )
    by_position: dict[str, list[EncodedSymbol]] = defaultdict(list)
    for head in heads:
        if head.rhythm.startswith(moment.kind + "_"):
            by_position[head.position].append(head)
    found = []
    for target in targets:
        for line in by_position.values():
            same = [h for h in line if target is None or _token_pitch(h) == target]
            if occurrence < len(same):
                found.append(same[occurrence])
    return found


def _moment_options(moment: _Moment, heads: list[EncodedSymbol]) -> list[tuple[str, float]]:
    """Each length a moment could take, scored by the noteheads that liked it best."""
    written = value_of(moment.duration, moment.kind)
    if written is None:
        return []
    values = [written]
    for head in heads:
        for alternative in _confidence(head, "rhythm").get("alternatives", [])[:_ALTERNATIVES]:
            value = alternative["value"]
            if value.startswith(moment.kind + "_") and _duration(value) and value not in values:
                values.append(value)
    options = []
    for value in values:
        best = None
        for head in heads:
            ranked = {
                a["value"]: a["probability"]
                for a in _confidence(head, "rhythm").get("alternatives", [])
            }
            floor = min(ranked.values()) / 2 if ranked else _UNRANKED
            score = math.log(max(ranked.get(value, floor), _UNRANKED))
            best = score if best is None else max(best, score)
        options.append((value, best if best is not None else 0.0))
    return options


def _triplet(value: str) -> bool:
    duration = _duration(value)
    return duration is not None and duration.denominator % 3 == 0


def _on(onset: Fraction, grid: Fraction) -> bool:
    return (onset / grid).denominator == 1


def _can_follow(value: str, onset: Fraction, after_triplet: bool, beat: Fraction) -> bool:
    """Whether `value` can start at `onset`, as music is engraved.

    A run of triplet notes starts on a beat and closes on one, and a plain note
    starts on the sixteenth grid outside a triplet. A reading of five triplet
    eighths and then a quarter adds up, and nobody engraves it.
    """
    if _triplet(value):
        return after_triplet and _on(onset, Fraction(1, 24)) or _on(onset, beat)
    if after_triplet and not _on(onset, beat):
        return False
    return _on(onset, Fraction(1, 16))


def best_readings(
    options: list[list[tuple[str, float]]],
    length: Fraction,
    top: int = TOP,
    beat: Fraction = Fraction(1, 4),
) -> list[tuple[list[str], float]]:
    """The `top` likeliest choices of one value per moment that fill `length` exactly."""
    State = tuple[Fraction, bool]  # time so far, and whether the last value was a triplet one
    states: dict[State, list[tuple[float, list[str]]]] = {(Fraction(0), False): [(0.0, [])]}
    for choices in options:
        following: dict[State, list[tuple[float, list[str]]]] = defaultdict(list)
        for (elapsed, after_triplet), paths in states.items():
            for value, score in choices:
                duration = _duration(value)
                if duration is None or elapsed + duration > length:
                    continue
                if not _can_follow(value, elapsed, after_triplet, beat):
                    continue
                following[(elapsed + duration, _triplet(value))].extend(
                    (s + score, [*p, value]) for s, p in paths
                )
        states = {
            state: sorted(paths, key=lambda sp: -sp[0])[:top] for state, paths in following.items()
        }
    finished = [
        path
        for (elapsed, after_triplet), paths in states.items()
        if elapsed == length and (not after_triplet or _on(elapsed, beat))
        for path in paths
    ]
    finished.sort(key=lambda sp: -sp[0])
    return [(path, score) for score, path in finished[:top]]


def _beat(measures: list[ET.Element], bar: int) -> Fraction:
    """The beat a triplet run starts on: a quarter, or an eighth in an x/8 meter."""
    beat_type = 4
    for measure in measures[:bar]:
        text = measure.findtext("attributes/time/beat-type")
        if text:
            beat_type = int(text)
    return Fraction(1, max(4, beat_type))


def bar_readings(
    xml: ET.Element, staffs: list[list[EncodedSymbol]], doubts: Doubts, top: int = TOP
) -> list[dict]:
    """For each doubted bar, each voice with more than one complete reading and its best few."""
    parts = xml.findall("part")
    lengths = bar_lengths(xml)
    out = []
    for part_index, staff, bar in sorted(doubts):
        if part_index >= len(parts) or part_index >= len(staffs):
            continue
        measures = parts[part_index].findall("measure")
        if bar > len(measures) or staff > _staves_of(parts[part_index]):
            continue
        divisions = 1
        for measure in measures[: bar - 1]:
            text = measure.findtext("attributes/divisions")
            if text:
                divisions = int(text)
        voices, _ = _voices(measures[bar - 1], staff, divisions)
        heads = _heads_in_bar(staffs[part_index], staff, bar)
        length = lengths[part_index][bar - 1]
        for voice, moments in sorted(voices.items()):
            if not any(m.kind == "note" for m in moments):
                continue
            # A voice that does not run from the head of the bar without a gap
            # is not one these readings can be laid on.
            at = Fraction(0)
            gapless = True
            for moment in moments:
                gapless = gapless and moment.onset == at
                at += moment.duration
            if not gapless:
                continue
            seen: dict[tuple, int] = defaultdict(int)
            options = []
            for moment in moments:
                key = (moment.kind, *((p["step"], p["octave"]) for p in moment.pitches))
                options.append(_moment_options(moment, _heads_for(moment, seen[key], heads)))
                seen[key] += 1
            if not all(options):
                continue
            readings = best_readings(options, length, top, _beat(measures, bar))
            if len(readings) < 2:
                continue
            out.append(
                {
                    "part": part_index,
                    "staff": staff,
                    "bar": bar,
                    "voice": voice,
                    "length": str(length),
                    "moments": [
                        {
                            "kind": m.kind,
                            "pitches": m.pitches,
                            "value": value_of(m.duration, m.kind),
                        }
                        for m in moments
                    ],
                    "readings": [
                        {"values": values, "score": round(score, 3)} for values, score in readings
                    ],
                }
            )
    return out


def embed_readings(xml: ET.Element, readings: list[dict]) -> None:
    """Write the readings into the score's identification, replacing any there."""
    identification = xml.find("identification")
    if identification is None:
        identification = ET.Element("identification")
        index = next(
            (i for i, child in enumerate(xml) if child.tag in ("defaults", "credit", "part-list")),
            len(xml),
        )
        xml.insert(index, identification)
    miscellaneous = identification.find("miscellaneous")
    if miscellaneous is None:
        miscellaneous = ET.SubElement(identification, "miscellaneous")
    for field in miscellaneous.findall("miscellaneous-field"):
        if field.get("name") == FIELD:
            miscellaneous.remove(field)
    field = ET.SubElement(miscellaneous, "miscellaneous-field", name=FIELD)
    field.text = json.dumps({"version": VERSION, "bars": readings}, separators=(",", ":"))


def read_readings(xml: ET.Element) -> list[dict]:
    """The readings a score carries, or none."""
    for field in xml.iterfind("identification/miscellaneous/miscellaneous-field"):
        if field.get("name") == FIELD and field.text:
            return json.loads(field.text).get("bars", [])
    return []
