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

eerovil/musescore-choir-plugins#295: a bar marked "notes?" is one the second
reading (the crop read again at 80% of its size) read differently, and that
second reading is a whole bar a person may want instead -- pitches and lengths
together. `second_readings` keeps it, per voice, beside the first.
"""

import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from fractions import Fraction
from itertools import permutations

from homr.doubt import (
    _ALTERNATIVES,
    _LIFT,
    _PITCH,
    _UNRANKED,
    SECOND_READING,
    Doubts,
    _confidence,
    _duration,
    _probability,
    _staff,
    _staves_of,
    bar_lengths,
)
from homr.transformer.vocabulary import EncodedSymbol

FIELD = "homr-bar-readings"
#: 2 adds `notes`: other pitches for the notes of a doubted bar the decoder was
#: unsure of the pitch or the accidental of (eerovil/musescore-choir-plugins#290).
#: 3 adds `second`: a doubted bar as the second reading read it (#295), and
#: gives the pitch a note was written with the decoder's probability for it.
VERSION = 3

#: How many readings of a voice are offered at most.
TOP = 3

_ALTER = {"#": 1, "b": -1, "N": 0}
_LIFT_ALTER = {"#": 1, "##": 2, "N": 0, "b": -1, "bb": -2}
_SHARP_ORDER = "FCGDAEB"
#: An alternative the decoder gave less than this is not worth a person's look.
_PITCH_FLOOR = 0.02


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
    targets: list[tuple[str, int] | None] = (
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


def _key_alter(step: str, fifths: int) -> int:
    if fifths > 0 and step in _SHARP_ORDER[:fifths]:
        return 1
    if fifths < 0 and step in _SHARP_ORDER[::-1][:-fifths]:
        return -1
    return 0


def _fifths(measures: list[ET.Element], bar: int) -> int:
    fifths = 0
    for measure in measures[:bar]:
        text = measure.findtext("attributes/key/fifths")
        if text:
            fifths = int(text)
    return fifths


def _alternative_pitches(written: dict, head: EncodedSymbol, fifths: int, top: int) -> list[dict]:
    """The pitches one written note could be: as written first, then the decoder's next.

    A pitch alternative keeps the step's accidental the key gives (nothing else was
    printed in that reading); an accidental alternative keeps the step and octave.
    Each carries the decoder's probability for the field that changed.
    """
    # The pitch as written is the decoder's own first choice for the field it doubted,
    # so it carries that probability, and a reader can rank it against the others.
    doubted = "pitch" if _probability(head, "pitch") < _PITCH else "lift"
    out = [{**written, "probability": round(_probability(head, doubted), 3)}]
    seen = {(written["step"], written["alter"], written["octave"])}
    candidates: list[tuple[float, dict]] = []
    if _probability(head, "pitch") < _PITCH:
        for alternative in _confidence(head, "pitch").get("alternatives", [])[:_ALTERNATIVES]:
            value = alternative["value"]
            if len(value) < 2 or value[0] not in "CDEFGAB" or not value[1:].isdigit():
                continue
            step, octave = value[0], int(value[1:])
            candidates.append(
                (
                    alternative["probability"],
                    {"step": step, "alter": _key_alter(step, fifths), "octave": octave},
                )
            )
    if _probability(head, "lift") < _LIFT:
        for alternative in _confidence(head, "lift").get("alternatives", [])[:_ALTERNATIVES]:
            value = alternative["value"]
            if value == "_":
                alter = _key_alter(written["step"], fifths)
            elif value in _LIFT_ALTER:
                alter = _LIFT_ALTER[value]
            else:
                continue
            candidates.append((alternative["probability"], {**written, "alter": alter}))
    for probability, pitch in sorted(candidates, key=lambda c: -c[0]):
        key = (pitch["step"], pitch["alter"], pitch["octave"])
        if probability < _PITCH_FLOOR or key in seen:
            continue
        seen.add(key)
        out.append({**pitch, "probability": round(probability, 3)})
    return out[:top]


def note_readings(
    xml: ET.Element, staffs: list[list[EncodedSymbol]], doubts: Doubts, top: int = TOP
) -> list[dict]:
    """For each doubted bar, each note whose pitch or accidental the decoder doubted.

    One entry per notehead with more than one pitch worth offering: the voice as
    written (so a reader can find the bar again), which moment and which head of
    its chord, and the pitches -- as written first, then the decoder's next.
    """
    parts = xml.findall("part")
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
        fifths = _fifths(measures, bar)
        for voice, moments in sorted(voices.items()):
            written = [
                {"kind": m.kind, "pitches": m.pitches, "value": value_of(m.duration, m.kind)}
                for m in moments
            ]
            # The k-th head at a pitch in the voice is the k-th decoded head at it.
            seen: dict[tuple[str, int], int] = defaultdict(int)
            for index, moment in enumerate(moments):
                if moment.kind != "note":
                    continue
                for chord, pitch in enumerate(moment.pitches):
                    occurrence = seen[(pitch["step"], pitch["octave"])]
                    seen[(pitch["step"], pitch["octave"])] += 1
                    single = _Moment(moment.onset, "note", moment.duration)
                    single.pitches = [pitch]
                    matched = _heads_for(single, occurrence, heads)
                    if not matched:
                        continue
                    pitches = _alternative_pitches(pitch, matched[0], fifths, top)
                    if len(pitches) < 2:
                        continue
                    out.append(
                        {
                            "part": part_index,
                            "staff": staff,
                            "bar": bar,
                            "voice": voice,
                            "moment": index,
                            "chord": chord,
                            "moments": written,
                            "pitches": pitches,
                        }
                    )
    return out


def _divisions_before(measures: list[ET.Element], bar: int) -> int:
    divisions = 1
    for measure in measures[: bar - 1]:
        text = measure.findtext("attributes/divisions")
        if text:
            divisions = int(text)
    return divisions


def _sounding(voices: dict[str, list]) -> dict[str, list]:
    return {v: m for v, m in voices.items() if any(x.kind == "note" for x in m)}


def _fills(moments: list, length: Fraction) -> bool:
    at = Fraction(0)
    for moment in moments:
        if moment.onset != at:
            return False
        at += moment.duration
    return at == length


def _heard(moments: list) -> set[tuple]:
    # The accidental is part of the note: two voices a semitone apart on one step
    # are different lines, and pairing them by step alone could hand a voice the
    # other's second reading.
    return {
        (m.onset, p["step"], p.get("alter", 0), p["octave"]) for m in moments for p in m.pitches
    }


def _as_written(moments: list) -> list[dict] | None:
    out = []
    for m in moments:
        value = value_of(m.duration, m.kind)
        if value is None:
            return None
        out.append({"kind": m.kind, "pitches": m.pitches, "value": value})
    return out


def second_readings(xml: ET.Element, second: ET.Element | None, doubts: Doubts) -> list[dict]:
    """For each bar the second reading read differently, each voice as it read it.

    Only where the two can be laid side by side: the same parts, staves and bars,
    the same number of sounding voices on the staff -- paired by the notes they
    share most, since a voice's number means nothing between two readings -- and
    both readings of the voice running from the head of the bar to its end
    without a gap. A voice the second reading read the same is left out.
    """
    if second is None:
        return []
    parts, theirs = xml.findall("part"), second.findall("part")
    if len(parts) != len(theirs):
        return []
    lengths = bar_lengths(xml)
    out = []
    for key in sorted(doubts):
        part_index, staff, bar = key
        if not any(reason.endswith(SECOND_READING) for reason in doubts[key]):
            continue
        if part_index >= len(parts) or part_index >= len(lengths):
            continue
        mine, other = parts[part_index], theirs[part_index]
        ours, read = mine.findall("measure"), other.findall("measure")
        if len(ours) != len(read) or bar > len(ours):
            continue
        if _staves_of(mine) != _staves_of(other) or staff > _staves_of(mine):
            continue
        first, _ = _voices(ours[bar - 1], staff, _divisions_before(ours, bar))
        again, _ = _voices(read[bar - 1], staff, _divisions_before(read, bar))
        first, again = _sounding(first), _sounding(again)
        if not first or len(first) != len(again):
            continue
        names = sorted(first)
        best = max(
            permutations(sorted(again)),
            key=lambda order: sum(
                len(_heard(first[a]) & _heard(again[b])) for a, b in zip(names, order, strict=True)
            ),
        )
        length = lengths[part_index][bar - 1]
        for voice, paired in zip(names, best, strict=True):
            if not (_fills(first[voice], length) and _fills(again[paired], length)):
                continue
            written, reread = _as_written(first[voice]), _as_written(again[paired])
            if written is None or reread is None or written == reread:
                continue
            out.append(
                {
                    "part": part_index,
                    "staff": staff,
                    "bar": bar,
                    "voice": voice,
                    "length": str(length),
                    "moments": written,
                    "second": reread,
                }
            )
    return out


def embed_readings(
    xml: ET.Element,
    readings: list[dict],
    notes: list[dict] | None = None,
    second: list[dict] | None = None,
) -> None:
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
    field.text = json.dumps(
        {"version": VERSION, "bars": readings, "notes": notes or [], "second": second or []},
        separators=(",", ":"),
    )


def read_readings(xml: ET.Element) -> list[dict]:
    """The readings a score carries, or none."""
    for field in xml.iterfind("identification/miscellaneous/miscellaneous-field"):
        if field.get("name") == FIELD and field.text:
            return json.loads(field.text).get("bars", [])
    return []


def read_note_readings(xml: ET.Element) -> list[dict]:
    """The other pitches a score carries for its doubted notes, or none (version 1)."""
    for field in xml.iterfind("identification/miscellaneous/miscellaneous-field"):
        if field.get("name") == FIELD and field.text:
            return json.loads(field.text).get("notes", [])
    return []


def read_second_readings(xml: ET.Element) -> list[dict]:
    """The bars a score carries as the second reading read them, or none (version 3)."""
    for field in xml.iterfind("identification/miscellaneous/miscellaneous-field"):
        if field.get("name") == FIELD and field.text:
            return json.loads(field.text).get("second", [])
    return []
