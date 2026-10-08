"""Resolve ambiguous slur-token streams before homr emits MusicXML.

The transformer predicts slur starts and stops independently. MusicXML pairs
those marks by ``number``, but homr numbers slurs by staff, so a missed stop can
leave a start open until an unrelated later stop. On the choir benchmark every
verified slur spans at most one barline; longer pairs are recognition accidents.

This module deliberately knows only about the generated MusicXML part. Tie
conversion runs first, because ties and slurs share the model class and genuine
ties must be removed from the slur stream before it is repaired.

**A loose end at the edge of the part is kept** (eerovil/musescore-choir-plugins#318).
A part is one system when the choir app reads a page a system at a time, and a
slur the page carries over a system break is a start in the last bar with no
stop, or a stop in the first bar with no start. Dropping those, as every other
loose end is dropped, lost every slur crossing a break. Kept, the app pairs the
two halves when it joins the systems. A reader that does not join them loses
nothing either: MuseScore drops an unpaired slur when it opens the file.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

MAX_SLUR_BARS = 1

#: How many bars from the end of the part a start left open may stand in and
#: still be a slur over the system edge. Two, because a slur crossing one
#: barline before the edge is common (Finlandia's basses, bars 7-8): measured on
#: the owner-checked key for eerovil/musescore-choir-plugins#318, one bar keeps
#: 17 of 21 arc ends at the edge and two keep 19, with nothing more invented.
EDGE_BARS = 2

#: Marks a slur stop inferred here rather than read: the doubt pass puts
#: `slur?` on its bar, then `forget_inferred` takes the mark off.
INFERRED = "homr-inferred-slur"

# note, notations, slur
_SlurRef = tuple[ET.Element, ET.Element, ET.Element]
# zero-based measure index, note, notations, slur
_OpenSlur = tuple[int, ET.Element, ET.Element, ET.Element]


def resolve_slurs(
    part: ET.Element,
    max_bars: int = MAX_SLUR_BARS,
    keep_edges: bool = True,
    edge_bars: int = EDGE_BARS,
) -> int:
    """Leave one unambiguous, short slur stream in ``part``.

    Starts/stops are paired independently per MusicXML slur number. A genuine
    pair may cross at most ``max_bars`` barlines. Redundant starts, unmatched
    stops, trailing starts, and both ends of longer pairs are removed. Empty
    ``<notations>`` containers created by the removals are removed as well.

    With ``keep_edges`` an unmatched stop in the first bar and every start left
    unmatched in the last ``edge_bars`` bars stay: they are slurs crossing the
    system edge, which the choir app joins. That includes a start made while
    its number was open, the rule that otherwise removes it: at the edge a slur
    and a tie both running into the next system are two such starts, and
    keeping only the first lost the second (ties at the edge 36 -> 41 of 43 on
    the owner-checked key).

    Returns the number of over-long pairs dropped. The operation is idempotent.
    """
    doomed: list[_SlurRef] = []
    dropped = 0
    open_slurs: dict[str, _OpenSlur] = {}
    pairs: list[tuple[ET.Element, ET.Element, ET.Element, ET.Element]] = []
    # Starts made while their number was already open, and whether a stop of
    # that number came after them.
    redundant: list[tuple[_OpenSlur, str]] = []
    stopped_after: set[int] = set()

    measures = part.findall("measure")
    last = len(measures) - 1
    for bar, measure in enumerate(measures):
        for note in measure.findall("note"):
            for notations in note.findall("notations"):
                for slur in list(notations.findall("slur")):
                    number = slur.get("number", "1")
                    kind = slur.get("type")
                    if kind == "start":
                        if number in open_slurs:
                            # MusicXML cannot distinguish overlapping slurs
                            # that share a number. Keep the first open start.
                            redundant.append(((bar, note, notations, slur), number))
                        else:
                            open_slurs[number] = (bar, note, notations, slur)
                    elif kind == "stop":
                        for index, (_, earlier) in enumerate(redundant):
                            if earlier == number:
                                stopped_after.add(index)
                        began = open_slurs.pop(number, None)
                        if began is None:
                            if not (keep_edges and bar == 0):
                                doomed.append((note, notations, slur))
                        elif bar - began[0] > max_bars:
                            doomed.append((began[1], began[2], began[3]))
                            doomed.append((note, notations, slur))
                            dropped += 1
                        else:
                            pairs.append((began[1], began[3], note, slur))

    def at_edge(bar: int) -> bool:
        return keep_edges and bar > last - edge_bars

    left_open = list(open_slurs.values()) + [
        began for index, (began, _) in enumerate(redundant) if index not in stopped_after
    ]
    doomed.extend(
        (began[1], began[2], began[3])
        for index, (began, _) in enumerate(redundant)
        if index in stopped_after
    )
    for bar, note, notations, slur in left_open:
        if _close_on_tie(measures, bar, note, slur, max_bars):
            continue
        if not at_edge(bar):
            doomed.append((note, notations, slur))

    for note, notations, slur in doomed:
        if slur in list(notations):
            notations.remove(slur)
        if len(notations) == 0 and notations in list(note):
            note.remove(notations)

    for start_note, start, stop_note, stop in pairs:
        _onto_one_voice(measures, start_note, start, stop_note, stop)

    return dropped


def forget_inferred(xml: ET.Element) -> None:
    """Take the working mark off every inferred slur stop."""
    for slur in xml.iter("slur"):
        slur.attrib.pop(INFERRED, None)


def _voice(note: ET.Element) -> tuple[str, str]:
    return note.findtext("staff", "1"), note.findtext("voice", "1")


def _pitch(note: ET.Element) -> tuple:
    pitch = note.find("pitch")
    if pitch is None:
        return ()
    return (pitch.findtext("step"), pitch.findtext("octave"), pitch.findtext("alter") or "0")


def _has_tie(note: ET.Element, kind: str) -> bool:
    return any(tie.get("type") == kind for tie in note.findall("tie"))


def _close_on_tie(
    measures: list[ET.Element], bar: int, start: ET.Element, slur: ET.Element, max_bars: int
) -> bool:
    """Close a start left open on the end of the tie that follows it.

    A slur drawn over a tie ends on the note the tie ends on, and the model
    writes one arc mark per note: Vielako s01 bar 8 reads a start on D, a start
    on the E tied over and one stop on the E it is tied to, and once the tie
    takes that stop the slur from D has none. So a start left open closes on
    the stop note of the first tie in its own voice that starts after it,
    within ``max_bars`` barlines. The stop is inferred, not read, and is marked
    (`INFERRED`) so the doubt pass says `slur?` there.
    """
    voice = _voice(start)
    seen = False
    tied: ET.Element | None = None
    for index in range(bar, min(len(measures), bar + max_bars + 1)):
        for note in measures[index].findall("note"):
            if note is start:
                seen = True
                continue
            if not seen or note.find("chord") is not None or _voice(note) != voice:
                continue
            if tied is None:
                if _has_tie(note, "start"):
                    tied = note
                continue
            if _has_tie(note, "stop") and _pitch(note) == _pitch(tied):
                if any(s.get("type") == "stop" for s in note.iter("slur")):
                    return False
                notations = note.find("notations")
                if notations is None:
                    notations = ET.SubElement(note, "notations")
                stop = ET.SubElement(notations, "slur", type="stop", number=slur.get("number", "1"))
                stop.set(INFERRED, "yes")
                return True
            return False
    return False


def _onsets(measure: ET.Element) -> dict[int, int]:
    """Each note's onset in its bar, in divisions, keyed by id."""
    found: dict[int, int] = {}
    at = previous = 0
    for node in measure:
        if node.tag == "backup":
            at -= int(node.findtext("duration", "0") or 0)
        elif node.tag == "forward":
            at += int(node.findtext("duration", "0") or 0)
        elif node.tag == "note":
            chord = node.find("chord") is not None
            onset = previous if chord else at
            if not chord and node.find("grace") is None:
                previous, at = at, at + int(node.findtext("duration", "0") or 0)
            found[id(node)] = onset
    return found


def _twin(
    measures: list[ET.Element], note: ET.Element, voice: tuple[str, str]
) -> ET.Element | None:
    """The same notehead written into ``voice``: same bar, beat and pitch."""
    for measure in measures:
        notes = measure.findall("note")
        if note not in notes:
            continue
        onsets = _onsets(measure)
        for other in notes:
            if (
                other is not note
                and _voice(other) == voice
                and onsets.get(id(other)) == onsets.get(id(note))
                and _pitch(other)
                and _pitch(other) == _pitch(note)
            ):
                return other
        return None
    return None


def _move(slur: ET.Element, source: ET.Element, target: ET.Element) -> None:
    notations = source.find("notations")
    if notations is not None and slur in list(notations):
        notations.remove(slur)
        if len(notations) == 0:
            source.remove(notations)
    into = target.find("notations")
    if into is None:
        into = ET.SubElement(target, "notations")
    into.append(slur)


def _onto_one_voice(
    measures: list[ET.Element],
    start_note: ET.Element,
    start: ET.Element,
    stop_note: ET.Element,
    stop: ET.Element,
) -> None:
    """Put a slur from one voice into the other back on a single voice, where
    one end sits on a notehead both voices share.

    Two voices singing one pitch share a notehead, written once in each voice,
    and which copy the model hangs a slur on is a coin toss: Illan s05 bar 1
    starts its slur on voice 2's copy of a G both voices hold and ends it on
    voice 1's F. A slur from one singer into the other is then taken out by
    the choir app's clean. The shared end moves to the other end's voice."""
    if _voice(start_note) == _voice(stop_note):
        return
    twin = _twin(measures, start_note, _voice(stop_note))
    if twin is not None:
        _move(start, start_note, twin)
        return
    twin = _twin(measures, stop_note, _voice(start_note))
    if twin is not None:
        _move(stop, stop_note, twin)
