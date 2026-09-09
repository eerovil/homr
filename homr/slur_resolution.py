"""Resolve ambiguous slur-token streams before homr emits MusicXML.

The transformer predicts slur starts and stops independently. MusicXML pairs
those marks by ``number``, but homr numbers slurs by staff, so a missed stop can
leave a start open until an unrelated later stop. On the choir benchmark every
verified slur spans at most one barline; longer pairs are recognition accidents.

This module deliberately knows only about the generated MusicXML part. Tie
conversion runs first, because ties and slurs share the model class and genuine
ties must be removed from the slur stream before it is repaired.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

MAX_SLUR_BARS = 1

# note, notations, slur
_SlurRef = tuple[ET.Element, ET.Element, ET.Element]
# zero-based measure index, note, notations, slur
_OpenSlur = tuple[int, ET.Element, ET.Element, ET.Element]


def resolve_slurs(part: ET.Element, max_bars: int = MAX_SLUR_BARS) -> int:
    """Leave one unambiguous, short slur stream in ``part``.

    Starts/stops are paired independently per MusicXML slur number. A genuine
    pair may cross at most ``max_bars`` barlines. Redundant starts, unmatched
    stops, trailing starts, and both ends of longer pairs are removed. Empty
    ``<notations>`` containers created by the removals are removed as well.

    Returns the number of over-long pairs dropped. The operation is idempotent.
    """
    doomed: list[_SlurRef] = []
    dropped = 0
    open_slurs: dict[str, _OpenSlur] = {}

    for bar, measure in enumerate(part.findall("measure")):
        for note in measure.findall("note"):
            for notations in note.findall("notations"):
                for slur in list(notations.findall("slur")):
                    number = slur.get("number", "1")
                    kind = slur.get("type")
                    if kind == "start":
                        if number in open_slurs:
                            # MusicXML cannot distinguish overlapping slurs
                            # that share a number. Keep the first open start.
                            doomed.append((note, notations, slur))
                        else:
                            open_slurs[number] = (bar, note, notations, slur)
                    elif kind == "stop":
                        began = open_slurs.pop(number, None)
                        if began is None:
                            doomed.append((note, notations, slur))
                        elif bar - began[0] > max_bars:
                            doomed.append((began[1], began[2], began[3]))
                            doomed.append((note, notations, slur))
                            dropped += 1

    doomed.extend((note, notations, slur) for _, note, notations, slur in open_slurs.values())

    for note, notations, slur in doomed:
        notations.remove(slur)
        if len(notations) == 0:
            note.remove(notations)

    return dropped
