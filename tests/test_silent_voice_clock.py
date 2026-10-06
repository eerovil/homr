"""A voice that falls silent does not pull the next moment in early.

eerovil/musescore-choir-plugins#274, Illan viimeinen tango s3 bar 3, in made-up
pitches: four voices start on a dotted eighth together, the lowest misread as a
plain eighth, and that voice is silent after it. The writer's one clock started
the next moment where that eighth ended, a sixteenth early for the three voices
that go on.
"""

import xml.etree.ElementTree as ET

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.transformer.vocabulary import EncodedSymbol

#: The decoder's own symbols for the bar (rhythm:pitch~stem@position), every pitch
#: moved up a step so they are not the song's notes.
BAR = (
    "note_8.:D5~up chord note_8.:A4~down@upper2 chord note_8.:F4b~up@lower chord "
    "note_8:D4~down@lower2 note_16:A4~up chord rest_16@upper2 chord rest_16@lower "
    "note_8:G4#~up chord rest_4@upper2 chord rest_4@lower note_8:A4~up "
)


def stream(bar: str) -> list[EncodedSymbol]:
    meter = EncodedSymbol("timeSignature/8")
    meter.printed_meters = ((4, 8),)
    out = [
        EncodedSymbol("clef_G2", position="upper"),
        EncodedSymbol("chord"),
        EncodedSymbol("clef_G2", position="lower"),
        EncodedSymbol("keySignature_0"),
        meter,
    ]
    for word in bar.split():
        if word == "chord":
            out.append(EncodedSymbol("chord"))
            continue
        head, _, position = word.partition("@")
        head, _, stem = head.partition("~")
        rhythm, _, pitch = head.partition(":")
        lift = pitch[-1] if pitch and pitch[-1] in "#b" else "_"
        pitch = pitch.rstrip("#b")
        symbol = EncodedSymbol(
            rhythm, pitch or ".", lift if pitch else ".", position=position or "upper"
        )
        symbol.stem_direction = stem or None
        out.append(symbol)
    out.append(EncodedSymbol("barline"))
    return out


def onsets(measure: ET.Element) -> dict[str, list[tuple[int, str]]]:
    """Each voice's notes as (onset, pitch), walking the measure's own cursor."""
    found: dict[str, list[tuple[int, str]]] = {}
    cursor = last = 0
    for child in measure:
        if child.tag in ("backup", "forward"):
            step = int(child.findtext("duration") or 0)
            cursor += -step if child.tag == "backup" else step
            continue
        if child.tag != "note":
            continue
        duration = int(child.findtext("duration") or 0)
        chord = child.find("chord") is not None
        at = last if chord else cursor
        if not chord:
            last, cursor = cursor, cursor + duration
        pitch = child.find("pitch")
        if pitch is not None:
            name = f"{pitch.findtext('step')}{pitch.findtext('octave')}"
            found.setdefault(child.findtext("voice") or "", []).append((at, name))
    return found


def test_the_voices_that_go_on_start_where_their_own_notes_end() -> None:
    measure = generate_xml(XmlGeneratorArguments(), [stream(BAR)], "").find(".//part/measure")
    assert measure is not None
    first_voice = onsets(measure)["1"]
    # Dotted eighth, then the sixteenth on its own beat: 3 sixteenths in.
    assert [at for at, _ in first_voice][:2] == [0, 3]
