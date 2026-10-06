"""A bar short of a symbol is left for the writer, not re-solved value by value.

eerovil/musescore-choir-plugins#274, Illan viimeinen tango s1 bar 2, in made-up
pitches: two voices on each of two staves; the second voice of each rests, then
all four sing the last three sixteenths with the heads shared (two stems). The
decoder wrote the first shared head once, for the upper voice, and left out the
lower staff's sixteenth rest. With the printed 4/8 in hand, solve_bar_rhythms
rebuilt those short voices one value after another and their last notes sounded
a sixteenth early.
"""

import xml.etree.ElementTree as ET

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.transformer.vocabulary import EncodedSymbol

#: The decoder's own symbols for the bar, as homr reads them (rhythm:pitch~stem@position),
#: every pitch moved up a step so they are not the song's notes. `chord` joins a
#: symbol to the moment before it.
BAR = (
    "note_16:E5~up chord note_16:A4~down@upper2 chord note_16:E4~down@upper2 chord "
    "note_16:C4~up@lower chord note_16:A3~down@lower2 note_8:E5~up chord rest_8.@upper2 "
    "chord rest_8.@lower2 chord rest_8.@lower note_16:D5#~up note_16:E5~up chord "
    "rest_16@upper2 chord rest_16@lower2 note_16:A4~both chord note_16:A3~both@lower2 "
    "chord note_16:A3~both@lower note_16:G4#~both@upper2 chord note_16:G4#~both chord "
    "note_16:G3#~both@lower2 chord note_16:G3#~both@lower note_16:A4~both@upper2 chord "
    "note_16:A4~both chord note_16:A3~both@lower2 chord note_16:A3~both@lower "
)


def stream() -> list[EncodedSymbol]:
    meter = EncodedSymbol("timeSignature/8")
    meter.printed_meters = ((4, 8),)
    out = [
        EncodedSymbol("clef_G2", position="upper"),
        EncodedSymbol("chord"),
        EncodedSymbol("clef_G2", position="lower"),
        EncodedSymbol("keySignature_0"),
        meter,
    ]
    for word in BAR.split():
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


def test_short_voices_keep_their_notes_on_the_beats_the_decoder_put_them() -> None:
    measure = generate_xml(XmlGeneratorArguments(), [stream()], "").find(".//part/measure")
    assert measure is not None
    after_the_rest = {
        voice: [note for note in notes if note[0] > 0] for voice, notes in onsets(measure).items()
    }
    # The three shared heads, on sixteenths 5, 6 and 7, in every voice; nothing earlier.
    assert after_the_rest["2"] == [(5, "A4"), (6, "G4"), (7, "A4")]
    assert after_the_rest["5"] == [(5, "A3"), (6, "G3"), (7, "A3")]
    assert after_the_rest["6"] == [(5, "A3"), (6, "G3"), (7, "A3")]
