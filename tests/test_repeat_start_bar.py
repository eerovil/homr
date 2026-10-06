"""A start-repeat after the clef and key opens the bar it stands in.

eerovil/musescore-choir-plugins#274: on Shakkitarina and Kantajani homr read the
opening of a system as `clef keySignature_0 repeatStart keySignature_-4 ...`.
The start-repeat closed the bar it was in, so the clef and the empty key
became a bar of their own and those staves ran one bar behind the others.
"""

import unittest
import xml.etree.ElementTree as ET

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.transformer.vocabulary import EncodedSymbol


def _measures(xml: ET.Element) -> list[ET.Element]:
    return xml.findall(".//part/measure")


def _tokens(*rhythms_and_pitches: str) -> list[EncodedSymbol]:
    out = []
    for token in rhythms_and_pitches:
        rhythm, _, pitch = token.partition(":")
        out.append(EncodedSymbol(rhythm, pitch) if pitch else EncodedSymbol(rhythm))
    return out


class TestRepeatStartBar(unittest.TestCase):
    def setUp(self) -> None:
        self.tokens = _tokens(
            "clef_G2", "keySignature_0", "repeatStart", "keySignature_-4", "timeSignature/4",
            "rest_1", "barline", "note_4:G4", "note_4:G4", "rest_2", "repeatEnd",
            "note_1:F4", "barline",
        )  # fmt: skip
        self.xml = generate_xml(XmlGeneratorArguments(), [self.tokens], "")

    def test_no_bar_is_made_of_the_clef_and_key_alone(self) -> None:
        measures = _measures(self.xml)
        self.assertEqual(len(measures), 3)
        self.assertTrue(all(m.find("note") is not None for m in measures))

    def test_the_repeat_opens_the_first_bar_on_its_left(self) -> None:
        first = _measures(self.xml)[0]
        repeat = first.find("barline[@location='left']/repeat")
        assert repeat is not None
        self.assertEqual(repeat.get("direction"), "forward")

    def test_the_bar_is_in_the_later_key_only(self) -> None:
        first = _measures(self.xml)[0]
        self.assertEqual([k.findtext("fifths") for k in first.iter("key")], ["-4"])

    def test_a_start_repeat_after_music_still_starts_a_new_bar(self) -> None:
        tokens = _tokens("clef_G2", "keySignature_0", "timeSignature/4", "note_1:G4",
                         "repeatStart", "note_1:A4", "barline")  # fmt: skip
        measures = _measures(generate_xml(XmlGeneratorArguments(), [tokens], ""))
        self.assertEqual(len(measures), 2)
        self.assertIsNotNone(measures[1].find("barline[@location='left']/repeat"))

    def test_a_key_change_after_music_is_kept(self) -> None:
        tokens = _tokens("clef_G2", "keySignature_0", "timeSignature/4", "note_2:G4",
                         "keySignature_1", "note_2:F4", "barline")  # fmt: skip
        first = _measures(generate_xml(XmlGeneratorArguments(), [tokens], ""))[0]
        self.assertEqual([k.findtext("fifths") for k in first.iter("key")], ["0", "1"])


if __name__ == "__main__":
    unittest.main()


def test_a_system_with_no_opening_signature_is_labelled_by_its_first_bar() -> None:
    """eerovil/musescore-choir-plugins#274, Kantajani s6: the 4/4 carries over from
    the system before, the 3/4 is printed a bar later; bar 1 was labelled 3/4."""
    tokens = _tokens(
        "clef_G2", "keySignature_-1",
        "note_4:E4", "note_4:F4", "note_4:G4", "note_4:A4", "repeatEnd",
        "timeSignature/4", "note_4:D4", "note_4:D4", "note_4:D4", "barline",
        "note_4:C4", "note_4:C4", "note_4:C4", "barline",
        "note_4:D4", "note_4:D4", "note_4:D4", "barline",
    )  # fmt: skip
    measures = _measures(generate_xml(XmlGeneratorArguments(), [tokens], ""))
    beats = [m.findtext(".//time/beats") for m in measures]
    assert beats[0] == "4"
    assert beats[1] == "3"
