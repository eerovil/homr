# ruff: noqa: E501, S101

import unittest
import xml.etree.ElementTree as ET

from homr.music_xml_generator import (
    SymbolChord,
    XmlGeneratorArguments,
    convert_ties,
    generate_xml,
    rebalance_measure_voices,
)
from homr.transformer.vocabulary import EncodedSymbol
from training.transformer.training_vocabulary import (
    read_token_lines,
)


def _notes(measure: ET.Element) -> list[ET.Element]:
    return [c for c in measure if c.tag == "note"]


def _pitch(note: ET.Element) -> str:
    p = note.find("pitch")
    if p is None:
        return "rest"
    step = p.findtext("step", "")
    return step


def _duration(note: ET.Element) -> int:
    d = note.findtext("duration")
    return int(d) if d is not None else 0


def _voice(note: ET.Element) -> str:
    return note.findtext("voice", "")


def _staff(note: ET.Element) -> str:
    return note.findtext("staff", "")


def _beats(measure: ET.Element) -> list[int]:
    """The beat each note sounds on, walking the measure's own cursor."""
    at = 0
    last = 0
    starts: list[int] = []
    for child in measure:
        if child.tag == "backup":
            at -= int(child.findtext("duration", "0"))
        elif child.tag == "forward":
            at += int(child.findtext("duration", "0"))
        elif child.tag == "note":
            is_chord = child.find("chord") is not None
            starts.append(last if is_chord else at)
            if not is_chord:
                last = at
                at += int(child.findtext("duration", "0"))
    return starts


def _backups(measure: ET.Element) -> list[int]:
    return [int(c.findtext("duration", "0")) for c in measure if c.tag == "backup"]


def _ties(xml: ET.Element) -> list[str]:
    return [t.get("type", "") for t in xml.iter("tie")]


def _tieds(xml: ET.Element) -> list[str]:
    return [t.get("type", "") for t in xml.iter("tied")]


def _slurs(xml: ET.Element) -> list[str]:
    return [s.get("type", "") for s in xml.iter("slur")]


def _first_measure(xml: ET.Element) -> ET.Element:
    part = xml.find("part")
    assert part is not None
    m = part.find("measure")
    assert m is not None
    return m


class TestMusicXmlGenerator(unittest.TestCase):
    """
    MusicXML testing is mostly covered by training/validate_music_xml_conversion.py
    This script requires that the data sets are downloaded and converted and uses
    the data sets to check that back and forth conversion works.
    """

    def test_chord_with_different_duratons(self) -> None:
        tabi_measure_18_upper = """clef_G2 . . . . upper
keySignature_4 . . . . .
timeSignature/8 . . . . .
note_4. G3 # _ _ upper &note_4. C4 # _ _ upper&note_16 E4 # _ _ upper
note_16 F4 # _ _ upper
note_4 E4 # _ _ upper
note_8 E4 # _ _ upper
note_8 C4 # _ _ upper
note_8 D4 # _ _ upper
barline . . . . ."""
        tokens = read_token_lines(tabi_measure_18_upper.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        measure = _first_measure(xml)
        notes = _notes(measure)
        backups = _backups(measure)

        # Pitches in order after rebalancing
        pitches = [_pitch(n) for n in notes]
        self.assertIn("E", pitches)
        self.assertIn("G", pitches)
        self.assertIn("F", pitches)
        self.assertIn("D", pitches)

        # There must be backups due to chord with different durations
        self.assertGreater(len(backups), 0)

        # All notes have a voice and staff assigned
        for note in notes:
            self.assertNotEqual(_voice(note), "")
            self.assertEqual(_staff(note), "1")

    def test_one_staff_is_untouched_by_per_staff_cursors(self) -> None:
        """A part with one staff has one stream, so nothing about it moves.

        Most of this repertoire is one staff to a part, and the cursors are what
        every note's beat is measured from -- so the change has to be provably
        confined to the parts that carry two staves.
        """
        one_staff = """clef_G2 _ _ _ _ upper
timeSignature/4 . . . . .
note_2 C4 _ _ _ upper&note_4 E4 _ _ _ upper
note_4 F4 _ _ _ upper
note_4 G4 _ _ _ upper
barline . . . . ."""
        tokens = read_token_lines(one_staff.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        measure = _first_measure(xml)
        self.assertEqual(
            [(_pitch(n), _duration(n)) for n in _notes(measure)],
            [("E", 1), ("C", 2), ("F", 1), ("G", 1)],
        )
        self.assertEqual(_beats(measure), [0, 0, 1, 2])
        self.assertNotIn("forward", [child.tag for child in measure])

    def test_a_misread_duration_moves_only_its_own_staff(self) -> None:
        """The upper staff is read exactly right, and stays where it was read.

        Both staves play two eighths and then a note; the lower staff's second
        note comes back a sixteenth, which the page prints as an eighth. Under
        one shared cursor the moment advanced by the shortest note in it, so the
        upper staff's third note landed a sixteenth early -- a wrong beat on a
        staff nothing had misread. Each staff now advances by its own note, so
        the lower staff carries its own error and nothing else does.
        """
        misread = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
timeSignature/4 . . . . .
note_8 C5 _ _ _ upper&note_8 C3 _ _ _ lower
note_8 D5 _ _ _ upper&note_16 D3 _ _ _ lower
note_4 E5 _ _ _ upper&note_8 E3 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(misread.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        measure = _first_measure(xml)
        divisions = int(measure.findtext("attributes/divisions", "1"))
        upper = [
            beat
            for beat, note in zip(_beats(measure), _notes(measure), strict=True)
            if _staff(note) == "1"
        ]
        self.assertEqual(upper, [0, divisions // 2, divisions])

    def test_grand_staff_generation(self) -> None:
        grandstaff = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_1 . . . . .
timeSignature/4 . . . . .
note_1 G4 _ _ _ upper&note_1 A3 # _ _ upper&rest_2 _ _ _ _ upper&note_4 G3 _ _ _ lower
rest_4 _ _ _ _ lower
note_2 E4 _ _ _ upper&note_2 C2 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(grandstaff.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        measure = _first_measure(xml)
        notes = _notes(measure)

        # Both staves must be present
        staves = {_staff(n) for n in notes}
        self.assertIn("1", staves)
        self.assertIn("2", staves)

        # Upper staff notes: G4, A3, rest, E4; lower: G3, rest, C2
        pitches_upper = [_pitch(n) for n in notes if _staff(n) == "1"]
        pitches_lower = [_pitch(n) for n in notes if _staff(n) == "2"]
        self.assertIn("G", pitches_upper)
        self.assertIn("E", pitches_upper)
        self.assertIn("G", pitches_lower)
        self.assertIn("C", pitches_lower)

        # Upper voices are 1-4, lower voices are 5-8
        for note in notes:
            v = int(_voice(note))
            s = int(_staff(note))
            if s == 1:
                self.assertLessEqual(v, 4)
            else:
                self.assertGreaterEqual(v, 5)

    def test_begin_chord_with_standalone_rests(self) -> None:
        """
        If the lower position consists of a standalone rest then start the
        chord with this. That fixes an issue where the upper position
        consists of tuplets because in that case backups must not be used.

        See tabi.jpg measure 9 for an example.
        """
        chord = SymbolChord(
            [
                EncodedSymbol("note_12", position="upper"),
                EncodedSymbol("note_12", position="upper"),
                EncodedSymbol("rest_8", position="lower"),
            ]
        )
        first, second = chord.into_positions()

        self.assertEqual(first.symbols, [EncodedSymbol("rest_8", position="lower")])
        self.assertEqual(
            second.symbols,
            [
                EncodedSymbol("note_12", position="upper"),
                EncodedSymbol("note_12", position="upper"),
            ],
        )

    def test_a_half_rest_does_not_swallow_the_notes_beside_it(self) -> None:
        """A rest sharing a stream with notes sounding inside it is not their silence.

        homr has no voice token, so a printed rest and the notes of the voice
        engraved beside it come out in one stream. Here the upper staff's half
        rest is written first and the upper staff sings again while the lower
        staff is only one quarter into the bar -- so the tokens place that note
        *inside* the rest, and the rest is somebody else's.

        Half, deliberately: the worst case in this repertoire is a half rest,
        and the narrow whole-bar version of this rule measured as nothing.
        """
        beside = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
timeSignature/4 . . . . .
rest_2 _ _ _ _ upper&note_8 C3 _ _ _ lower
note_8 D3 _ _ _ lower
note_4 E5 _ _ _ upper&note_8 E3 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(beside.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        measure = _first_measure(xml)
        quarter = int(measure.findtext("attributes/divisions", "1"))
        upper = [
            (_pitch(note), beat)
            for beat, note in zip(_beats(measure), _notes(measure), strict=True)
            if _staff(note) == "1"
        ]
        self.assertEqual(upper, [("rest", 0), ("E", quarter)])

        # The rest keeps its own length and its own place, so the two overlap --
        # which is how they come out as the two voices the page prints.
        voices = {_pitch(note): _voice(note) for note in _notes(measure) if _staff(note) == "1"}
        self.assertNotEqual(voices["rest"], voices["E"])

    def test_a_whole_bar_rest_does_not_swallow_them_either(self) -> None:
        """The shape the choir app repairs at its own boundary, at the source.

        A rest written for the whole bar is the loudest instance of the same
        sentence: it takes every beat there is, so the notes engraved beside it
        are written past the end of the bar.
        """
        whole = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
timeSignature/4 . . . . .
rest_1 _ _ _ _ upper&note_4 C3 _ _ _ lower
note_4 E5 _ _ _ upper&note_4 D3 _ _ _ lower
note_4 F5 _ _ _ upper&note_4 E3 _ _ _ lower
note_4 G5 _ _ _ upper&note_4 F3 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(whole.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        measure = _first_measure(xml)
        quarter = int(measure.findtext("attributes/divisions", "1"))
        upper = [
            (_pitch(note), beat)
            for beat, note in zip(_beats(measure), _notes(measure), strict=True)
            if _staff(note) == "1"
        ]
        self.assertEqual(
            upper,
            [("rest", 0), ("E", quarter), ("F", 2 * quarter), ("G", 3 * quarter)],
        )

    def test_a_rest_the_staff_is_really_silent_for_keeps_its_span(self) -> None:
        """Nothing sounds inside it, so it is silence and the lane advances.

        The upper staff rests for two quarters while the lower staff plays two,
        and sings again only once the lower staff has caught up. There is no
        note inside the rest's span, so the rest is exactly what it says.
        """
        silent = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
timeSignature/4 . . . . .
rest_2 _ _ _ _ upper&note_4 C3 _ _ _ lower
note_4 D3 _ _ _ lower
note_2 E5 _ _ _ upper&note_2 E3 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(silent.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        measure = _first_measure(xml)
        quarter = int(measure.findtext("attributes/divisions", "1"))
        upper = [
            (_pitch(note), beat)
            for beat, note in zip(_beats(measure), _notes(measure), strict=True)
            if _staff(note) == "1"
        ]
        self.assertEqual(upper, [("rest", 0), ("E", 2 * quarter)])

    def test_a_rest_is_not_disowned_by_another_rest(self) -> None:
        """Only a note can say a rest was not this stream's silence.

        Two rests in a row on the upper staff stay end to end. Piling the second
        onto the first would say the staff was silent twice over the same beats,
        which is not a reading of anything.
        """
        two_rests = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
timeSignature/4 . . . . .
rest_4 _ _ _ _ upper&note_8 C3 _ _ _ lower
rest_4 _ _ _ _ upper&note_8 D3 _ _ _ lower
note_2 E5 _ _ _ upper&note_2 E3 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(two_rests.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        measure = _first_measure(xml)
        quarter = int(measure.findtext("attributes/divisions", "1"))
        upper = [
            beat
            for beat, note in zip(_beats(measure), _notes(measure), strict=True)
            if _staff(note) == "1"
        ]
        self.assertEqual(upper[:2], [0, quarter])

    def test_one_staff_keeps_its_rests(self) -> None:
        """A part of one staff has one stream, so there is no evidence and no change.

        Which is most of this repertoire. What says a note sounds inside a rest
        is another stream standing there; with nothing to compare against, the
        rest is read as written rather than guessed at.
        """
        one_staff = """clef_G2 _ _ _ _ upper
timeSignature/4 . . . . .
rest_2 _ _ _ _ upper
note_4 F4 _ _ _ upper
note_4 G4 _ _ _ upper
barline . . . . ."""
        tokens = read_token_lines(one_staff.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        measure = _first_measure(xml)
        quarter = int(measure.findtext("attributes/divisions", "1"))
        self.assertEqual(_beats(measure), [0, 2 * quarter, 3 * quarter])
        self.assertEqual(_backups(measure), [])

    def test_rebalance_measure_voices_assigns_stable_voices_per_staff(self) -> None:
        measure = ET.Element("measure")

        note1 = self._build_test_note(duration=4, staff=1, voice=1)
        measure.append(note1)
        measure.append(self._build_test_backup(duration=4))

        note2 = self._build_test_note(duration=2, staff=1, voice=1)
        measure.append(note2)

        note3 = self._build_test_note(duration=2, staff=1, voice=1)
        measure.append(note3)

        note4 = self._build_test_note(duration=2, staff=1, voice=1, is_chord=True)
        measure.append(note4)

        measure.append(self._build_test_backup(duration=4))
        note5 = self._build_test_note(duration=4, staff=2, voice=1)
        measure.append(note5)
        measure.append(self._build_test_backup(duration=4))

        note6 = self._build_test_note(duration=2, staff=2, voice=1)
        measure.append(note6)

        rebalance_measure_voices(measure)

        self.assertEqual(self._read_note_voice(note1), "2")
        self.assertEqual(self._read_note_voice(note2), "1")
        self.assertEqual(self._read_note_voice(note3), "1")
        self.assertEqual(self._read_note_voice(note4), "1")
        self.assertEqual(self._read_note_voice(note5), "6")
        self.assertEqual(self._read_note_voice(note6), "5")

    def test_rebalance_measure_voices_prefers_stem_direction_when_available(self) -> None:
        measure = ET.Element("measure")
        up = self._build_test_note(duration=4, staff=1, voice=1)
        ET.SubElement(up, "stem").text = "up"
        down = self._build_test_note(duration=4, staff=1, voice=1)
        ET.SubElement(down, "stem").text = "down"
        measure.extend([up, self._build_test_backup(duration=4), down])

        rebalance_measure_voices(measure)

        self.assertEqual(self._read_note_voice(up), "1")
        self.assertEqual(self._read_note_voice(down), "2")

    def test_rebalance_measure_voices_keeps_stem_direction_when_voice_is_busy(self) -> None:
        measure = ET.Element("measure")
        first = self._build_test_note(duration=4, staff=1, voice=1)
        ET.SubElement(first, "stem").text = "down"
        second = self._build_test_note(duration=4, staff=1, voice=1)
        ET.SubElement(second, "stem").text = "down"
        # Stem direction identifies a voice only on a polyphonic staff.
        up = self._build_test_note(duration=4, staff=1, voice=1)
        ET.SubElement(up, "stem").text = "up"
        measure.extend(
            [
                up,
                self._build_test_backup(duration=4),
                first,
                self._build_test_backup(duration=4),
                second,
            ]
        )

        rebalance_measure_voices(measure)

        self.assertEqual(self._read_note_voice(up), "1")

        self.assertEqual(self._read_note_voice(first), "2")
        self.assertEqual(self._read_note_voice(second), "2")

    def test_rebalance_measure_voices_uses_the_agreeing_hinted_chord_tone(self) -> None:
        measure = ET.Element("measure")
        first = self._build_test_note(duration=4, staff=1, voice=1)
        ET.SubElement(first, "stem").text = "down"
        second = self._build_test_note(duration=4, staff=1, voice=1, is_chord=True)
        up = self._build_test_note(duration=4, staff=1, voice=1)
        ET.SubElement(up, "stem").text = "up"
        measure.extend([up, self._build_test_backup(duration=4), first, second])

        rebalance_measure_voices(measure)

        self.assertEqual(self._read_note_voice(up), "1")

        self.assertEqual(self._read_note_voice(first), "2")
        self.assertEqual(self._read_note_voice(second), "2")

    def test_rebalance_measure_voices_keeps_a_lone_down_stem_in_voice_one(self) -> None:
        measure = ET.Element("measure")
        note = self._build_test_note(duration=4, staff=1, voice=2)
        ET.SubElement(note, "stem").text = "down"
        measure.append(note)

        rebalance_measure_voices(measure)

        self.assertEqual(self._read_note_voice(note), "1")
        self.assertEqual(note.findtext("stem"), "down")

    def _build_test_note(
        self, duration: int, staff: int, voice: int, is_chord: bool = False
    ) -> ET.Element:
        note = ET.Element("note")
        if is_chord:
            ET.SubElement(note, "chord")
        ET.SubElement(note, "duration").text = str(duration)
        ET.SubElement(note, "staff").text = str(staff)
        ET.SubElement(note, "voice").text = str(voice)
        return note

    def _build_test_backup(self, duration: int) -> ET.Element:
        backup = ET.Element("backup")
        ET.SubElement(backup, "duration").text = str(duration)
        return backup

    def _read_note_voice(self, note: ET.Element) -> str:
        v = note.findtext("voice")
        self.assertIsNotNone(v)
        return str(v)

    def test_slur_between_same_pitches_becomes_a_tie(self) -> None:
        """A slur joining two identical pitches is a tie, and needs both elements."""
        tied = """clef_G2 . . . . upper
timeSignature/4 . . . . .
note_4 C4 _ _ slurStart upper
note_4 C4 _ _ slurStop upper
note_2 D4 _ _ _ upper
barline . . . . ."""
        tokens = read_token_lines(tied.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        # <tie> is what sounds, <tied> is what is drawn: both are required, or
        # the notes are drawn joined and still played separately.
        self.assertEqual(_ties(xml), ["start", "stop"])
        self.assertEqual(_tieds(xml), ["start", "stop"])
        # the slur it came from is gone
        self.assertEqual(_slurs(xml), [])

    def test_tie_between_two_chords(self) -> None:
        """Every notehead of a chord ties to its own pitch in the next one."""
        chords = """clef_G2 . . . . upper
timeSignature/4 . . . . .
note_4 C4 _ _ slurStart upper&note_4 E4 _ _ slurStart upper
note_4 C4 _ _ slurStop upper&note_4 E4 _ _ slurStop upper
barline . . . . ."""
        tokens = read_token_lines(chords.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        self.assertEqual(sorted(_ties(xml)), ["start", "start", "stop", "stop"])
        self.assertEqual(_slurs(xml), [])

    def test_tie_on_one_notehead_of_a_chord(self) -> None:
        """The model marks the notehead the curve touches, not the whole chord.

        On real output almost every slurred chord carries the slur on some of
        its members only, so a tie has to be found for that pitch alone and
        leave the rest of the chord as it is.
        """
        chords = """clef_G2 . . . . upper
timeSignature/4 . . . . .
note_4 C4 _ _ _ upper&note_4 E4 _ _ slurStart upper
note_4 C4 _ _ _ upper&note_4 E4 _ _ slurStop upper
barline . . . . ."""
        tokens = read_token_lines(chords.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        self.assertEqual(_ties(xml), ["start", "stop"])
        self.assertEqual(_slurs(xml), [])
        tied = [n for n in xml.iter("note") if n.find("tie") is not None]
        self.assertEqual([_pitch(n) for n in tied], ["E", "E"])

    def test_slur_from_a_chord_to_a_different_pitch_stays_a_slur(self) -> None:
        chords = """clef_G2 . . . . upper
timeSignature/4 . . . . .
note_4 C4 _ _ _ upper&note_4 E4 _ _ slurStart upper
note_4 C4 _ _ _ upper&note_4 G4 _ _ slurStop upper
barline . . . . ."""
        tokens = read_token_lines(chords.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        self.assertEqual(_slurs(xml), ["start", "stop"])
        self.assertEqual(_ties(xml), [])

    def test_phrase_slur_over_three_events_stays_a_slur(self) -> None:
        """Adjacency is what separates a tie from a phrase mark.

        A curve that leaves a pitch and comes back to it later is a phrase,
        however identical its two ends are, so only the immediately following
        event may close a tie.
        """
        phrase = """clef_G2 . . . . upper
timeSignature/4 . . . . .
note_4 C4 _ _ _ upper&note_4 G4 _ _ slurStart upper
note_4 A4 _ _ _ upper
note_4 C4 _ _ _ upper&note_4 G4 _ _ slurStop upper
barline . . . . ."""
        tokens = read_token_lines(phrase.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        self.assertEqual(_slurs(xml), ["start", "stop"])
        self.assertEqual(_ties(xml), [])

    def test_shared_pitch_without_a_slur_of_its_own_stays_untied(self) -> None:
        """A pitch in both chords is not enough; the curve has to be on it.

        Here C4 is in both chords and the curve runs from G4 to C4, so nothing
        ties: G4 has no stop to reach, and C4 has no start behind it.
        """
        crossing = """clef_G2 . . . . upper
timeSignature/4 . . . . .
note_4 C4 _ _ _ upper&note_4 G4 _ _ slurStart upper
note_4 C4 _ _ slurStop upper&note_4 A4 _ _ _ upper
barline . . . . ."""
        tokens = read_token_lines(crossing.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        self.assertEqual(_slurs(xml), ["start", "stop"])
        self.assertEqual(_ties(xml), [])

    def test_slur_between_different_pitches_stays_a_slur(self) -> None:
        phrase = """clef_G2 . . . . upper
timeSignature/4 . . . . .
note_4 C4 _ _ slurStart upper
note_4 E4 _ _ slurStop upper
note_2 D4 _ _ _ upper
barline . . . . ."""
        tokens = read_token_lines(phrase.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        self.assertEqual(_slurs(xml), ["start", "stop"])
        self.assertEqual(_ties(xml), [])

    def test_tie_is_recognised_across_a_barline(self) -> None:
        """Ties cross barlines constantly, which is why the pass runs per part."""
        across = """clef_G2 . . . . upper
timeSignature/4 . . . . .
note_1 C4 _ _ slurStart upper
barline . . . . .
note_1 C4 _ _ slurStop upper
barline . . . . ."""
        tokens = read_token_lines(across.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        self.assertEqual(_ties(xml), ["start", "stop"])
        self.assertEqual(_slurs(xml), [])

    def test_same_pitch_but_not_adjacent_stays_a_slur(self) -> None:
        """Same pitch is not enough: a tie joins a note to its immediate successor."""
        apart = """clef_G2 . . . . upper
timeSignature/4 . . . . .
note_4 C4 _ _ slurStart upper
note_4 E4 _ _ _ upper
note_4 C4 _ _ slurStop upper
note_4 D4 _ _ _ upper
barline . . . . ."""
        tokens = read_token_lines(apart.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        self.assertEqual(_ties(xml), [])
        self.assertEqual(_slurs(xml), ["start", "stop"])

    def test_convert_ties_leaves_a_part_without_slurs_alone(self) -> None:
        part = ET.Element("part", id="P1")
        measure = ET.SubElement(part, "measure", number="1")
        note = ET.SubElement(measure, "note")
        pitch = ET.SubElement(note, "pitch")
        ET.SubElement(pitch, "step").text = "C"
        ET.SubElement(pitch, "octave").text = "4"
        ET.SubElement(note, "duration").text = "4"
        ET.SubElement(note, "voice").text = "1"
        ET.SubElement(note, "staff").text = "1"

        convert_ties(part)
        self.assertEqual(_ties(part), [])

    def test_tie_found_while_another_slur_is_open_on_the_same_staff(self) -> None:
        """Several slurs share a staff, and so share a slur number.

        The tie here opens and closes inside a longer slur. Nothing can tell
        the two apart by number, which is why ties are detected from a note
        and its successor rather than by pairing starts with stops.
        """
        nested = """clef_G2 . . . . upper
timeSignature/4 . . . . .
note_4 E4 _ _ slurStart upper
note_4 C4 _ _ slurStart upper
note_4 C4 _ _ slurStop upper
note_4 G4 _ _ slurStop upper
barline . . . . ."""
        tokens = read_token_lines(nested.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        # the inner pair became a tie
        self.assertEqual(_ties(xml), ["start", "stop"])
        self.assertEqual(_tieds(xml), ["start", "stop"])
        # the outer slur is untouched
        self.assertEqual(_slurs(xml), ["start", "stop"])
