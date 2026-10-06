"""A staff whose two voices never sound at once is one line.

eerovil/musescore-choir-plugins#274: a lone x-notehead eighth of a one-voice
staff (Vieläkö huvittaisi) came out in voice 2 because its stem was read the
wrong way round, leaving voice 1 an eighth short.
"""

import xml.etree.ElementTree as ET

from homr.music_xml_generator import TimedNoteEvent, _rejoin_one_line


def event(
    start: int, end: int, step: str = "F", octave: int = 4, shared: bool = False
) -> TimedNoteEvent:
    note = ET.fromstring(
        f"<note><pitch><step>{step}</step><octave>{octave}</octave></pitch>"
        f"<duration>{end - start}</duration><voice>1</voice></note>"
    )
    if shared:
        note.set("stem-shared", "1")
    return TimedNoteEvent(1, start, end, [note])


def rest(start: int, end: int) -> TimedNoteEvent:
    note = ET.fromstring(f"<note><rest/><duration>{end - start}</duration></note>")
    return TimedNoteEvent(1, start, end, [note])


def voices(assignments: list[tuple[int, TimedNoteEvent, int]]) -> list[int]:
    return [voice for _, _, voice in assignments]


def test_a_lone_note_on_the_line_returns_to_voice_one() -> None:
    line = [(1, event(0, 4), 1), (1, event(4, 6), 1), (1, event(6, 8), 2)]
    _rejoin_one_line(1, line)
    assert voices(line) == [1, 1, 1]


def test_a_note_below_the_line_stays_a_second_voice() -> None:
    line = [(1, event(0, 4, "D", 5), 1), (1, event(4, 8, "G", 4), 2)]
    _rejoin_one_line(1, line)
    assert voices(line) == [1, 2]


def test_voices_that_sound_together_are_left_alone() -> None:
    line = [(1, event(0, 8, "A", 4), 1), (1, event(0, 8, "F", 4), 2)]
    _rejoin_one_line(1, line)
    assert voices(line) == [1, 2]


def test_a_bar_with_a_head_shared_by_both_voices_is_left_alone() -> None:
    line = [(1, event(0, 4, shared=True), 1), (1, event(4, 8), 2)]
    _rejoin_one_line(1, line)
    assert voices(line) == [1, 2]


def test_a_voice_with_no_pitched_neighbour_is_left_alone() -> None:
    line = [(1, rest(0, 4), 1), (1, event(4, 8), 2)]
    _rejoin_one_line(1, line)
    assert voices(line) == [1, 2]


def test_another_staffs_voices_are_not_touched() -> None:
    line = [(1, event(0, 4), 1), (2, event(4, 8), 2), (1, event(4, 8), 2)]
    _rejoin_one_line(1, line)
    assert voices(line) == [1, 2, 1]
