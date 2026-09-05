"""Opening a fault out into its bar: the notes, and where the bar is on the page.

A fault row says a fault happened and cannot say what happened. These pin the
two halves of the answer -- the bar's contents on both sides, and cutting that
bar out of the printed picture -- and above all they pin the refusal, because a
crop of the wrong bar is a confident picture of the wrong music underneath a
finding.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from fixturecheck import bars, compare

TWO_BARS = """<?xml version="1.0"?>
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>V</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>2</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch>
            <duration>4</duration><voice>1</voice><type>half</type></note>
      <note><pitch><step>E</step><octave>4</octave></pitch>
            <duration>4</duration><voice>1</voice><type>half</type></note>
    </measure>
    <measure number="2">
      <note><pitch><step>G</step><octave>4</octave></pitch>
            <duration>8</duration><voice>1</voice><type>whole</type></note>
    </measure>
  </part>
</score-partwise>
"""


def score(tmp_path, text=TWO_BARS, name="s.musicxml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_a_bar_can_be_cut_out_to_be_drawn_on_its_own(tmp_path):
    kept = bars.one_bar(score(tmp_path), "2", tmp_path / "one.musicxml")
    root = ET.parse(kept).getroot()
    numbers = [m.get("number") for m in root.findall("part/measure")]
    assert numbers == ["2"]


def test_the_cut_bar_keeps_the_clef_and_meter_it_is_read_in(tmp_path):
    """A bar in the middle of a system declares none of them.

    Without this the two sides get engraved under whatever MuseScore assumes,
    and the point is to look at the same music spelled the same way.
    """
    kept = bars.one_bar(score(tmp_path), "2", tmp_path / "one.musicxml")
    attributes = ET.parse(kept).getroot().find("part/measure/attributes")
    assert attributes is not None
    assert attributes.findtext("clef/sign") == "G"
    assert attributes.findtext("time/beats") == "4"
    assert attributes.findtext("divisions") == "2"


def test_a_bar_that_is_not_there_is_not_invented(tmp_path):
    assert bars.one_bar(score(tmp_path), "7", tmp_path / "one.musicxml") is None


def test_bars_are_listed_in_the_order_they_are_written(tmp_path):
    assert bars.bars_in(score(tmp_path)) == ["1", "2"]


def _geo(lines, staves=1):
    return {"bar_lines": lines,
            "staves": [{"top": 0.1 * (i + 1), "bottom": 0.1 * (i + 1) + 0.05,
                        "left": 0.05, "right": 0.98} for i in range(staves)]}


def test_the_same_line_seen_on_every_staff_is_one_line():
    """Barlines are detected per staff; three staves give three of each."""
    geo = _geo([0.30, 0.3005, 0.301, 0.60, 0.6005, 0.90], staves=3)
    assert bars.boundaries_for(geo, 3) == [0.05, 0.30, 0.60, 0.90]


def test_a_system_opening_line_is_added_back():
    """homr finds the internal lines and the final one, not the opening rule.

    So four bars come back as four lines, and the staff's own left edge is where
    the first one starts.
    """
    assert bars.boundaries_for(_geo([0.3, 0.5, 0.7, 0.95]), 4) == [0.05, 0.3, 0.5, 0.7, 0.95]


def test_an_opening_line_that_was_detected_is_not_added_twice():
    assert bars.boundaries_for(_geo([0.05, 0.4, 0.95]), 2) == [0.05, 0.4, 0.95]


def test_a_count_that_matches_nothing_is_refused():
    """The numbering would be a guess, and a guessed crop is worse than none."""
    assert bars.boundaries_for(_geo([0.3, 0.9]), 4) is None
    assert bars.boundaries_for(_geo([0.2, 0.4, 0.6, 0.8, 0.9]), 2) is None
    assert bars.boundaries_for({"bar_lines": [], "staves": []}, 3) is None


def test_the_box_is_the_bar_and_the_staff_it_is_on():
    geo = _geo([0.3, 0.6, 0.95], staves=2)
    box = bars.bar_box(geo, 2, 2, 3)
    assert box is not None
    assert box["left"] < 0.3 < box["right"] and box["right"] > 0.6
    # ...and it is the second staff's band, with room for what hangs off it.
    assert box["top"] < 0.20 and box["bottom"] > 0.25


def test_a_bar_or_staff_out_of_range_is_refused():
    geo = _geo([0.3, 0.6, 0.95], staves=2)
    assert bars.bar_box(geo, 3, 1, 3) is None      # no third staff
    assert bars.bar_box(geo, 1, 4, 3) is None      # no fourth bar
    assert bars.bar_box(geo, 1, 0, 3) is None


def _mpos(count, width=3000.0, y=4650.79, sy=2637.07, page=0):
    return [{"x": 1000.0 + i * width, "y": y, "sx": width, "sy": sy, "page": page}
            for i in range(count)]


def test_musescore_says_where_a_bar_is_and_it_is_cut_out_by_that():
    """The mapping, pinned. Measured on two scores at 220 and 150 dpi, with the
    crop landing on the barlines both times; see `bars.MPOS_UNITS_PER_INCH`."""
    boxes = _mpos(3)
    box = bars.engraved_box(boxes, ["1", "2", "3"], "2", (1819, 2572), dpi=220)
    assert box is not None
    scale = 220 / bars.MPOS_UNITS_PER_INCH
    # The second bar starts at 4000 units and is 3000 wide. The crop is centred
    # on it...
    middle = (box["left"] + box["right"]) / 2 * 1819
    assert abs(middle - 5500 * scale) < 1.0
    # ...and is drawn wider than the bar, so it does not read as a fragment.
    tight = 3000 * scale + 16
    assert abs((box["right"] - box["left"]) * 1819 - tight * (1 + bars.ZOOM_OUT)) < 1.0
    assert 0.0 <= box["top"] < box["bottom"] <= 1.0


def test_a_crop_is_wider_than_the_bar_on_both_paths():
    """The printed crop and the engraved ones widen alike, or the three
    pictures are three different shots of the same music."""
    printed = bars.bar_box(_geo([0.3, 0.6, 0.95], staves=2), 1, 2, 3)
    engraved = bars.engraved_box(_mpos(3), ["1", "2", "3"], "2", (1819, 2572))
    for box in (printed, engraved):
        assert box is not None
    # Both come back wider than the bar they name; the tight width is recoverable.
    assert printed["right"] - printed["left"] > (0.6 - 0.3)
    assert bars.ZOOM_OUT == 0.30


def test_a_bar_count_that_disagrees_is_refused():
    """MuseScore's boxes and the score have to be talking about the same bars."""
    assert bars.engraved_box(_mpos(3), ["1", "2"], "2", (1819, 2572)) is None
    assert bars.engraved_box(_mpos(2), ["1", "2"], "9", (1819, 2572)) is None


def test_a_bar_on_a_later_page_is_refused():
    """The render kept is page one; a box on page two would crop the wrong paper."""
    assert bars.engraved_box(_mpos(2, page=1), ["1", "2"], "1", (1819, 2572)) is None


def test_a_box_that_falls_outside_the_picture_is_refused():
    """Which is what a wrong resolution or paper size looks like from here."""
    wide = [{"x": 1000.0, "y": 100.0, "sx": 90000.0, "sy": 2637.07, "page": 0}]
    assert bars.engraved_box(wide, ["1"], "1", (1819, 2572)) is None


def test_a_row_knows_which_bar_it_is_about():
    """The report opens the bar named here; re-parsing `where` would be a second
    definition of the same fact."""
    row = compare.Row("bar 3, staff 2, beat 1", "A", "B", "", "pitch",
                      bar="3", staff=2)
    assert (row.bar, row.staff) == ("3", 2)


def test_a_bar_is_read_back_with_its_beats_and_positions(tmp_path):
    """What a fault row cannot say: which beats, and which notes."""
    held = compare.bar_contents(score(tmp_path), "1", 1)
    assert [note["beat"] for note in held] == [0.0, 2.0]
    assert [note["name"] for note in held] == ["C4", "E4"]
    assert all(isinstance(note["position"], int) for note in held)
    assert compare.bar_contents(score(tmp_path), "9", 1) == []
