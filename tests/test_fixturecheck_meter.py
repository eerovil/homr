"""A misread time signature, which the check could not see at all.

`<divisions>` was read to turn durations into beats and the signature itself
never was, so homr putting the music in the wrong meter cost nothing.
`sammon-ryosto` goes 3/4, 5/4, 5/2 on the page and 7/4, nothing, 3/2 in homr's
reading, and the report said 88.4% and not one word about it.

**Judged only where the meter changes**, on the meter in force there — one rule
covering a change the page makes and homr misses, a change homr invents, and a
change both make but to different meters. A run of bars nobody changed is never
looked at, so an early mistake cannot report itself once per bar for the rest of
the system.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from fixturecheck import compare


def score(tmp_path, meters, name="s.musicxml"):
    """A one-part score whose bars declare the meters given, `None` for none."""
    bars = []
    for index, meter in enumerate(meters, start=1):
        attributes = ""
        if meter:
            beats, kind = meter.split("/")
            attributes = (f"<attributes><divisions>1</divisions>"
                          f"<time><beats>{beats}</beats>"
                          f"<beat-type>{kind}</beat-type></time></attributes>")
        bars.append(
            f'<measure number="{index}">{attributes}'
            f'<note><pitch><step>C</step><octave>4</octave></pitch>'
            f'<duration>1</duration><voice>1</voice></note></measure>')
    path = tmp_path / name
    path.write_text(
        '<?xml version="1.0"?><score-partwise version="3.1">'
        '<part-list><score-part id="P1"><part-name>V</part-name></score-part></part-list>'
        f'<part id="P1">{"".join(bars)}</part></score-partwise>')
    return path


def kinds(rows):
    return [(row.bar, row.page, row.homr) for row in rows]


def test_a_meter_change_homr_missed_is_a_fault(tmp_path):
    """`sammon-ryosto`'s bar 2: the page changes to 5/4 and homr carries on."""
    page = score(tmp_path, ["3/4", "5/4"], "page.musicxml")
    homr = score(tmp_path, ["3/4", None], "homr.musicxml")
    rows = compare.compare_meter(page, homr)
    assert kinds(rows) == [("2", "5/4", "3/4")]
    assert rows[0].kind == "meter"
    assert "did not make" in rows[0].verdict


def test_a_meter_change_homr_invented_is_a_fault(tmp_path):
    page = score(tmp_path, ["4/4", None], "page.musicxml")
    homr = score(tmp_path, ["4/4", "7/8"], "homr.musicxml")
    rows = compare.compare_meter(page, homr)
    assert kinds(rows) == [("2", "4/4", "7/8")]
    assert "does not make" in rows[0].verdict


def test_both_changing_to_different_meters_is_a_fault(tmp_path):
    """Comparing only *where* changes happen would call this agreement."""
    page = score(tmp_path, ["3/4", "5/2"], "page.musicxml")
    homr = score(tmp_path, ["3/4", "3/2"], "homr.musicxml")
    rows = compare.compare_meter(page, homr)
    assert kinds(rows) == [("2", "5/2", "3/2")]
    assert "different meter" in rows[0].verdict


def test_the_opening_meter_is_judged(tmp_path):
    """Everything after it is read in it, and bar 1 is a change from nothing."""
    page = score(tmp_path, ["3/4"], "page.musicxml")
    homr = score(tmp_path, ["7/4"], "homr.musicxml")
    assert kinds(compare.compare_meter(page, homr)) == [("1", "3/4", "7/4")]


def test_a_meter_both_sides_agree_on_is_silent(tmp_path):
    page = score(tmp_path, ["4/4", None, "3/4", None], "page.musicxml")
    homr = score(tmp_path, ["4/4", None, "3/4", None], "homr.musicxml")
    assert compare.compare_meter(page, homr) == []


def test_bars_nobody_changed_are_never_judged(tmp_path):
    """A wrong meter is reported once, where it was set — not on every bar after.

    Both sides carry their meter forward through bars 2, 3 and 4; only bar 1
    declares anything, so only bar 1 is compared. Otherwise one mistake at the
    top of a system would report itself the whole way down.
    """
    page = score(tmp_path, ["3/4", None, None, None], "page.musicxml")
    homr = score(tmp_path, ["7/4", None, None, None], "homr.musicxml")
    rows = compare.compare_meter(page, homr)
    assert [row.bar for row in rows] == ["1"]


def test_a_signature_repeated_on_every_staff_is_one_change(tmp_path):
    """A system declares its meter once per staff; that is not four changes."""
    path = tmp_path / "many.musicxml"
    part = ('<part id="P{n}"><measure number="1"><attributes><divisions>1</divisions>'
            '<time><beats>3</beats><beat-type>4</beat-type></time></attributes>'
            '<note><rest/><duration>1</duration></note></measure></part>')
    path.write_text(
        '<?xml version="1.0"?><score-partwise version="3.1"><part-list>'
        + "".join(f'<score-part id="P{n}"><part-name>V</part-name></score-part>'
                  for n in range(4))
        + "</part-list>"
        + "".join(part.format(n=n) for n in range(4))
        + "</score-partwise>")
    force, changes = compare.meter_in_force(path)
    assert changes == ["1"] and force["1"] == "3/4"


def test_a_misread_meter_stops_a_fixture_being_perfect(tmp_path):
    """Which is what makes it fail the gate.

    A parse that put the music in the wrong time signature is not a correct
    reading of the page however well its noteheads line up.
    """
    clean = compare.Result(agree=10)
    assert clean.perfect
    wrong_meter = compare.Result(agree=10, meter=1)
    assert not wrong_meter.perfect
    # ...and it stays out of the note percentage, which counts note events.
    assert wrong_meter.score == 100.0


def test_the_real_fixture_is_reported(tmp_path):
    """`sammon-ryosto` as it actually is: all three shapes in one system."""
    page = score(tmp_path, ["3/4", "5/4", "5/2"], "page.musicxml")
    homr = score(tmp_path, ["7/4", None, "3/2"], "homr.musicxml")
    rows = compare.compare_meter(page, homr)
    assert kinds(rows) == [("1", "3/4", "7/4"),
                           ("2", "5/4", "7/4"),
                           ("3", "5/2", "3/2")]
