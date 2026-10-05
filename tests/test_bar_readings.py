"""The readings a person can choose between for a doubtful bar (homr/bar_readings.py).

eerovil/musescore-choir-plugins#269: the acceptance is the bar that started it,
Legenda system 11's bar 25. homr writes "E-flat plain quarter, triplet on C-F#"
there; the page prints "E-flat-C triplet, then F#". That reading has to be one
of the three a person is offered.
"""

import xml.etree.ElementTree as ET
from fractions import Fraction

from homr.bar_readings import (
    _can_follow,
    bar_readings,
    best_readings,
    embed_readings,
    read_readings,
    value_of,
)
from homr.doubt import find_doubts
from tests.test_doubt import DATA, _symbols_from_sidecar

#: Legenda system 11 bar 25 (bar 3 of the crop), bass, as the page prints it.
PAGE = ["note_12", "note_12", "note_12", "note_6", "note_12", "note_4", "note_6", "note_12"]
#: ...and as homr wrote it.
WRITTEN = ["note_12", "note_12", "note_12", "note_4", "note_12", "note_6", "note_6", "note_12"]


def _legenda():
    staffs = _symbols_from_sidecar(DATA / "legenda-s11.confidence.json")
    xml = ET.parse(DATA / "legenda-s11.musicxml").getroot()
    second = ET.parse(DATA / "legenda-s11.second.musicxml").getroot()
    return xml, bar_readings(xml, staffs, find_doubts(staffs, xml, second))


def test_legenda_bar_25_offers_the_page_reading():
    _, readings = _legenda()
    bass = [r for r in readings if (r["part"], r["staff"], r["bar"]) == (0, 2, 3)]
    assert bass, "the doubted bass bar has no readings"
    for entry in bass:
        offered = [r["values"] for r in entry["readings"]]
        assert len(offered) <= 3
        assert [m["value"] for m in entry["moments"]] == WRITTEN
        assert PAGE in offered
        # Every option fills the bar.
        for values in offered:
            assert (
                sum(
                    Fraction(1, int(v.split("_")[1].rstrip(".")))
                    * (Fraction(3, 2) if v.endswith(".") else 1)
                    for v in values
                )
                == 1
            )
    assert bass[0]["moments"][3]["pitches"] == [{"step": "E", "alter": -1, "octave": 3}]


def test_readings_are_ranked_best_first():
    _, readings = _legenda()
    for entry in readings:
        scores = [r["score"] for r in entry["readings"]]
        assert scores == sorted(scores, reverse=True)
        assert len(entry["readings"]) >= 2


def test_readings_travel_inside_the_musicxml():
    xml, readings = _legenda()
    embed_readings(xml, readings)
    embed_readings(xml, readings)  # replaced, not added twice
    again = ET.fromstring(ET.tostring(xml))
    assert read_readings(again) == readings
    assert len(again.findall("identification/miscellaneous/miscellaneous-field")) == 1
    assert read_readings(ET.fromstring("<score-partwise/>")) == []


def test_values_are_spelled_as_the_vocabulary_spells_them():
    assert value_of(Fraction(1, 4)) == "note_4"
    assert value_of(Fraction(1, 12)) == "note_12"
    assert value_of(Fraction(3, 8)) == "note_4."
    assert value_of(Fraction(7, 16)) == "note_4.."
    assert value_of(Fraction(1, 2), "rest") == "rest_2"
    assert value_of(Fraction(5, 16)) is None


def test_a_plain_note_never_starts_inside_a_triplet():
    beat = Fraction(1, 4)
    assert not _can_follow("note_4", Fraction(1, 12), True, beat)
    assert _can_follow("note_4", Fraction(1, 4), True, beat)
    assert _can_follow("note_12", Fraction(1, 12), True, beat)
    # A triplet run starts on a beat.
    assert not _can_follow("note_12", Fraction(1, 8), False, beat)
    assert _can_follow("note_12", Fraction(1, 2), False, beat)


def test_five_triplet_eighths_and_a_quarter_is_not_offered():
    # Six moments in a 2/4 bar; arithmetic alone also allows 5 x 1/12 + ... nonsense.
    options = [[("note_12", -0.1), ("note_8", -1.0)]] * 3 + [[("note_4", -0.1), ("note_12", -0.5)]]
    found = [values for values, _ in best_readings(options, Fraction(1, 2), top=10)]
    assert ["note_12", "note_12", "note_12", "note_4"] in found
    for values in found:
        triplets = [v for v in values if v == "note_12"]
        assert len(triplets) % 3 == 0


def test_a_voice_with_one_reading_offers_nothing():
    options = [[("note_4", -0.1)], [("note_4", -0.1)]]
    assert len(best_readings(options, Fraction(1, 2))) == 1
