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
    note_readings,
    read_note_readings,
    read_readings,
    read_second_readings,
    second_readings,
    value_of,
)
from homr.doubt import SECOND_READING, find_doubts
from homr.transformer.vocabulary import EncodedSymbol
from tests.test_doubt import DATA, _symbols_from_sidecar

#: Legenda system 11 bar 25 (bar 3 of the crop), bass, as the page prints it.
PAGE = ["note_12", "note_12", "note_12", "note_6", "note_12", "note_4", "note_6", "note_12"]
#: ...and as homr wrote it.
WRITTEN = ["note_12", "note_12", "note_12", "note_4", "note_12", "note_6", "note_6", "note_12"]


def _legenda() -> tuple[ET.Element, list[dict]]:
    staffs = _symbols_from_sidecar(DATA / "legenda-s11.confidence.json")
    xml = ET.parse(DATA / "legenda-s11.musicxml").getroot()
    second = ET.parse(DATA / "legenda-s11.second.musicxml").getroot()
    return xml, bar_readings(xml, staffs, find_doubts(staffs, xml, second))


def test_legenda_bar_25_offers_the_page_reading() -> None:
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


def test_readings_are_ranked_best_first() -> None:
    _, readings = _legenda()
    for entry in readings:
        scores = [r["score"] for r in entry["readings"]]
        assert scores == sorted(scores, reverse=True)
        assert len(entry["readings"]) >= 2


def test_readings_travel_inside_the_musicxml() -> None:
    xml, readings = _legenda()
    embed_readings(xml, readings)
    embed_readings(xml, readings)  # replaced, not added twice
    again = ET.fromstring(ET.tostring(xml))
    assert read_readings(again) == readings
    assert len(again.findall("identification/miscellaneous/miscellaneous-field")) == 1
    assert read_readings(ET.fromstring("<score-partwise/>")) == []


def test_values_are_spelled_as_the_vocabulary_spells_them() -> None:
    assert value_of(Fraction(1, 4)) == "note_4"
    assert value_of(Fraction(1, 12)) == "note_12"
    assert value_of(Fraction(3, 8)) == "note_4."
    assert value_of(Fraction(7, 16)) == "note_4.."
    assert value_of(Fraction(1, 2), "rest") == "rest_2"
    assert value_of(Fraction(5, 16)) is None


def test_a_plain_note_never_starts_inside_a_triplet() -> None:
    beat = Fraction(1, 4)
    assert not _can_follow("note_4", Fraction(1, 12), True, beat)
    assert _can_follow("note_4", Fraction(1, 4), True, beat)
    assert _can_follow("note_12", Fraction(1, 12), True, beat)
    # A triplet run starts on a beat.
    assert not _can_follow("note_12", Fraction(1, 8), False, beat)
    assert _can_follow("note_12", Fraction(1, 2), False, beat)


def test_five_triplet_eighths_and_a_quarter_is_not_offered() -> None:
    # Six moments in a 2/4 bar; arithmetic alone also allows 5 x 1/12 + ... nonsense.
    options = [[("note_12", -0.1), ("note_8", -1.0)]] * 3 + [[("note_4", -0.1), ("note_12", -0.5)]]
    found = [values for values, _ in best_readings(options, Fraction(1, 2), top=10)]
    assert ["note_12", "note_12", "note_12", "note_4"] in found
    for values in found:
        triplets = [v for v in values if v == "note_12"]
        assert len(triplets) % 3 == 0


def test_a_voice_with_one_reading_offers_nothing() -> None:
    options = [[("note_4", -0.1)], [("note_4", -0.1)]]
    assert len(best_readings(options, Fraction(1, 2))) == 1


# ------------------------------------------------- other pitches (choir-plugins#290)

_BAR = """<score-partwise version="4.0"><part-list><score-part id="P1"><part-name/></score-part>
</part-list><part id="P1"><measure number="1"><attributes><divisions>1</divisions>
<key><fifths>1</fifths></key><time><beats>2</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>C</step><octave>5</octave></pitch><duration>1</duration><voice>1</voice>
<type>quarter</type></note>
<note><pitch><step>F</step><alter>1</alter><octave>5</octave></pitch><duration>1</duration>
<voice>1</voice><type>quarter</type></note></measure></part></score-partwise>"""


def _head(
    pitch: str, lift: str, pitches: dict[str, float], lifts: dict[str, float]
) -> EncodedSymbol:
    def ranked(values: dict[str, float]) -> dict:
        best = max(values.values())
        return {
            "probability": best,
            "alternatives": [
                {"value": v, "probability": p}
                for v, p in sorted(values.items(), key=lambda kv: -kv[1])
            ],
        }

    return EncodedSymbol(
        "note_4",
        pitch,
        lift,
        position="upper",
        confidence={
            "rhythm": {
                "probability": 0.99,
                "alternatives": [{"value": "note_4", "probability": 0.99}],
            },
            "pitch": ranked(pitches),
            "lift": ranked(lifts),
            "position": {"probability": 0.99},
        },
    )


def _pitch_doubts() -> tuple[ET.Element, list[dict]]:
    xml = ET.fromstring(_BAR)
    staffs = [
        [
            _head("C5", "_", {"C5": 0.6, "D5": 0.3, "B4": 0.05, ".": 0.01}, {"_": 0.99}),
            # In G major an unprinted F is an F sharp: that reading is the one written.
            _head("F5", "#", {"F5": 0.99}, {"#": 0.55, "N": 0.4, "_": 0.04}),
            EncodedSymbol("barline"),
        ]
    ]
    return xml, note_readings(xml, staffs, {(0, 1, 1): {"pitch unsure"}})


def test_an_unsure_pitch_offers_the_decoders_next_pitches() -> None:
    _, notes = _pitch_doubts()
    first = next(n for n in notes if n["moment"] == 0)
    assert (first["part"], first["staff"], first["bar"], first["voice"], first["chord"]) == (
        0,
        1,
        1,
        "1",
        0,
    )
    offered = [(p["step"], p["alter"], p["octave"]) for p in first["pitches"]]
    assert offered == [("C", 0, 5), ("D", 0, 5), ("B", 0, 4)]
    # As written first, with the decoder's own probability for it, so it can be ranked.
    assert first["pitches"][0]["probability"] == 0.6
    assert [m["value"] for m in first["moments"]] == ["note_4", "note_4"]


def test_an_unsure_accidental_offers_the_same_note_otherwise_altered() -> None:
    _, notes = _pitch_doubts()
    second = next(n for n in notes if n["moment"] == 1)
    # "_" in G major is the F sharp already written, so it is not offered twice.
    assert [(p["step"], p["alter"]) for p in second["pitches"]] == [("F", 1), ("F", 0)]


def test_a_sure_note_and_an_undoubted_bar_offer_nothing() -> None:
    xml = ET.fromstring(_BAR)
    sure = [
        [_head("C5", "_", {"C5": 0.99}, {"_": 0.99}), _head("F5", "#", {"F5": 0.99}, {"#": 0.99})]
    ]
    assert note_readings(xml, sure, {(0, 1, 1): {"x"}}) == []
    _, notes = _pitch_doubts()
    assert note_readings(ET.fromstring(_BAR), [[]], {}) == []
    assert notes


def test_other_pitches_travel_beside_the_readings() -> None:
    xml, notes = _pitch_doubts()
    embed_readings(xml, [], notes)
    again = ET.fromstring(ET.tostring(xml))
    assert read_note_readings(again) == notes
    assert read_readings(again) == []
    assert read_note_readings(ET.fromstring("<score-partwise/>")) == []


# ------------------------------------------ the second reading (choir-plugins#295)


def _two_voice_bar(upper: str, lower: str) -> ET.Element:
    """One 2/4 bar of two voices on one staff; each voice is "step octave value ..."."""

    def notes(voice: str, text: str) -> str:
        out = []
        for spec in text.split(","):
            step, octave, duration = spec.split()
            pitch = (
                "<rest/>"
                if step == "R"
                else f"<pitch><step>{step}</step><octave>{octave}</octave></pitch>"
            )
            out.append(f"<note>{pitch}<duration>{duration}</duration><voice>{voice}</voice></note>")
        return "".join(out)

    return ET.fromstring(
        '<score-partwise version="4.0"><part-list><score-part id="P1"><part-name/>'
        '</score-part></part-list><part id="P1"><measure number="1"><attributes>'
        "<divisions>2</divisions><time><beats>2</beats><beat-type>4</beat-type></time>"
        f"</attributes>{notes('1', upper)}<backup><duration>4</duration></backup>"
        f"{notes('2', lower)}</measure></part></score-partwise>"
    )


_DOUBTED = {(0, 1, 1): {SECOND_READING}}


def test_the_second_reading_of_a_voice_is_kept_beside_the_first() -> None:
    first = _two_voice_bar("C 5 2,D 5 2", "A 4 4")
    second = _two_voice_bar("C 5 3,E 5 1", "A 4 4")
    [entry] = second_readings(first, second, _DOUBTED)
    assert (entry["part"], entry["staff"], entry["bar"], entry["voice"]) == (0, 1, 1, "1")
    assert [m["value"] for m in entry["moments"]] == ["note_4", "note_4"]
    assert [(m["value"], m["pitches"][0]["step"]) for m in entry["second"]] == [
        ("note_4.", "C"),
        ("note_8", "E"),
    ]


def test_voices_are_paired_by_their_notes_not_their_numbers() -> None:
    first = _two_voice_bar("C 5 2,D 5 2", "A 4 4")
    # The second reading wrote the lower line first.
    second = _two_voice_bar("A 4 4", "C 5 2,E 5 2")
    [entry] = second_readings(first, second, _DOUBTED)
    assert entry["voice"] == "1"
    assert [m["pitches"][0]["step"] for m in entry["second"]] == ["C", "E"]


def test_no_second_reading_where_it_cannot_be_laid_beside_the_first() -> None:
    first = _two_voice_bar("C 5 2,D 5 2", "A 4 4")
    # Short of the bar.
    assert second_readings(first, _two_voice_bar("C 5 2,D 5 1", "A 4 4"), _DOUBTED) == []
    # A different number of voices.
    one = ET.fromstring(ET.tostring(first))
    measure = one.find("part/measure")
    for el in list(measure):
        if el.tag == "backup" or el.findtext("voice") == "2":
            measure.remove(el)
    assert second_readings(first, one, _DOUBTED) == []
    # Read the same, not doubted for it, or no second reading at all.
    assert second_readings(first, first, _DOUBTED) == []
    changed = _two_voice_bar("C 5 3,E 5 1", "A 4 4")
    assert second_readings(first, changed, {(0, 1, 1): {"pitch unsure"}}) == []
    assert second_readings(first, None, _DOUBTED) == []


def test_the_second_reading_travels_in_the_musicxml() -> None:
    first = _two_voice_bar("C 5 2,D 5 2", "A 4 4")
    again = second_readings(first, _two_voice_bar("C 5 3,E 5 1", "A 4 4"), _DOUBTED)
    embed_readings(first, [], [], again)
    parsed = ET.fromstring(ET.tostring(first))
    assert read_second_readings(parsed) == again
    assert read_second_readings(ET.fromstring("<score-partwise/>")) == []
