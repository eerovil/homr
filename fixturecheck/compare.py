"""What homr wrote against what the page holds, and separately what it saw.

Two layers, kept apart on purpose. **Detection** is noteheads and stems found in
the picture; that is what the fixture tests pin. **Output** is the MusicXML homr
writes, which is what anything downstream sings from. They are not the same and
they can disagree in both directions: a note the detector misses still reaches
the output (in the wrong voice, having no stem to place it), and a head detected
a step off the page is still written at the right pitch, because the pitch is
read from the image by the transformer rather than taken from that geometry.

Three reports were filed in one day claiming homr had misread music when what
had been compared was the detector. So the two are separate functions here, they
return separate results, and the report names which is which.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

STEPS = "CDEFGAB"
CLEF_REFERENCE = {"G": ("G", 4), "F": ("F", 3), "C": ("C", 4)}


def _diatonic(step: str, octave: int) -> int:
    return 7 * octave + STEPS.index(step)


def _clefs(measure: ET.Element) -> dict[int, tuple[str, int, int]]:
    found = {}
    for attributes in measure.findall("attributes"):
        for clef in attributes.findall("clef"):
            found[int(clef.get("number", "1"))] = (
                clef.findtext("sign", "G"),
                int(clef.findtext("line", "2")),
                int(clef.findtext("clef-octave-change", "0")),
            )
    return found


def staff_position(pitch: ET.Element, clef: tuple[str, int, int]) -> int:
    """Bottom line 1, one step per line or space.

    Position and not pitch name, because a male-choir score is written an octave
    above where it sounds: the reference says A3 where homr says A4 for the same
    printed note, and comparing names would report every note in the piece.
    """
    sign, line, octave_change = clef
    reference_step, reference_octave = CLEF_REFERENCE[sign]
    written = _diatonic(pitch.findtext("step", "C"), int(pitch.findtext("octave", "4")))
    return 2 * line - 1 + written - 7 * octave_change - _diatonic(reference_step, reference_octave)


def read_score(path: Path) -> dict[tuple[str, int, float], list[dict]]:
    """Every sounding note, keyed by when and where it sounds.

    Notes are placed by walking the measure with the duration cursor -- a note
    advances it, a chord tone does not, a backup winds it back -- and not by
    document order: the reference writes one voice out and backs up for the next
    while homr interleaves them, so document order pairs a tenor with a bass. It
    did, and produced twenty confident nonsense rows.

    The printed staff is counted across the whole file rather than read from
    `<staff>`, because both sides write their staves as separate parts that each
    call themselves staff 1, and keying on that folds the bass into the treble.

    Onsets are in **quarter notes**, not in the file's own duration units. Two
    scores of the same music routinely declare different `<divisions>` -- Heraa
    Suomi's reference and its parse do -- and comparing raw durations then lines
    a note up with whichever note happens to share an offset and reports it as a
    different pitch. Every "3 positions lower" on that system was this, on notes
    whose positions in fact agree exactly.
    """
    found: dict[tuple[str, int, float], list[dict]] = {}
    printed = 0
    for part in ET.parse(path).getroot().findall("part"):
        staves = max((int(n.text or 1) for n in part.iter("staves")), default=1)
        base, printed = printed, printed + staves
        clefs: dict[int, tuple[str, int, int]] = {}
        divisions = 1.0
        for measure in part.findall("measure"):
            clefs.update(_clefs(measure))
            for attributes in measure.findall("attributes"):
                declared = attributes.findtext("divisions")
                if declared:
                    divisions = float(declared) or 1.0
            at = previous = 0.0
            for node in measure:
                if node.tag == "backup":
                    at -= float(node.findtext("duration", "0")) / divisions
                    continue
                if node.tag == "forward":
                    at += float(node.findtext("duration", "0")) / divisions
                    continue
                if node.tag != "note":
                    continue
                length = float(node.findtext("duration", "0")) / divisions
                onset = previous if node.find("chord") is not None else at
                if node.find("chord") is None:
                    previous, at = at, at + length
                if node.find("rest") is not None:
                    continue
                pitch = node.find("pitch")
                if pitch is None:
                    continue
                inner = int(node.findtext("staff", "1"))
                found.setdefault((measure.get("number", "?"), base + inner, round(onset, 3)),
                                 []).append({
                    "voice": node.findtext("voice", "1"),
                    "name": f"{pitch.findtext('step')}{pitch.findtext('octave')}",
                    "position": staff_position(pitch, clefs.get(inner, ("G", 2, 0))),
                    "stem": node.findtext("stem", ""),
                    "chord": node.find("chord") is not None,
                })
    return found


def collapse_unisons(found: dict) -> dict:
    """Two voices in unison print ONE notehead, so count it once.

    Older choral engraving writes a unison as a single head serving both voices,
    and a reference with one staff per part has to write it twice. Counting those
    two against homr's one reported a lost note on ten moments of Sammon ryosto,
    every one of them a unison and not one of them wrong.
    """
    merged: dict = {}
    for key, group in found.items():
        seen: dict[int, dict] = {}
        for note in group:
            at = seen.get(note["position"])
            if at is None:
                seen[note["position"]] = dict(note, unison=False)
            else:
                at["unison"] = True
        merged[key] = list(seen.values())
    return merged


def _voice_rank(found: dict) -> dict[int, dict[str, int]]:
    """The staff's voices as first, second, ... counted from the top line down.

    Compared as a rank and not as a number: the two files number voices
    differently and neither numbering means anything on its own.
    """
    order: dict[int, list[str]] = {}
    for (_, staff, _), group in sorted(found.items()):
        for note in sorted(group, key=lambda n: -n["position"]):
            order.setdefault(staff, [])
            if note["voice"] not in order[staff]:
                order[staff].append(note["voice"])
    return {staff: {voice: rank + 1 for rank, voice in enumerate(voices)}
            for staff, voices in order.items()}


@dataclass
class Row:
    where: str
    page: str
    homr: str
    verdict: str
    kind: str          # agree | voice | pitch | size | timing | structure | unison
    #: Which bar and staff this row is about, as data rather than as prose in
    #: `where`. The report opens a fault out into that bar in full, on both
    #: sides, and re-parsing our own sentence to find out which bar to show
    #: would be a second definition of the same fact.
    bar: str = ""
    staff: int = 0


@dataclass
class Result:
    agree: int = 0
    voice: int = 0
    pitch: int = 0
    size: int = 0
    timing: int = 0
    unison: int = 0
    staves_page: int = 0
    staves_homr: int = 0
    staves_printed: int = 0
    #: Bars where the two disagree about the meter, judged only where one side
    #: or the other changes it. A wrong answer about the bar rather than about
    #: a notehead, so it is counted like `structure` and not folded into the
    #: note percentage -- see `compare_meter`.
    meter: int = 0
    rows: list[Row] = field(default_factory=list)

    @property
    def judged(self) -> int:
        """The notes matched one against one: right, wrong voice, wrong pitch."""
        return self.agree + self.voice + self.pitch

    @property
    def scored(self) -> int:
        """Everything the score is taken over — the matched notes and the misses.

        `size` and `timing` are in here and were not before, and the difference
        is not cosmetic. The old denominator was `judged` alone, which asks only
        *of the notes homr wrote, how many are right* — so a system whose notes
        were half missing, or whose every beat had walked out of step, could
        score 100%. Both are wrong parses and a singer meets both of them.

        The consequence is that no percentage from this file is comparable with
        one quoted before 2026-09-05. That is the intended trade.
        """
        return self.judged + self.size + self.timing

    @property
    def score(self) -> float:
        return 100.0 * self.agree / self.scored if self.scored else 0.0

    @property
    def warnings(self) -> int:
        """Moments the parse holds a voice short of what the page means.

        A unison is printed as one notehead serving two parts, so homr writes
        one note and the second part is simply not in the file. Nothing about
        the reading is wrong -- the head is where the page puts it, at the pitch
        the page prints -- which is why this is not a fault and stays out of the
        note percentage.

        It is not nothing either. A choir sings the second part, and everything
        downstream reads the parts out of the file, so a system full of unisons
        is a system whose lower voice does not exist as far as a practice track
        is concerned. Counted, said, and counted against `perfect`, which is
        what makes it visible rather than a green row nobody reads.
        """
        return self.unison

    @property
    def perfect(self) -> bool:
        """Nothing wrong at all: every note right, no argument about staves, no warning.

        What the gate on the committed fixtures is written in terms of. A case
        with nothing to judge is not perfect — it is unread, and saying
        otherwise would let an empty parse pass the gate.

        A warning costs perfection although it is not a fault, because the five
        committed fixtures are held to "nothing to look at here" rather than to
        "nothing measurably wrong". Note that this is a claim about the
        *fixtures*, not about the repertoire: unisons are ordinary in it, and the
        song systems are not gated on anything.
        """
        return (bool(self.scored) and self.agree == self.scored
                and not self.structure and not self.meter and not self.warnings)

    @property
    def structure(self) -> int:
        """1 where the reference and homr disagree about how many staves there are.

        A case and not a note, because that is what it is: one wrong answer,
        which then costs a row at every moment of every staff below it.
        """
        return int(self.staves_page != self.staves_homr)

    @property
    def at_fault(self) -> str:
        """Which side the printed page says is wrong, where anyone has looked.

        `""` when the two agree or nobody has recorded what the page prints --
        which is most cases, and is not the same as "nobody is wrong".
        """
        if not self.structure or not self.staves_printed:
            return ""
        if self.staves_homr == self.staves_printed:
            return "reference"
        if self.staves_page == self.staves_printed:
            return "homr"
        return "both"

    @property
    def faults(self) -> int:
        return self.voice + self.pitch + self.size


PRINTED = Path(__file__).resolve().parent / "printed.json"


def staves_a_person_counted(case: str, path: Path = PRINTED) -> int:
    """How many staves the page prints here, if anyone has said so.

    0 when nobody has, which is most cases. The check has no third party to ask
    when the reference and a parse disagree about the staves, and on this
    repertoire it has been the reference's fault as often as homr's -- so the
    answer is worth writing down the once. See `printed.json`.
    """
    if not path.exists():
        return 0
    try:
        recorded = json.loads(path.read_text()).get("systems", {})
    except ValueError:
        return 0
    return int(recorded.get(case, {}).get("staves", 0))


def printed_staves(path: Path) -> int:
    """How many printed staves the file holds, counted as `read_score` keys them.

    Off the file rather than off the notes, so a staff that rests through the
    whole system still counts: it is printed, and homr inventing one is the
    same mistake whether or not anything sounds on it.
    """
    return sum(max((int(n.text or 1) for n in part.iter("staves")), default=1)
               for part in ET.parse(path).getroot().findall("part"))


def _heads_per_bar(found: dict) -> dict[tuple[str, int], int]:
    """How many noteheads each staff holds in each bar, whenever they sound."""
    heads: dict[tuple[str, int], int] = {}
    for (bar, staff, _), group in found.items():
        heads[(bar, staff)] = heads.get((bar, staff), 0) + len(group)
    return heads


def _lines_per_bar(found: dict) -> dict[tuple[str, int], int]:
    """How many voices the page prints on each staff in each bar."""
    lines: dict[tuple[str, int], set[str]] = {}
    for (bar, staff, _), group in found.items():
        lines.setdefault((bar, staff), set()).update(note["voice"] for note in group)
    return {key: len(voices) for key, voices in lines.items()}


def compare_output(reference: Path, parsed: Path, case: str = "") -> Result:
    """The page against homr's MusicXML, note by note.

    `case` is the system's name, and it buys one thing: where somebody has
    recorded how many staves the page actually prints there (`printed.json`),
    a structural disagreement is **adjudicated** rather than merely reported.
    Without it this function has no third party to ask, and on this repertoire
    the reference has been wrong as often as homr -- three systems in a row
    were written up as homr losing a staff when the page agreed with homr every
    time.

    A moment where the two files hold a different number of noteheads used to be
    one number, `size`, and across the 93 systems of this repertoire it came to
    505 -- read, reasonably, as five hundred misread notes. Checked one at a
    time, almost none of them was a note homr had got wrong.

    **Nineteen of the 93 systems disagree about how many staves are printed.**
    The staff is part of the key every note is matched on, so from the first
    staff that diverges the two files are no longer being compared on the same
    music: 328 of the 505 live in those nineteen, and so do 54 of the 128 wrong
    pitches. That is one wrong answer, about structure, so it is counted once as
    `structure` and said at the top of the case page, and the rows beneath it
    are to be read as its consequence.

    **Which side is wrong is not assumed, and on this repertoire it is ours.**
    The first version of this said homr had invented a staff. Three of the
    nineteen were then looked at against the printed page, and homr had the
    staff count right in all three: `kaksi-laulua-krapulasta-2-s13` prints a
    tenor staff and two separate bass staves, `kayttaytymisohjeita-s5` prints
    two tenor staves and one bass, and `laulun-aika-3-s7` prints four, each
    labelled T3, T1, T2, B with a lyric line of its own. homr wrote 3, 3 and 4.
    The reference said 2 every time.

    All nineteen are in four songs, three of them per-system scores, and the
    reference is built by imploding the cleaned score with **one grouping for
    the whole song** -- `implode._from_system_map` unions the per-system map, so
    two voices that share a printed staff in *any* system are merged in *every*
    system. On a score whose staves regroup from system to system that is wrong
    wherever they are printed apart. So `kaksi-laulua-krapulasta-2-s13` reports
    26 lost noteheads on a system homr read *note for note correctly*: its 78
    notes are all there and all right, and it is the reference that put them on
    two staves instead of three.

    This is the failure the two layers here exist to prevent, arriving from the
    side nobody was watching -- not a detection reported as an output, but our
    own reference reported as the page. The count is kept, because a
    disagreement about structure is worth one loud line either way; the wording
    names the disagreement and not a culprit.

    **Of the 177 left in the 74 systems whose staves do agree, 118 are the
    bar's own rhythm.** homr wrote nothing at that beat while that bar of that
    staff still holds at least as many noteheads as the page prints: a duration
    read differently early in the bar walks the cursor out of step, and every
    moment after it reports one against zero. The notes are in the file, at the
    wrong time, which is a real fault and a different one, so it is `timing`.

    Asking merely whether the bar held *anything* is not enough, and the first
    version of this asked exactly that. homr writes **one** notehead into bar 2
    of `sammon-ryosto` where the page prints four, and being asked the looser
    question the check called three lost notes a beat read differently -- the
    tidy answer, on the one case somebody had already looked at by hand and
    found notes genuinely missing. So the bar's heads are counted on both sides,
    and a bar that is short of them has lost notes rather than moved them.

    That the remaining 118 really are only moved is checked rather than assumed:
    in 102 of them the bar holds **exactly the same staff positions** on both
    sides, the same notes at different beats. The other 16 are bars where homr
    has a head to spare and one of them sits a position out.

    What is left in `size` is the moment both files agree exists and disagree
    about: two heads printed and one written, or one printed and two written.
    **That is 59 across the 74 comparable systems, not 505**, and it is the
    number worth chasing.
    """
    want = collapse_unisons(read_score(reference))
    got = read_score(parsed)
    here, there = _voice_rank(want), _voice_rank(got)
    # A voice is only wrong where the page gave it a choice. Where a staff prints
    # one line through a bar, homr numbering its notes voice 5 and then voice 6
    # is untidy and costs the singer nothing -- there is no second line for the
    # note to be on. Counting it anyway made five faults out of one such bar and
    # put the worst case in the sample somewhere it did not belong.
    lines = _lines_per_bar(want)
    # How many noteheads each bar of each staff holds, on both sides. A moment
    # homr left empty is only a beat read differently if the notes are still in
    # that bar somewhere; if the bar is short of heads they are not shifted,
    # they are gone. Asking merely whether the bar had *anything* in it is not
    # enough, and says so on real music: homr writes one notehead into bar 2 of
    # `sammon-ryosto` where the page prints four, and the looser question called
    # that a beat read differently.
    heads_page = _heads_per_bar(want)
    heads_homr = _heads_per_bar(got)
    result = Result(staves_page=printed_staves(reference), staves_homr=printed_staves(parsed),
                    staves_printed=staves_a_person_counted(case) if case else 0)
    # The meter first: it is what the bars below are read in, so a disagreement
    # about it belongs above them rather than buried among the notes.
    for row in compare_meter(reference, parsed):
        result.meter += 1
        result.rows.append(row)
    if result.structure:
        blame = {
            "reference": "the page agrees with homr — the REFERENCE is wrong here, "
                         "and the rows below are comparing it against music it "
                         "does not describe",
            "homr": "the page agrees with the reference — HOMR is wrong here",
            "both": "the page agrees with neither",
            "": "which side is wrong is decided by looking at the page above, not "
                "by this row — record it in fixturecheck/printed.json and it will "
                "be decided here next time",
        }[result.at_fault]
        printed = (f", page prints {result.staves_printed}"
                   if result.staves_printed else "")
        result.rows.append(Row(
            "the system", f"{result.staves_page} staves (the reference){printed}",
            f"{result.staves_homr} staves",
            f"a different number of staves — every note below is keyed on the "
            f"staff, so from the first one that diverges nothing lines up. {blame}",
            "structure"))
    for key in sorted(want, key=lambda k: (int(re.sub(r"\D", "", k[0]) or 0), k[1], k[2])):
        bar, staff, onset = key
        where = f"bar {bar}, staff {staff}, beat {onset:g}"

        def emit(page: str, homr: str, verdict: str, kind: str,
                 _bar: str = bar, _staff: int = staff, _where: str = where) -> None:
            """One row, stamped with the bar and staff it is about."""
            result.rows.append(Row(_where, page, homr, verdict, kind,
                                   bar=_bar, staff=_staff))

        mine = sorted(want[key], key=lambda n: -n["position"])
        theirs = sorted(got.get(key, []), key=lambda n: -n["position"])
        if len(mine) != len(theirs):
            if not theirs and heads_homr.get((bar, staff), 0) >= heads_page.get((bar, staff), 0):
                result.timing += 1
                emit(f"{len(mine)} notehead(s)", "nothing at this beat",
                     "homr has this bar's notes at other beats — a duration read "
                     "differently, not a note lost", "timing")
            else:
                result.size += 1
                emit(f"{len(mine)} notehead(s)", f"{len(theirs)} notehead(s)",
                     "a different number of notes here", "size")
            continue
        for a, b in zip(mine, theirs):
            if a["position"] != b["position"]:
                result.pitch += 1
                gap = abs(a["position"] - b["position"])
                emit(f"{a['name']} · position {a['position']}",
                     f"{b['name']} · position {b['position']}",
                     f"a different note, {gap} position(s) "
                     f"{'higher' if b['position'] > a['position'] else 'lower'}", "pitch")
                continue
            if a.get("unison"):
                result.unison += 1
                emit(f"{a['name']} · both voices", f"{b['name']} · one voice",
                     "a unison — the page means both parts, the parse holds one",
                     "unison")
                continue
            mine_rank, their_rank = here[staff][a["voice"]], there[staff][b["voice"]]
            if lines.get((bar, staff), 1) < 2:
                result.agree += 1
                emit(f"{a['name']} · the only line", b["name"],
                     "one line here — no voice to get wrong", "agree")
                continue
            if mine_rank != their_rank:
                result.voice += 1
                emit(f"{a['name']} · voice {mine_rank}",
                     f"{b['name']} · voice {their_rank}"
                     + ("" if b["stem"] else " (no stem)"),
                     "folded into the other voice as a chord member" if b["chord"]
                     else "written in the other voice", "voice")
                continue
            result.agree += 1
            emit(f"{a['name']} · voice {mine_rank}", f"{b['name']} · voice {their_rank}",
                 "the voice the page prints", "agree")
    return result


def meter_in_force(path: Path) -> tuple[dict, list]:
    """The meter each bar is read in, and the bars where it changes.

    A signature is declared on every staff of a system, so the same change
    arrives several times; what matters is the meter *in force*, which is the
    last one declared. A bar is a **change** when what it declares differs from
    what was already running — including the first bar, which changes from
    nothing.
    """
    force: dict = {}
    changes: list = []
    order: list = []
    running = ""
    seen = set()
    for part in ET.parse(path).getroot().findall("part"):
        for measure in part.findall("measure"):
            number = measure.get("number", "")
            if number not in seen:
                seen.add(number)
                order.append(number)
    for number in order:
        declared = ""
        for part in ET.parse(path).getroot().findall("part"):
            for measure in part.findall("measure"):
                if measure.get("number") != number:
                    continue
                for attributes in measure.findall("attributes"):
                    for time in attributes.findall("time"):
                        beats = time.findtext("beats", "")
                        kind = time.findtext("beat-type", "")
                        if beats and kind:
                            declared = f"{beats}/{kind}"
        if declared and declared != running:
            running = declared
            changes.append(number)
        force[number] = running
    return force, changes


def compare_meter(reference: Path, parsed: Path) -> list[Row]:
    """Where the two disagree about the meter — judged only where it changes.

    The check read `<divisions>` to turn durations into beats and never looked
    at the signature itself, so a misread meter was invisible: `sammon-ryosto`
    goes 3/4, 5/4, 5/2 on the page and 7/4, nothing, 3/2 in homr's reading, and
    the report said 88.4% and nothing else.

    **Only bars where one side or the other declares a change are judged**, and
    they are judged on the meter in force there. That one rule covers the three
    shapes this can take — a change the page makes and homr misses, a change
    homr invents, and a change both make but to different meters — while a run
    of bars nobody changed is never looked at, so an early mistake cannot report
    itself once per bar for the rest of the system.

    A **missing** signature is the same error as a wrong one. Whether homr wrote
    7/4 or wrote nothing and carried 7/4 forward, the bar is in the wrong meter
    and a singer meets the same problem.
    """
    want, want_changes = meter_in_force(reference)
    got, got_changes = meter_in_force(parsed)
    rows = []
    for bar in sorted(set(want_changes) | set(got_changes),
                      key=lambda b: (len(b), b)):
        here, there = want.get(bar, ""), got.get(bar, "")
        if here == there:
            continue
        rows.append(Row(
            f"bar {bar}", here or "no meter", there or "no meter",
            "a change homr did not make" if bar in want_changes
            and bar not in got_changes else
            ("a change the page does not make" if bar in got_changes
             and bar not in want_changes else "changed to a different meter"),
            "meter", bar=bar, staff=0))
    return rows


def bar_contents(path: Path, bar: str, staff: int) -> list[dict]:
    """Everything one bar of one staff holds, in the order it sounds.

    What a fault row cannot say on its own. "homr has this bar's notes at other
    beats" is true and useless: *which* beats, and which of the two noteheads
    did it write? Both sides of the bar, laid out, answer that without opening
    the score — which is the whole point of there being a report.
    """
    found = []
    for (this_bar, this_staff, onset), group in sorted(read_score(path).items()):
        if this_bar != bar or this_staff != staff:
            continue
        for note in sorted(group, key=lambda n: -n["position"]):
            found.append({"beat": onset, "name": note["name"],
                          "position": note["position"], "voice": note["voice"],
                          "chord": note.get("chord", False),
                          "stem": note.get("stem", True)})
    return found
