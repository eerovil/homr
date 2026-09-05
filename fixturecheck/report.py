"""The report, which is the output rather than a thing you ask for afterwards.

Every run writes it. There is no flag for "just the numbers": a count says a
system agrees on staves, bars and noteheads and cannot say whether the parse is
the music, and asking for the pictures separately is how a whole afternoon went
on hand-built pages.

Each case page opens with the same three images in the same order -- the printed
band, homr's output engraved, the reference engraved -- so two cases can be read
against each other without working out what is being shown.
"""

from __future__ import annotations

import html
import os
import shutil
import subprocess
from pathlib import Path

from fixturecheck import bars, cases, compare
from fixturecheck.compare import Result

#: Where the report is written, and **outside any checkout by default**.
#:
#: It used to be `check-report/` beside the source, which meant every worktree
#: had its own and none of them was at an address. A report you can only reach
#: by ssh-ing to the host and knowing which branch produced it is most of what
#: made "how good is it now" unanswerable. One fixed path instead: every run
#: from every checkout lands here, so there is one thing to serve and it is
#: never stale for the reason that somebody ran the harness somewhere else.
#:
#: The page names the homr it measured, so a run from a branch overwriting a run
#: from `main` is legible rather than confusing -- and the series, not this, is
#: what accumulates.
OUT = Path(os.environ.get("FIXTURECHECK_REPORT")
           or Path.home() / ".local/share/homr-fixturecheck/report")

#: Where that directory is reachable from, when somebody has served it. Only
#: used to print a link and to put one in QUALITY.md: nothing here serves
#: anything, because a static folder on a tailnet is one `tailscale serve`
#: command and not a service to keep alive. See fixturecheck/README.md.
URL = os.environ.get("FIXTURECHECK_REPORT_URL", "")

STYLE = """
body { font: 14px/1.55 system-ui, sans-serif; margin: 0 auto; max-width: 1150px;
       padding: 24px; color: #1a1a1a; background: #fafafa; }
h1 { font-size: 22px; margin-bottom: 2px; }
h2 { font-size: 17px; margin: 30px 0 6px; }
h3 { font-size: 13px; margin: 18px 0 2px; color: #444; text-transform: uppercase;
     letter-spacing: .06em; }
p.lead { color: #555; margin-top: 0; }
a { color: #1b3a7a; }
img { display: block; width: 100%; border: 1px solid #e2e2e2; border-radius: 6px;
      background: #fff; margin: 4px 0 12px; }
table { border-collapse: collapse; width: 100%; background: #fff;
        border: 1px solid #e2e2e2; border-radius: 6px; overflow: hidden;
        margin-bottom: 10px; }
th, td { padding: 5px 9px; text-align: left; border-bottom: 1px solid #f0f0f0;
         font-variant-numeric: tabular-nums; }
th { background: #f4f4f4; font-size: 12px; text-transform: uppercase;
     letter-spacing: .04em; color: #555; }
tr.voice td { background: #fdecec; }
tr.pitch td { background: #fde8d8; }
tr.size  td { background: #fff8e1; }
tr.timing td { background: #eef4fb; }
tr.structure td { background: #f3e8fb; font-weight: 600; }
tr.unison td { background: #f4f4f7; }
td.ok { color: #1c5c2c; } td.no { color: #8a1f1f; font-weight: 600; }
.sum { display: flex; gap: 22px; flex-wrap: wrap; margin: 8px 0 16px; }
.sum b { font-size: 20px; display: block; }
.up { color: #1c5c2c; font-weight: 600; } .down { color: #8a1f1f; font-weight: 600; }
.same { color: #888; }
p.warn { background: #f3e8fb; border: 1px solid #ddc7ee; border-radius: 6px;
         padding: 9px 12px; margin: 0 0 14px; }
p.prov { color: #555; margin: 0 0 10px; font-variant-numeric: tabular-nums; }
p.pass { background: #eaf6ec; border: 1px solid #c3e2c9; border-radius: 6px;
         padding: 9px 12px; margin: 0 0 14px; color: #1c5c2c; }
p.fail { background: #fdecec; border: 1px solid #f0c2c2; border-radius: 6px;
         padding: 9px 12px; margin: 0 0 14px; color: #8a1f1f; }
tr.detail td { background: #fcfcfc; padding: 0 9px 10px; }
tr.detail summary { cursor: pointer; color: #1b3a7a; font-size: 12px;
                    padding: 6px 0; }
.sides { display: flex; gap: 18px; flex-wrap: wrap; }
.sides > div { flex: 1 1 260px; min-width: 0; }
.sides h4 { margin: 6px 0 4px; font-size: 12px; text-transform: uppercase;
            letter-spacing: .05em; color: #666; }
table.bar { margin: 0 0 8px; }
table.bar th, table.bar td { padding: 3px 7px; font-size: 12px; }
.crops { display: flex; gap: 18px; flex-wrap: wrap; margin-top: 6px; }
.crops > div { flex: 1 1 260px; min-width: 0; }
.crops img { margin: 2px 0 6px; }
"""


def judged_over(total: dict) -> int:
    """Everything the headline is taken over — matched notes, losses and shifts."""
    return sum(total.get(k, 0) for k in ("agree", "voice", "pitch", "size", "timing"))


def _provenance(run: dict | None) -> str:
    """Which homr, and which references. Without both, no number here is readable."""
    if not run:
        return ""
    drifted = run.get("reference_drift", {}).get("changed", [])
    warn = ""
    if drifted:
        warn = (f" &mdash; <b>{len(drifted)} reference(s) have moved since they were "
                f"frozen</b>, so this run is not measured against the manifest: "
                f"{html.escape(', '.join(drifted))}")
    tally = run.get("outcomes", {})
    lost = tally.get("unreadable", 0) + tally.get("unbuildable", 0)
    missed = (f" {lost} case(s) could not be read and are recorded as such rather "
              f"than skipped." if lost else "")
    return (f"<p class='prov'>homr <b>{html.escape(str(run.get('homr', '?')))}</b>, "
            f"references <b>{html.escape(str(run.get('references', '?')))}</b>, "
            f"{html.escape(str(run.get('at', '')))}{warn}.{missed}</p>")


def _gate(run: dict | None) -> str:
    if not run or "gate" not in run or not run["gate"].get("fixtures"):
        return ""
    gate = run["gate"]
    if gate.get("passed"):
        return (f"<p class='pass'>Gate <b>passed</b>: all {gate['fixtures']} committed "
                f"fixture(s) in this run are perfect.</p>")
    failing = html.escape(", ".join(gate.get("failing", [])))
    return (f"<p class='fail'>Gate <b>FAILED</b>: {gate['perfect']}/{gate['fixtures']} "
            f"committed fixture(s) perfect. Below 100%: <b>{failing}</b>. These are "
            f"small systems this repository owns outright and are expected to be "
            f"exactly right.</p>")


#: Said on the page as well as in QUALITY.md, because the table above it invites
#: precisely this conclusion and the conclusion is wrong.
NOT_MEASURED = """
<h2>What this does not measure</h2>
<p class="lead"><b>Whether the choir gets a correct practice track.</b> This
compares homr's output against a reference for the same printed system &mdash;
one stage before the score anybody sings from, and several before a video.
Everything <code>clean_score</code> does afterwards is unmeasured, and so is
every repair a person made by hand.</p>
<p class="lead"><b>Detection.</b> The noteheads and stems found in the picture
are a different layer from the MusicXML homr writes and disagree with it in both
directions. There is no ground truth for detection here, so there is no number
for it.</p>
"""


def _engrave(source: Path, into: Path, dpi: int = 220) -> str:
    """One score as a picture, trimmed to the music rather than an empty A4."""
    cli = (os.environ.get("MUSESCORE_CLI_PATH") or "musescore3").strip().strip('"')
    run = subprocess.run([cli, "-T", "10", "-r", str(dpi), str(source), "-o", str(into)],
                         capture_output=True, text=True, timeout=600)
    numbered = into.with_name(f"{into.stem}-1.png")
    if numbered.exists():
        numbered.replace(into)
    return into.name if run.returncode == 0 and into.exists() else ""


#: Rows that get opened out into the whole bar. A row that agrees needs no
#: explaining, and a structural row is about the system rather than a bar.
FAULTS = ("voice", "pitch", "size", "timing")


def _beats(notes: list[dict]) -> str:
    """One side of a bar, as the notes it holds in the order they sound."""
    if not notes:
        return "<p class='lead'>nothing in this bar</p>"
    cells = "".join(
        f"<tr><td>{note['beat']:g}</td><td>{html.escape(str(note['name']))}</td>"
        f"<td>{note['position']}</td>"
        f"<td>{html.escape(str(note['voice']))}"
        f"{' · chord' if note.get('chord') else ''}"
        f"{'' if note.get('stem', True) else ' · no stem'}</td></tr>"
        for note in notes)
    return ("<table class='bar'><tr><th>beat</th><th>note</th><th>position</th>"
            f"<th>voice</th></tr>{cells}</table>")


def _bar_pictures(case, parsed: Path, row) -> str:
    """The same bar three ways: the printed ink, homr's reading, the reference.

    The tables above say what each side holds. Only the page says which of them
    is right, and on this repertoire the reference has been the wrong one every
    time anybody checked -- so the printed crop is the picture that matters and
    the other two are what it is being read against.

    Every piece is optional and says so when it is missing. A missing MuseScore
    costs the engravings, a bar the detection cannot place costs the crop, and
    neither costs the finding.
    """
    cli = (os.environ.get("MUSESCORE_CLI_PATH") or "musescore3").strip().strip('"')
    stem = f"{case.name}-b{row.bar}-s{row.staff}"
    parts = []

    printed, why = _printed_crop(case, row, stem)
    if printed:
        parts.append(f"<div><h4>the printed bar &mdash; staff {row.staff}</h4><img src='{printed}' "
                     f"alt='bar {html.escape(row.bar)} as printed'></div>")
    else:
        parts.append(f"<div><h4>the printed bar &mdash; staff {row.staff}</h4><p class='lead'>{html.escape(why)}"
                     f"</p></div>")

    for label, source, suffix in (("homr, engraved &mdash; the whole system", parsed, "homr"),
                                  ("the reference, engraved &mdash; the whole system", case.reference, "ref")):
        cut, why = _engraved_crop(case, source, row, suffix, stem, cli)
        if cut:
            parts.append(f"<div><h4>{label}</h4><img src='{cut}' alt='{label}'></div>")
        else:
            parts.append(f"<div><h4>{label}</h4><p class='lead'>"
                         f"{html.escape(why)}</p></div>")
    return f"<div class='crops'>{''.join(parts)}</div>"


#: One render of a score, kept for as long as the report is being written. Each
#: side is drawn once per case rather than once per fault -- the bar is cut out
#: of the picture, so the picture is the thing worth keeping.
_ENGRAVED: dict = {}


def _engraved_crop(case, source: Path, row, suffix: str, stem: str,
                   cli: str) -> tuple[str, str]:
    """One bar cut out of the score's own engraving, or why it is not shown.

    Cut rather than drawn again. Engraving a single bar on its own gave it a
    title, a fresh layout, and a clef and key it does not carry in context, so
    the detail under a fault looked like different music from the system at the
    top of the page. MuseScore will say where each bar landed (`.mpos`), which
    makes cutting exact and needs no detection at all.
    """
    key = (case.name, suffix)
    if key not in _ENGRAVED:
        _ENGRAVED[key] = bars.engraved(source, cli,
                                       OUT / f"{case.name}-{suffix}-page.png")
    drawn = _ENGRAVED[key]
    if not drawn:
        return "", "MuseScore could not draw this score, so there is no bar to cut out."
    page, boxes = drawn
    numbers = bars.bars_in(source)
    try:
        from PIL import Image
        with Image.open(page) as picture:
            size = picture.size
    except Exception:                                        # noqa: BLE001
        return "", "the engraving could not be read."
    box = bars.engraved_box(boxes, numbers, row.bar, size)
    if not box:
        return "", (f"MuseScore reported {len(boxes)} bar position(s) for "
                    f"{len(numbers)} bar(s) here, so which box is this bar is a "
                    f"guess.")
    cut = bars.crop(page, box, OUT / f"{stem}-{suffix}.png")
    return (cut.name, "") if cut else ("", "the crop could not be written.")


def _printed_crop(case, row, stem: str) -> tuple[str, str]:
    """The bar cut out of the printed band, or why it is not shown."""
    geo = bars.geometry(case.image, cases.CACHE / "geometry" / f"{case.name}.json")
    if not geo:
        return "", ("homr's own staff detection could not measure this picture, "
                    "so there is nothing to cut the bar out by.")
    numbers = bars.bars_in(case.reference)
    if row.bar not in numbers:
        return "", "the reference does not hold a bar of this number."
    box = bars.bar_box(geo, row.staff, numbers.index(row.bar) + 1, len(numbers))
    if not box:
        found = len(bars._boundaries(geo.get("bar_lines", [])))
        return "", (f"homr's detection found {found} barline(s) here, which cuts "
                    f"this system into neither {len(numbers)} bars nor "
                    f"{len(numbers)} plus its opening line — so which bar is "
                    f"which is a guess, and a crop of the wrong bar is worse "
                    f"than none.")
    cut = bars.crop(case.image, box, OUT / f"{stem}-page.png")
    if not cut:
        return "", "the crop could not be written."
    return cut.name, ""


def _bar_detail_row(case, parsed: Path, row) -> str:
    """The whole bar, both sides, folded away under a fault.

    A fault row says a fault happened; it cannot say what happened. "homr has
    this bar's notes at other beats" is exactly true and tells you nothing about
    *which* beats, and "2 noteheads against 1" does not say which one survived.
    Deciding whether a reading is homr's mistake or our reference's needs the
    bar, and needing the bar meant opening the score -- which is the work the
    report exists to save.

    Collapsed, so nothing moves for the rows that agree, and only on faults.
    """
    if row.kind not in FAULTS or not row.bar:
        return ""
    try:
        page = compare.bar_contents(case.reference, row.bar, row.staff)
        homr = compare.bar_contents(parsed, row.bar, row.staff)
    except Exception:                                        # noqa: BLE001
        # A detail that cannot be built must cost the detail and not the report:
        # the row above it is the finding, and this is an explanation of it.
        return ""
    pictures = _bar_pictures(case, parsed, row)
    return (
        "<tr class='detail'><td colspan='4'>"
        f"<details><summary>bar {html.escape(row.bar)}, staff {row.staff} in full"
        "</summary><div class='sides'>"
        f"<div><h4>the page</h4>{_beats(page)}</div>"
        f"<div><h4>homr</h4>{_beats(homr)}</div>"
        f"</div>{pictures}</details></td></tr>")


def case_page(case, parsed: Path, result: Result, before: dict | None) -> str:
    """Write one case's page and return its filename."""
    OUT.mkdir(parents=True, exist_ok=True)
    page = OUT / f"{case.name}-page.png"
    shutil.copy(case.image, page)
    output = _engrave(parsed, OUT / f"{case.name}-homr.png")
    reference = _engrave(case.reference, OUT / f"{case.name}-ref.png")

    def moved(field: str) -> str:
        if not before or field not in before:
            return ""
        was = before[field]
        now = getattr(result, field)
        if was == now:
            return "<span class='same'>no change</span>"
        better = now < was if field != "agree" else now > was
        return (f"<span class='{'up' if better else 'down'}'>"
                f"{now - was:+d} since the last run</span>")

    structure = ""
    if result.structure:
        verdict = {
            "reference": (f"<b>The page prints {result.staves_printed}, so the "
                          f"reference is the wrong one here</b> &mdash; the rows below "
                          f"compare it against music it does not describe."),
            "homr": (f"<b>The page prints {result.staves_printed}, so homr is the "
                     f"wrong one here.</b>"),
            "both": (f"<b>The page prints {result.staves_printed}, which is neither "
                     f"of them.</b>"),
            "": ("<b>Look at the printed page above before deciding which side is "
                 "wrong.</b> Nobody has recorded what this system prints; when you "
                 "have looked, put the count in <code>fixturecheck/printed.json</code> "
                 "and this line will decide it next time."),
        }[result.at_fault]
        structure = (
            f"<p class='warn'>The reference says <b>{result.staves_page}</b> staves and "
            f"homr wrote <b>{result.staves_homr}</b>. Every note is matched on its "
            f"staff, so from the first staff that diverges the rows below are "
            f"comparing different music &mdash; read them as one wrong answer about "
            f"the staves, not as many wrong notes. {verdict}</p>")

    rows = "".join(
        f"<tr class='{row.kind if row.kind != 'agree' else ''}'>"
        f"<td>{html.escape(row.where)}</td><td>{html.escape(row.page)}</td>"
        f"<td>{html.escape(row.homr)}</td>"
        f"<td class='{'ok' if row.kind in ('agree', 'unison') else 'no'}'>"
        f"{html.escape(row.verdict)}</td></tr>"
        + _bar_detail_row(case, parsed, row)
        for row in result.rows)

    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(case.name)}</title><style>{STYLE}</style></head><body>
<p><a href="index.html">&larr; every case</a></p>
<h1>{html.escape(case.name)}</h1>
<p class="lead">{html.escape(case.origin)}
{'&mdash; committed fixture' if case.committed else ''}</p>
<div class="sum">
  <div><b>{result.agree}</b>agree {moved('agree')}</div>
  <div><b>{result.voice}</b>wrong voice {moved('voice')}</div>
  <div><b>{result.pitch}</b>wrong pitch {moved('pitch')}</div>
  <div><b>{result.size}</b>different number of notes {moved('size')}</div>
  <div><b>{result.timing}</b>beat shifted {moved('timing')}</div>
  <div><b>{result.unison}</b>unisons</div>
</div>
{structure}

<h3>Input &mdash; the printed page</h3>
<img src="{page.name}" alt="the printed system">
<h3>Output &mdash; what homr writes</h3>
{f'<img src="{output}" alt="homr">' if output else '<p class="lead">(could not engrave)</p>'}
<h3>Fixture &mdash; the reference</h3>
{f'<img src="{reference}" alt="reference">' if reference else '<p class="lead">(could not engrave)</p>'}

<h2>Every note, the page against homr's output</h2>
<p class="lead">Compared on staff position, so a score written an octave above
where it sounds is not counted wrong, and on which of the staff's voices a note
is in rather than the voice's number. This is homr's <b>output</b> &mdash; not
the noteheads it detected, which is a different layer and can disagree in either
direction.</p>
<table><tr><th>where</th><th>the page</th><th>homr</th><th></th></tr>
{rows}</table>
</body></html>"""
    target = OUT / f"{case.name}.html"
    target.write_text(body)
    return target.name


def _staves(entry: dict) -> str:
    """The staff disagreement, and who the page says is wrong where anyone looked."""
    if not entry["structure"]:
        return ""
    said = f"{entry['staves_page']} vs {entry['staves_homr']}"
    blame = entry.get("at_fault")
    return f"{said} <span class='down'>({blame})</span>" if blame else said


def index_page(entries: list[dict], tier: str, run: dict | None = None) -> Path:
    """The table of every case in this run, with what moved since the last one.

    `run` is the series record this run just wrote. It carries the two things
    the page could not say before and that made every number here ambiguous:
    which homr was measured, and what state the references were in. "The current
    homr" meant the fork's tip to one reader and the venv the choir sings from
    to another, and no figure anywhere distinguished them.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    total = {k: sum(e[k] for e in entries)
             for k in ("agree", "voice", "pitch", "size", "timing", "structure", "unison")}
    # Notes homr lost and beats it moved are faults, and are in the denominator.
    # Leaving them out asks only "of the notes homr wrote, how many are right",
    # under which a system missing half its notes reads 100%.
    judged = judged_over(total)

    def cell(entry: dict, field: str) -> str:
        now = entry[field]
        was = (entry.get("before") or {}).get(field)
        if was is None or was == now:
            return str(now)
        better = now < was if field != "agree" else now > was
        return f"{now} <span class='{'up' if better else 'down'}'>({now - was:+d})</span>"

    rows = "".join(
        f"<tr><td><a href='{html.escape(e['page'])}'>{html.escape(e['name'])}</a></td>"
        f"<td>{cell(e, 'agree')}</td><td>{cell(e, 'voice')}</td>"
        f"<td>{cell(e, 'pitch')}</td><td>{cell(e, 'size')}</td>"
        f"<td>{cell(e, 'timing')}</td>"
        f"<td>{_staves(e)}</td>"
        f"<td>{e['score']:.1f}%</td></tr>"
        # Worst first means worst by what went wrong, and a case can go wrong
        # without a single note pairing up to be called a wrong pitch:
        # laulun-aika-3-s5 agrees on one note out of fifty, loses 41 moments to
        # a count mismatch, and sorted 17th of 98 on voice+pitch alone. Counting
        # every fault puts it first, where a near-total loss belongs.
        for e in sorted(entries, key=lambda e: (
            -(e["voice"] + e["pitch"] + e["size"] + e.get("timing", 0)), e["name"])))

    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>fixture check &mdash; {html.escape(tier)}</title><style>{STYLE}</style></head><body>
<h1>Fixture check &mdash; {html.escape(tier)}</h1>
{_provenance(run)}
{_gate(run)}
<p class="lead">{len(entries)} case(s), each one printed system judged against a
reference built from its song's cleaned score. Sorted worst first. A number in
green or red is what changed since the last run of the same case &mdash; taken
from the series, so a run of ten cases is compared against each case's own last
measurement rather than against whatever ran immediately before.</p>
<div class="sum">
  <div><b>{total['agree']}</b>agree</div>
  <div><b>{total['voice']}</b>wrong voice</div>
  <div><b>{total['pitch']}</b>wrong pitch</div>
  <div><b>{total['size']}</b>different number of notes</div>
  <div><b>{total['timing']}</b>beat shifted</div>
  <div><b>{total['structure']}</b>case(s) with the wrong staves</div>
  <div><b>{100.0 * total['agree'] / judged if judged else 0:.1f}%</b>of everything judged is right</div>
</div>
<p class="lead">The percentage counts a note homr <b>lost</b> and a beat it
<b>moved</b> against it, as well as a note it read wrongly. It did not before,
and under the older definition a system missing half its notes could read 100%.
No figure here is comparable with one quoted before 2026-09-05.</p>
<p class="lead">A case whose staff count disagrees is one wrong answer about the
structure, and the note rows under it are then comparing different music &mdash;
read its counts as a consequence of that, not as many wrong notes. Whose wrong
answer it is has to be settled against the printed page. Where somebody has looked and
written the count into <code>fixturecheck/printed.json</code>, the staves column
names the side at fault; where nobody has, it does not guess.</p>
<table><tr><th>case</th><th>agree</th><th>wrong voice</th><th>wrong pitch</th>
<th>note count</th><th>beat shifted</th><th>staves (who is wrong)</th><th>score</th></tr>
{rows}</table>
{NOT_MEASURED}
</body></html>"""
    target = OUT / "index.html"
    target.write_text(body)
    return target
