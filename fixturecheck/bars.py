"""One bar, as a picture — the printed ink, and what each side made of it.

A fault row says a fault happened. The table beside it says what each side holds
at that beat. Neither shows the *music*, and on this repertoire the question a
fault actually raises is "which of these two is wrong about the page", which is
settled by looking at the page.

So a fault can be opened out into three pictures of the same bar: the printed
crop, that bar engraved from homr's parse, and that bar engraved from the
reference. Two of the three are reliable — a bar can always be cut out of a
MusicXML file and handed to MuseScore. The third is the interesting one and the
one that can fail, because finding a bar in a photograph means finding its
barlines, and that is homr's own detection rather than anything this file knows.

**A crop of the wrong bar is worse than no crop.** The detected barlines are
counted against the bars the reference says the system has, and where they
disagree nothing is shown and the row says so. Guessing would put a confident
picture of the wrong music under a fault, which is the failure this whole
harness was built to stop happening one level up.
"""

from __future__ import annotations

import copy
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

#: Two barlines closer than this fraction of the width are the same line seen on
#: another staff. A system is rarely more than ~12 bars, so bars are wide.
SAME_LINE = 0.01

#: How close to the staff's own left edge a detected line has to be to *be* the
#: system's opening rule. Further than this and the opening was not detected,
#: which is the ordinary case: `detect_bar_lines` looks for barlines, and a
#: system's opening rule is one of a pair with the bracket.
OPENING_NEAR = 0.03

#: A gap this much smaller than the system's typical bar is not a bar. Two lines
#: that close are a **double barline** — a section end, thin against thick — or
#: one of them is not a barline at all. `sammon-ryosto` has such a pair 0.049 of
#: the width apart where its bars average 0.23, and treating it as a fifth bar
#: is what put every crop on that case one bar out.
#:
#: Measured against the median gap rather than a fixed distance, because how
#: wide a bar is depends entirely on how many the system holds.
NARROW_SHARE = 0.35

#: How much wider than the bar itself a crop is drawn, as a fraction of the
#: bar's own size. A bar cut exactly at its barlines reads as a fragment: what
#: the note before it was, and whether the phrase carries on, are part of
#: judging whether a reading is wrong. Applied to both the printed crop and the
#: engraved ones, so the three stay the same shot of the same music.
ZOOM_OUT = 0.30


def _widen(left: float, top: float, right: float, bottom: float,
           by: float = ZOOM_OUT) -> tuple[float, float, float, float]:
    """Grow a box around its own middle, without leaving the picture."""
    grow_x = (right - left) * by / 2
    grow_y = (bottom - top) * by / 2
    return (max(0.0, left - grow_x), max(0.0, top - grow_y),
            min(1.0, right + grow_x), min(1.0, bottom + grow_y))


def bars_in(path: Path) -> list[str]:
    """The bar numbers a score holds, in order, as they are written."""
    seen: list[str] = []
    for part in ET.parse(path).getroot().findall("part"):
        for measure in part.findall("measure"):
            number = measure.get("number", "")
            if number and number not in seen:
                seen.append(number)
    return seen


def one_bar(source: Path, bar: str, into: Path) -> Path | None:
    """A score holding only one bar, so MuseScore can draw just that bar.

    The kept bar carries the attributes of the first one — divisions, key, time
    and clef — because a bar in the middle of a system declares none of them and
    would otherwise be engraved in whatever MuseScore assumes. The point is to
    look at the same music both sides wrote, so it has to be spelled the same.
    """
    tree = ET.parse(source)
    root = tree.getroot()
    kept_any = False
    for part in root.findall("part"):
        measures = part.findall("measure")
        first = measures[0] if measures else None
        opening = first.find("attributes") if first is not None else None
        keep = [m for m in measures if m.get("number") == bar]
        for measure in measures:
            if measure not in keep:
                part.remove(measure)
        for measure in keep:
            kept_any = True
            measure.set("number", bar)
            if measure.find("attributes") is None and opening is not None:
                measure.insert(0, copy.deepcopy(opening))
    if not kept_any:
        return None
    into.parent.mkdir(parents=True, exist_ok=True)
    tree.write(into, encoding="utf-8", xml_declaration=True)
    return into


# --- where the bar is on the printed page --------------------------------


def geometry(image: Path, cache: Path) -> dict | None:
    """Where the staves and barlines are, from homr's own detection.

    Cached: it is a segmentation pass over the picture, seconds rather than the
    ~30s of a full read, but a report page has many faults and they are all
    about the same picture.

    Returns `None` rather than raising. This is an explanation of a finding, not
    the finding; a page that cannot be measured must cost its crops and nothing
    else.
    """
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except ValueError:
            pass
    try:
        found = _detect(image)
    except Exception:                                        # noqa: BLE001
        return None
    cache.write_text(json.dumps(found))
    return found


def _detect(image: Path) -> dict:
    """homr's segmentation and `detect_staff`, stopped before any music is read.

    Deliberately the same sequence the choir app's `scripts/homr_staves.py`
    uses, for the same reason: it is where homr's own pipeline stops caring
    about pixels. `autocrop` is skipped there and skipped here — the input is a
    band this project rasterised out of a PDF, not a photograph of a page on a
    desk, so there is no page to find inside it, and autocrop returns the
    cropped image without saying where it cut.
    """
    import cv2
    import numpy as np

    from homr import color_adjust
    from homr.bar_line_detection import detect_bar_lines
    from homr.debug import Debug
    from homr.main import get_predictions, predict_symbols
    from homr.noise_filtering import filter_predictions
    from homr.note_detection import combine_noteheads_with_stems
    from homr.resize import resize_image
    from homr.staff_detection import break_wide_fragments, detect_staff, make_lines_stronger

    picture = cv2.imread(str(image))
    if picture is None:
        raise ValueError(f"could not read {image}")
    picture = resize_image(picture)
    preprocessed = color_adjust.apply_clahe(picture)
    predictions = get_predictions(picture, preprocessed, str(image), False, False)
    debug = Debug(predictions.original, str(image), False)
    predictions = filter_predictions(predictions, debug)
    predictions.staff = make_lines_stronger(predictions.staff, (1, 2))

    symbols = predict_symbols(debug, predictions)
    symbols.staff_fragments = break_wide_fragments(symbols.staff_fragments)
    with_stems = combine_noteheads_with_stems(symbols.noteheads, symbols.stems_rest)
    if not with_stems:
        raise ValueError("no noteheads")
    head_height = float(np.median([n.notehead.size[1] for n in with_stems]))
    heads = [n.notehead for n in with_stems]
    stems = [n.stem for n in with_stems if n.stem is not None]
    lines = [line for line in symbols.bar_lines
             if not line.is_overlapping_with_any(heads)
             and not line.is_overlapping_with_any(stems)]
    boxes = detect_bar_lines(lines, head_height)
    staves = detect_staff(debug, predictions.staff, symbols.staff_fragments,
                          symbols.clefs_keys, boxes)

    height, width = predictions.staff.shape[:2]
    return {
        "staves": [{"top": s.min_y / height, "bottom": s.max_y / height,
                    "left": s.min_x / width, "right": s.max_x / width}
                   for s in staves],
        "bar_lines": sorted(b.center[0] / width for b in boxes),
    }


def _boundaries(bar_lines: list[float]) -> list[float]:
    """The distinct vertical lines, one per barline rather than one per staff."""
    distinct: list[float] = []
    for x in bar_lines:
        if not distinct or x - distinct[-1] > SAME_LINE:
            distinct.append(x)
    return distinct


def boundaries_for(geo: dict, expected_bars: int) -> list[float] | None:
    """The vertical lines that cut this system into exactly the bars it holds.

    **The line a system opens with is usually not detected**, and that is not a
    failure: `detect_bar_lines` is looking for barlines, and a system's opening
    rule is one of a pair with the bracket. So `n` bars normally come back as
    `n` lines -- the internal ones and the final -- and the left edge of the
    staff is where the first bar starts. Adding it back is reading the same fact
    `system_finder` already relies on from the other side, that an opening line
    is not evidence about grouping.

    Anything other than `n` or `n+1` lines is refused. The numbering would then
    be a guess, and a guessed crop is a confident picture of the wrong music.
    """
    lines = _boundaries(geo.get("bar_lines", []))
    staves = geo.get("staves", [])
    if not lines or not staves:
        return None
    # **Whether the opening line is there is decided by where it would be, not
    # by the count.** Counting was the first rule and it is not safe: on
    # `sammon-ryosto` the detection missed the opening rule *and* found one
    # spurious line, which came to exactly `n + 1` and was accepted, so every
    # crop on that case was a bar to the right of the one its row named. Two
    # errors cancelling in a count is indistinguishable from no errors; two
    # errors cancelling against the staff's own left edge is not.
    opening = min(s.get("left", 0.0) for s in staves)
    if lines[0] - opening > OPENING_NEAR:
        lines = [opening] + lines
    lines = _merge_narrow(lines)
    if len(lines) - 1 != expected_bars:
        return None
    return lines


def _merge_narrow(lines: list[float]) -> list[float]:
    """Fold away boundaries too close together to be separate bars.

    A double barline is two lines a few millimetres apart and one bar boundary,
    and so is a line the detector invented beside a real one. Either way the
    gap it opens is a small fraction of a real bar, so it is measured against
    the system's own median gap rather than against a distance.

    The **right-hand** line is the one kept: at a thin-thick double bar the
    music ends at the thick one, and where the extra line is noise the two are
    close enough that the crop's own widening covers the difference.

    Nothing here looks at how many bars the score has. Merging until the count
    came out right would be fitting the geometry to the answer, which is how
    the off-by-one it fixes got in.
    """
    if len(lines) < 3:
        return lines
    gaps = sorted(b - a for a, b in zip(lines, lines[1:]))
    median = gaps[len(gaps) // 2]
    if median <= 0:
        return lines
    kept = [lines[0]]
    for line in lines[1:]:
        if line - kept[-1] < NARROW_SHARE * median:
            kept[-1] = line          # the right-hand line is the boundary
        else:
            kept.append(line)
    return kept


def implied_bars(geo: dict) -> int:
    """How many bars the detected lines cut this system into.

    The same reading `boundaries_for` does, minus the check against the score,
    so a refusal can say what was actually found instead of restating its own
    rule back at the reader.
    """
    lines = _boundaries(geo.get("bar_lines", []))
    staves = geo.get("staves", [])
    if not lines or not staves:
        return 0
    opening = min(s.get("left", 0.0) for s in staves)
    if lines[0] - opening > OPENING_NEAR:
        lines = [opening] + lines
    return max(0, len(lines) - 1)


def bar_box(geo: dict, staff: int, index: int, expected_bars: int) -> dict | None:
    """The box around one bar of one staff, as fractions of the picture.

    `index` is which bar of the system, counted from 1; `staff` likewise from 1.

    **It refuses more often than it answers, on purpose.** The detected lines
    have to imply exactly as many bars as the reference says the system holds,
    or the numbering is a guess — and a guessed crop is a confident picture of
    the wrong music underneath a fault, which is precisely the mistake this
    project keeps finding in its own conclusions.
    """
    staves = geo.get("staves", [])
    if not (1 <= staff <= len(staves)) or not (1 <= index <= expected_bars):
        return None
    lines = boundaries_for(geo, expected_bars)
    if lines is None:
        return None
    left, right = lines[index - 1], lines[index]
    row = staves[staff - 1]
    # **The staff the fault is on, not the whole system.** Taking the union of
    # the staves was tried and is worse on real scans: printed choral staves are
    # spaced apart to leave room for the words between them, so on `hanget-soi`
    # the system spans 60% of the band and the crop came out three times the
    # height of the engravings beside it, half of it white paper and a line of
    # lyrics. The engraved crops show the system because MuseScore reports one
    # box per measure and that is what it gives; the labels say which is which,
    # and the tables above have already isolated the staff.
    pad = 0.35 * (row["bottom"] - row["top"])
    l, t, r, b = _widen(left - 0.004, row["top"] - pad, right + 0.004,
                        row["bottom"] + pad)
    return {"left": l, "right": r, "top": t, "bottom": b}


#: A MuseScore staff is 4 spatia tall and the default spatium is 1.764 mm, so a
#: five-line staff is 0.2778 in however big the paper is. Used to put an
#: engraving on the same scale as a scan.
STAFF_INCHES = 4 * 1.764 / 25.4

#: How tall one staff is drawn in a crop, in pixels. The three pictures under a
#: fault come from a scan and from two engravings and are naturally at three
#: different scales; left alone they were shown at three more, because each was
#: stretched to the column it sat in. Scaling every crop so that **a staff is
#: the same size** puts the music at one scale, which is the thing being
#: compared.
#:
#: Small enough that the three of them fit side by side without the row having
#: to scroll: on the widest fixture bar they come to a little over a thousand
#: pixels together, against the page's 1150.
STAFF_PIXELS = 38

#: The resolution the engravings are drawn at, and what `.mpos` units are
#: converted with. One place, because the crop's scale depends on it too.
RENDER_DPI = 220


def crop(image: Path, box: dict, into: Path, staff_px: float = 0.0) -> Path | None:
    """Cut the box out of the picture, and put the music on the common scale.

    `staff_px` is how tall one staff is in *this* picture. Given it, the crop is
    resized so a staff comes out `STAFF_PIXELS` tall, the same as in the crops
    beside it. Without it the crop is cut and not resized.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(image) as picture:
            width, height = picture.size
            cut = picture.crop((int(box["left"] * width), int(box["top"] * height),
                                int(box["right"] * width), int(box["bottom"] * height)))
            if not cut.width or not cut.height:
                return None
            if cut.mode in ("RGBA", "LA", "P"):
                # MuseScore writes transparent paper. Left alone, the crop takes
                # the page's colour rather than the white the pictures beside it
                # are on, and the same bar looks like two different prints.
                cut = cut.convert("RGBA")
                paper = Image.new("RGBA", cut.size, (255, 255, 255, 255))
                cut = Image.alpha_composite(paper, cut).convert("RGB")
            if staff_px > 1:
                scale = STAFF_PIXELS / staff_px
                # Bounded, so a badly measured staff cannot produce a picture
                # of a few pixels or of tens of thousands.
                scale = max(0.15, min(6.0, scale))
                cut = cut.resize((max(1, round(cut.width * scale)),
                                  max(1, round(cut.height * scale))),
                                 Image.LANCZOS)
            into.parent.mkdir(parents=True, exist_ok=True)
            cut.save(into)
    except Exception:                                        # noqa: BLE001
        return None
    return into


# --- where the bar is in an engraving ------------------------------------

#: `.mpos` coordinates are absolute page units with the origin at the page
#: corner, and this converts them to pixels at a given resolution:
#:
#:     pixels = units * dpi / MPOS_UNITS_PER_INCH
#:
#: Measured rather than derived, and checked before it was relied on: two
#: different scores at 220 and 150 dpi, with the crop landing on the barlines at
#: both edges each time (`test_fixturecheck_bars`). Being a rate per inch rather
#: than a page width, it does not assume the paper size.
MPOS_UNITS_PER_INCH = 2649.2


def engraved(source: Path, cli: str, into: Path, dpi: int = RENDER_DPI
             ) -> tuple[Path, list[dict]] | None:
    """Draw a score once, and ask MuseScore where each bar of it landed.

    Two outputs from two calls, and the second is the point: `-o <file>.mpos`
    writes a box per measure, so a bar can be **cut out of the picture the score
    already makes** instead of being engraved again on its own. Re-engraving one
    bar gives it a title, a fresh layout and a clef and key it does not have in
    context, so the detail looked like different music from the system above it.

    The page is rendered **untrimmed**: `-T` crops to the ink and moves the
    origin, which is exactly what the measure boxes are measured from.
    """
    into.parent.mkdir(parents=True, exist_ok=True)
    positions = into.with_suffix(".mpos")
    for target in (into, positions):
        run = subprocess.run([cli, "-r", str(dpi), str(source), "-o", str(target)],
                             capture_output=True, text=True, timeout=600)
        if run.returncode != 0:
            return None
    page = into.with_name(f"{into.stem}-1.png")
    if page.exists():
        page.replace(into)
    if not into.exists() or not positions.exists():
        return None
    try:
        boxes = [
            {"x": float(e.get("x", 0)), "y": float(e.get("y", 0)),
             "sx": float(e.get("sx", 0)), "sy": float(e.get("sy", 0)),
             "page": int(e.get("page", 0))}
            for e in ET.parse(positions).getroot().findall("elements/element")]
    except (ET.ParseError, ValueError):
        return None
    return into, boxes


def engraved_box(boxes: list[dict], numbers: list[str], bar: str,
                 size: tuple[int, int], dpi: int = RENDER_DPI) -> dict | None:
    """The box around one bar of an engraving, as fractions of the picture.

    Refused when the boxes and the score disagree about how many bars there are,
    when the bar is on a page other than the first, or when the box falls
    outside the picture — each of which would put a crop of the wrong music
    under a finding, which is the one thing this must not do.
    """
    if len(boxes) != len(numbers) or bar not in numbers:
        return None
    box = boxes[numbers.index(bar)]
    if box["page"] != 0:
        return None
    width, height = size
    scale = dpi / MPOS_UNITS_PER_INCH
    # A little air, so the crop is a bar of music and not a slice through it.
    pad_x, pad_y = 8, 0.09 * box["sy"] * scale
    left = box["x"] * scale - pad_x
    right = (box["x"] + box["sx"]) * scale + pad_x
    top = box["y"] * scale - pad_y
    bottom = (box["y"] + box["sy"]) * scale + pad_y
    if right > width + pad_x or bottom > height + pad_y or right <= left:
        return None
    l, t, r, b = _widen(left / width, top / height, right / width, bottom / height)
    return {"left": l, "right": r, "top": t, "bottom": b}
