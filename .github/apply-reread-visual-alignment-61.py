from pathlib import Path

path = Path("homr/reread.py")
text = path.read_text()

old = """from __future__ import annotations\n\nfrom fractions import Fraction\n"""
new = """from __future__ import annotations\n\nimport math\nfrom fractions import Fraction\nfrom statistics import median\n"""
assert text.count(old) == 1
text = text.replace(old, new)

old = """DOUBT_THRESHOLD = 0.5\n\n_HEADINGS = (\"clef\", \"keySignature\", \"timeSignature\")\n"""
new = """DOUBT_THRESHOLD = 0.5\n\n#: Horizontal decoder-attention distance that still means one printed column.\n#:\n#: This is the same scale already used to match decoder attention to segmented\n#: noteheads. On sammon-ryosto, corresponding upper/lower moments differ by\n#: 0.2--8.9 px; the unstable rest/D3 pair differs by 0.4 px, while the preceding\n#: lower note is about 40 px away. Keep the threshold below that inter-onset gap.\nMOMENT_X_TOLERANCE = 12.0\n\n_HEADINGS = (\"clef\", \"keySignature\", \"timeSignature\")\n"""
assert text.count(old) == 1
text = text.replace(old, new)

needle = """    return headings, timed\n\n\ndef _splice_headings(\n"""
insert = """    return headings, timed\n\n\ndef _moment_x(moment: list[EncodedSymbol]) -> float | None:\n    \"\"\"Median horizontal attention of a timed moment, when every head has one.\"\"\"\n    xs: list[float] = []\n    for symbol in moment:\n        if not _is_timed(symbol):\n            continue\n        coordinates = symbol.coordinates\n        if coordinates is None or len(coordinates) < 2:\n            return None\n        try:\n            x = float(coordinates[0])\n        except (TypeError, ValueError):\n            return None\n        if not math.isfinite(x):\n            return None\n        xs.append(x)\n    return float(median(xs)) if xs else None\n\n\ndef _merge_timed_by_page_x(\n    upper: list[tuple[Fraction, list[EncodedSymbol]]],\n    lower: list[tuple[Fraction, list[EncodedSymbol]]],\n) -> list[list[EncodedSymbol]] | None:\n    \"\"\"Merge two separately read staffs by their printed horizontal columns.\n\n    The separate reads come from the same crop, so decoder attention x is a\n    direct observation of horizontal placement on the page. Rhythm-derived\n    onsets are still kept as the fallback for old/model-free callers, but they\n    must not decide whether two staffs share a moment when a low-confidence\n    duration flips across CPU kernels while the printed x positions stay put.\n\n    A two-pointer merge preserves each staff's decoded order. Only one upper\n    and one lower moment can be joined, and only inside the measured x tolerance.\n    \"\"\"\n    upper_x = [_moment_x(moment) for _, moment in upper]\n    lower_x = [_moment_x(moment) for _, moment in lower]\n    if any(x is None for x in [*upper_x, *lower_x]):\n        return None\n\n    out: list[list[EncodedSymbol]] = []\n    upper_index = 0\n    lower_index = 0\n    while upper_index < len(upper) or lower_index < len(lower):\n        if upper_index >= len(upper):\n            out.append([_as_lower(symbol) for symbol in lower[lower_index][1]])\n            lower_index += 1\n            continue\n        if lower_index >= len(lower):\n            out.append(upper[upper_index][1])\n            upper_index += 1\n            continue\n\n        ux = upper_x[upper_index]\n        lx = lower_x[lower_index]\n        assert ux is not None and lx is not None\n        if abs(ux - lx) <= MOMENT_X_TOLERANCE:\n            out.append(\n                [\n                    *upper[upper_index][1],\n                    *(_as_lower(symbol) for symbol in lower[lower_index][1]),\n                ]\n            )\n            upper_index += 1\n            lower_index += 1\n        elif ux < lx:\n            out.append(upper[upper_index][1])\n            upper_index += 1\n        else:\n            out.append([_as_lower(symbol) for symbol in lower[lower_index][1]])\n            lower_index += 1\n    return out\n\n\ndef _splice_headings(\n"""
assert text.count(needle) == 1
text = text.replace(needle, insert)

old = """    at_onset: dict[Fraction, list[EncodedSymbol]] = {}\n    for onset, moment in upper_timed:\n        at_onset.setdefault(onset, []).extend(moment)\n    for onset, moment in lower_timed:\n        at_onset.setdefault(onset, []).extend(_as_lower(symbol) for symbol in moment)\n    moments.extend(at_onset[onset] for onset in sorted(at_onset))\n"""
new = """    visual = _merge_timed_by_page_x(upper_timed, lower_timed)\n    if visual is not None:\n        moments.extend(visual)\n    else:\n        at_onset: dict[Fraction, list[EncodedSymbol]] = {}\n        for onset, moment in upper_timed:\n            at_onset.setdefault(onset, []).extend(moment)\n        for onset, moment in lower_timed:\n            at_onset.setdefault(onset, []).extend(_as_lower(symbol) for symbol in moment)\n        moments.extend(at_onset[onset] for onset in sorted(at_onset))\n"""
assert text.count(old) == 1
text = text.replace(old, new)
path.write_text(text)
