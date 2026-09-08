from pathlib import Path


def replace(path: str, old: str, new: str, count: int = 1) -> None:
    file = Path(path)
    text = file.read_text()
    actual = text.count(old)
    assert actual == count, (path, actual, old[:120])
    file.write_text(text.replace(old, new, count))


score = "homr/score_reconstruction.py"
xml = "homr/music_xml_generator.py"
main = "homr/main.py"

replace(
    score,
    "\n\nclass SymbolChord:\n",
    '''\n\n@dataclass(frozen=True)\nclass ReconstructionChange:\n    """One deliberate change made after decoding and before serialization."""\n\n    kind: str\n    bar: int\n    group: int\n    symbol: int | None\n    staff: str | None\n    pitch: str | None\n    before: str | None\n    after: str\n    reason: str\n\n\nclass SymbolChord:\n''',
)

replace(
    score,
    "def repair_bar_arithmetic(voice: list[SymbolChord]) -> list[SymbolChord]:",
    "def repair_bar_arithmetic(\n    voice: list[SymbolChord], changes: list[ReconstructionChange] | None = None\n) -> list[SymbolChord]:",
)
replace(
    score,
    '''    for span, target in zip(bars, _bar_targets(voice, bars), strict=True):\n        if target is None:\n            continue\n        repair = _repair_for_bar(voice, span, target)\n        if repair is not None:\n            chord_index, symbol_index, rhythm, why = repair\n            repairs[(chord_index, symbol_index)] = (rhythm, why)\n''',
    '''    for bar_number, (span, target) in enumerate(\n        zip(bars, _bar_targets(voice, bars), strict=True), start=1\n    ):\n        if target is None:\n            continue\n        repair = _repair_for_bar(voice, span, target)\n        if repair is not None:\n            chord_index, symbol_index, rhythm, why = repair\n            repairs[(chord_index, symbol_index)] = (rhythm, why, bar_number)\n''',
)
replace(
    score,
    '''            rhythm, why = accepted_repair\n            eprint(\n                f"Bar arithmetic: reading {symbol.pitch} as {rhythm} rather than "\n                f"{symbol.rhythm}, {why}"\n            )\n            symbols[symbol_index] = symbol.change_rhythm(rhythm)\n''',
    '''            rhythm, why, bar_number = accepted_repair\n            if changes is not None:\n                changes.append(\n                    ReconstructionChange(\n                        kind="rhythm_repair",\n                        bar=bar_number,\n                        group=chord_index,\n                        symbol=symbol_index,\n                        staff=symbol.position,\n                        pitch=symbol.pitch,\n                        before=symbol.rhythm,\n                        after=rhythm,\n                        reason=why,\n                    )\n                )\n            eprint(\n                f"Bar arithmetic: reading {symbol.pitch} as {rhythm} rather than "\n                f"{symbol.rhythm}, {why}"\n            )\n            symbols[symbol_index] = symbol.change_rhythm(rhythm)\n''',
)

replace(
    score,
    "def infer_meter_changes(voice: list[SymbolChord]) -> list[SymbolChord]:",
    "def infer_meter_changes(\n    voice: list[SymbolChord], changes: list[ReconstructionChange] | None = None\n) -> list[SymbolChord]:",
)
replace(
    score,
    "    for start, end in bars:\n",
    "    for bar_number, (start, end) in enumerate(bars, start=1):\n",
)
replace(
    score,
    '''        eprint(\n            f"Bar length {length} contradicts the {declared[span_no]} in force and every "\n            f"staff agrees on it: writing a {int(length * int(denominator))}/{denominator}"\n        )\n        inserted[start] = SymbolChord([EncodedSymbol(f"timeSignature/{denominator}")])\n        declared[span_no] = length\n''',
    '''        before = f"{int(declared[span_no] * int(denominator))}/{denominator}"\n        after = f"{int(length * int(denominator))}/{denominator}"\n        reason = "every detected staff agrees on the bar length"\n        if changes is not None:\n            changes.append(\n                ReconstructionChange(\n                    kind="meter_inference",\n                    bar=bar_number,\n                    group=start,\n                    symbol=None,\n                    staff=None,\n                    pitch=None,\n                    before=before,\n                    after=after,\n                    reason=reason,\n                )\n            )\n        eprint(\n            f"Bar length {length} contradicts the {declared[span_no]} in force and every "\n            f"staff agrees on it: writing a {after}"\n        )\n        inserted[start] = SymbolChord([EncodedSymbol(f"timeSignature/{denominator}")])\n        declared[span_no] = length\n''',
)

replace(
    score,
    '''    nominators: list[Fraction]\n\n\ndef reconstruct_voice(voice: list[EncodedSymbol]) -> ReconstructedVoice:\n    """Apply homr's musical reconstruction passes in their established order."""\n    groups = infer_meter_changes(\n        add_tuplet_start_stop(repair_bar_arithmetic(group_into_chords(voice)))\n    )\n    division, nominator = find_division_and_time_signature_nominator(groups)\n    return ReconstructedVoice(\n        groups=groups,\n        division=division,\n        nominator=nominator,\n        nominators=find_nominator_per_time_signature(groups, nominator),\n    )\n''',
    '''    nominators: list[Fraction]\n    changes: tuple[ReconstructionChange, ...]\n\n\ndef reconstruct_voice(voice: list[EncodedSymbol]) -> ReconstructedVoice:\n    """Apply homr's musical reconstruction passes in their established order."""\n    changes: list[ReconstructionChange] = []\n    groups = group_into_chords(voice)\n    groups = repair_bar_arithmetic(groups, changes)\n    groups = add_tuplet_start_stop(groups)\n    groups = infer_meter_changes(groups, changes)\n    division, nominator = find_division_and_time_signature_nominator(groups)\n    return ReconstructedVoice(\n        groups=groups,\n        division=division,\n        nominator=nominator,\n        nominators=find_nominator_per_time_signature(groups, nominator),\n        changes=tuple(changes),\n    )\n''',
)

replace(
    xml,
    '''from homr.score_reconstruction import (\n    SymbolChord,\n''',
    '''from homr.score_reconstruction import (\n    ReconstructionChange,\n    SymbolChord,\n''',
)
replace(
    xml,
    '''def generate_xml(\n    args: XmlGeneratorArguments, staffs: list[list[EncodedSymbol]], title: str\n) -> ET.Element:\n''',
    '''def generate_xml(\n    args: XmlGeneratorArguments,\n    staffs: list[list[EncodedSymbol]],\n    title: str,\n    reconstruction_changes: list[list[ReconstructionChange]] | None = None,\n) -> ET.Element:\n''',
)
replace(
    xml,
    '''    for index, staff in enumerate(staffs):\n        root.append(build_part(args, staff, index, has_two_staves_by_part[index]))\n    return root\n''',
    '''    for index, staff in enumerate(staffs):\n        changes: list[ReconstructionChange] | None = (\n            [] if reconstruction_changes is not None else None\n        )\n        root.append(\n            build_part(\n                args,\n                staff,\n                index,\n                has_two_staves_by_part[index],\n                reconstruction_changes=changes,\n            )\n        )\n        if reconstruction_changes is not None:\n            assert changes is not None\n            reconstruction_changes.append(changes)\n    return root\n''',
)
replace(
    xml,
    '''def build_part(\n    args: XmlGeneratorArguments, voice: list[EncodedSymbol], index: int, has_two_staves: bool\n) -> ET.Element:\n''',
    '''def build_part(\n    args: XmlGeneratorArguments,\n    voice: list[EncodedSymbol],\n    index: int,\n    has_two_staves: bool,\n    reconstruction_changes: list[ReconstructionChange] | None = None,\n) -> ET.Element:\n''',
)
replace(
    xml,
    '''    for measure in build_measures(args, voice, is_first_part, has_two_staves):\n''',
    '''    for measure in build_measures(\n        args,\n        voice,\n        is_first_part,\n        has_two_staves,\n        reconstruction_changes=reconstruction_changes,\n    ):\n''',
)
replace(
    xml,
    '''    is_first_part: bool,\n    has_two_staves: bool = False,\n) -> list[ET.Element]:\n''',
    '''    is_first_part: bool,\n    has_two_staves: bool = False,\n    reconstruction_changes: list[ReconstructionChange] | None = None,\n) -> list[ET.Element]:\n''',
)
replace(
    xml,
    '''    reconstructed = reconstruct_voice(voice)\n    groups = reconstructed.groups\n''',
    '''    reconstructed = reconstruct_voice(voice)\n    if reconstruction_changes is not None:\n        reconstruction_changes.extend(reconstructed.changes)\n    groups = reconstructed.groups\n''',
)

replace(
    main,
    "from homr.music_xml_generator import XmlGeneratorArguments, generate_xml\n",
    "from homr.music_xml_generator import XmlGeneratorArguments, generate_xml\nfrom homr.score_reconstruction import ReconstructionChange\n",
)
replace(
    main,
    '''        eprint("Writing XML", result_staffs)\n        xml = generate_xml(xml_generator_args, result_staffs, title)\n        ET.ElementTree(xml).write(xml_file, encoding="unicode", xml_declaration=True)\n        if config.write_confidence:\n            confidence_file = replace_extension(image_path, ".confidence.json")\n            _write_confidence(confidence_file, result_staffs)\n            eprint("Confidence was written to", confidence_file)\n''',
    '''        eprint("Writing XML", result_staffs)\n        reconstruction_changes: list[list[ReconstructionChange]] | None = (\n            [] if config.write_confidence else None\n        )\n        xml = generate_xml(\n            xml_generator_args,\n            result_staffs,\n            title,\n            reconstruction_changes=reconstruction_changes,\n        )\n        ET.ElementTree(xml).write(xml_file, encoding="unicode", xml_declaration=True)\n        if config.write_confidence:\n            confidence_file = replace_extension(image_path, ".confidence.json")\n            _write_confidence(\n                confidence_file, result_staffs, reconstruction_changes or []\n            )\n            eprint("Confidence was written to", confidence_file)\n''',
)
replace(
    main,
    "def _write_confidence(path: str, staffs: list[list[EncodedSymbol]]) -> None:\n",
    '''def _write_confidence(\n    path: str,\n    staffs: list[list[EncodedSymbol]],\n    reconstruction_changes: list[list[ReconstructionChange]] | None = None,\n) -> None:\n''',
)
replace(
    main,
    '''    with open(path, "w") as file:\n        json.dump({"version": 1, "symbols": records}, file, indent=2)\n        file.write("\\n")\n''',
    '''    changes = [\n        {\n            "staff": staff_index,\n            "kind": change.kind,\n            "bar": change.bar,\n            "group": change.group,\n            "symbol": change.symbol,\n            "position": change.staff,\n            "pitch": change.pitch,\n            "before": change.before,\n            "after": change.after,\n            "reason": change.reason,\n        }\n        for staff_index, staff_changes in enumerate(reconstruction_changes or [])\n        for change in staff_changes\n    ]\n    with open(path, "w") as file:\n        json.dump(\n            {\n                "version": 1,\n                "symbols": records,\n                "reconstruction": {"version": 1, "changes": changes},\n            },\n            file,\n            indent=2,\n        )\n        file.write("\\n")\n''',
)

print("Applied reconstruction provenance changes")
