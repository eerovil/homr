from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "homr/music_xml_generator.py"
TARGET = ROOT / "homr/score_reconstruction.py"

MOVE = [
    "SymbolChord",
    "find_common_division",
    "_bar_boundaries",
    "_staff_lengths",
    "_corroborated_length",
    "repair_bar_arithmetic",
    "_bar_targets",
    "_repair_for_bar",
    "_holds_a_rest",
    "_moments_agree",
    "_plain_value",
    "_stands_alone",
    "_rhythm_alternatives",
    "_length_with",
    "infer_meter_changes",
    "find_nominator_per_time_signature",
    "prevailing_length",
    "find_division_and_time_signature_nominator",
    "group_into_chords",
    "TupletParser",
    "add_tuplet_start_stop",
]

text = SOURCE.read_text()
lines = text.splitlines(keepends=True)
tree = ast.parse(text)
by_name = {
    node.name: node
    for node in tree.body
    if isinstance(node, ast.FunctionDef | ast.ClassDef)
}
segments: list[tuple[int, int, str]] = []
for name in MOVE:
    node = by_name[name]
    assert node.end_lineno is not None
    segments.append((node.lineno, node.end_lineno, "".join(lines[node.lineno - 1 : node.end_lineno])))
for node in tree.body:
    if not isinstance(node, ast.Assign | ast.AnnAssign):
        continue
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    if not any(
        isinstance(target, ast.Name) and target.id == "_REPAIR_WITNESS_BARS"
        for target in targets
    ):
        continue
    assert node.end_lineno is not None
    segments.append((node.lineno, node.end_lineno, "".join(lines[node.lineno - 1 : node.end_lineno])))

HEADER = '''"""Musical reconstruction between decoder tokens and MusicXML serialization.

This module owns the decisions that turn a flat decoder token stream into the
musical structure the serializer consumes: sounding moments, tuplet grouping,
bar-arithmetic repair, inferred meter changes, and the timing metadata derived
from those reconstructed bars. It deliberately contains no XML construction.

Keeping this phase explicit means a MusicXML serializer can be changed or
replaced without silently changing what homr believes the music is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol, SymbolDuration, sort_token_chords

'''
BODY = "\n\n".join(segment.rstrip() for _, _, segment in sorted(segments)) + "\n\n"
BODY = BODY.replace('list["SymbolChord"]', "list[SymbolChord]")
FOOTER = '''@dataclass(frozen=True)
class ReconstructedVoice:
    """The complete token-level musical structure ready for serialization."""

    groups: list[SymbolChord]
    division: int
    nominator: Fraction
    nominators: list[Fraction]


def reconstruct_voice(voice: list[EncodedSymbol]) -> ReconstructedVoice:
    """Apply homr's musical reconstruction passes in their established order."""
    groups = infer_meter_changes(
        add_tuplet_start_stop(repair_bar_arithmetic(group_into_chords(voice)))
    )
    division, nominator = find_division_and_time_signature_nominator(groups)
    return ReconstructedVoice(
        groups=groups,
        division=division,
        nominator=nominator,
        nominators=find_nominator_per_time_signature(groups, nominator),
    )
'''
TARGET.write_text(HEADER + BODY + FOOTER)

for start, end, _ in sorted(segments, reverse=True):
    del lines[start - 1 : end]
text = "".join(lines)
text = text.replace("import math\n", "")
text = text.replace("\nimport numpy as np\n", "\n")
text = text.replace("    SymbolDuration,\n", "")
text = text.replace("    sort_token_chords,\n", "")
needle = "from homr import constants\n"
imports = '''from homr.score_reconstruction import (
    SymbolChord,
    find_division_and_time_signature_nominator,
    find_nominator_per_time_signature,
    infer_meter_changes,
    prevailing_length,
    reconstruct_voice,
    repair_bar_arithmetic,
)
'''
assert text.count(needle) == 1
text = text.replace(needle, needle + imports)
old = '''    groups = infer_meter_changes(
        add_tuplet_start_stop(repair_bar_arithmetic(group_into_chords(voice)))
    )
    division, nominator = find_division_and_time_signature_nominator(groups)
    state = ConversionState(
        division, nominator, find_nominator_per_time_signature(groups, nominator)
    )
'''
new = '''    reconstructed = reconstruct_voice(voice)
    groups = reconstructed.groups
    division = reconstructed.division
    state = ConversionState(
        division, reconstructed.nominator, reconstructed.nominators
    )
'''
assert text.count(old) == 1
text = text.replace(old, new)
export_needle = '''from homr.transformer.vocabulary import (
    EncodedSymbol,
    empty,
    nonote,
)

'''
exports = '''from homr.transformer.vocabulary import (
    EncodedSymbol,
    empty,
    nonote,
)

__all__ = [
    "SymbolChord",
    "find_division_and_time_signature_nominator",
    "find_nominator_per_time_signature",
    "infer_meter_changes",
    "prevailing_length",
    "repair_bar_arithmetic",
]

'''
assert text.count(export_needle) == 1
SOURCE.write_text(text.replace(export_needle, exports))
