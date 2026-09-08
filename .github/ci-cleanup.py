from pathlib import Path
r=Path.cwd()
def edit(name, old, new, count=1):
 p=r/name; s=p.read_text(); assert s.count(old)==count,(name,old,s.count(old)); p.write_text(s.replace(old,new))
# Both old tests require polyphonic evidence. Keep all old assertions and add a voice-1 control.
edit('tests/test_music_xml_generator.py', '''        measure.extend([first, self._build_test_backup(duration=4), second])

        rebalance_measure_voices(measure)''', '''        # Stem direction identifies a voice only on a polyphonic staff.
        up = self._build_test_note(duration=4, staff=1, voice=1)
        ET.SubElement(up, "stem").text = "up"
        measure.extend([
            up, self._build_test_backup(duration=4),
            first, self._build_test_backup(duration=4), second,
        ])

        rebalance_measure_voices(measure)

        self.assertEqual(self._read_note_voice(up), "1")''')
edit('tests/test_music_xml_generator.py', '''        measure.extend([first, second])

        rebalance_measure_voices(measure)''', '''        up = self._build_test_note(duration=4, staff=1, voice=1)
        ET.SubElement(up, "stem").text = "up"
        measure.extend([up, self._build_test_backup(duration=4), first, second])

        rebalance_measure_voices(measure)

        self.assertEqual(self._read_note_voice(up), "1")''')
edit('tests/test_music_xml_generator.py', '    def _build_test_note(\n', '''    def test_rebalance_measure_voices_keeps_a_lone_down_stem_in_voice_one(self) -> None:
        measure = ET.Element("measure")
        note = self._build_test_note(duration=4, staff=1, voice=2)
        ET.SubElement(note, "stem").text = "down"
        measure.append(note)

        rebalance_measure_voices(measure)

        self.assertEqual(self._read_note_voice(note), "1")
        self.assertEqual(note.findtext("stem"), "down")

    def _build_test_note(
''')
# Distinct variables for two different repair payload shapes.
edit('homr/music_xml_generator.py', '''            repair = repairs.get((chord_index, symbol_index))
            if repair is None:
                continue
            rhythm, why = repair''','''            accepted_repair = repairs.get((chord_index, symbol_index))
            if accepted_repair is None:
                continue
            rhythm, why = accepted_repair''')
# Use OpenCV's actual named parameter; the positional slot is the optional label array.
edit('homr/note_detection.py', '(noteheads > 0).astype(np.uint8), 8\n', '(noteheads > 0).astype(np.uint8), connectivity=8\n')
edit('homr/note_detection.py', '    h = bbox[3] - bbox[1]\n', '')
edit('homr/note_detection.py', '    return ink, ink & (1 - horizontal)\n', '    return ink, ink & (1 - horizontal.astype(np.uint8))\n')
edit('homr/note_detection.py', '    return [longest[direction] for direction in StemDirection if longest[direction] is not None]\n', '    return [stem for stem in longest.values() if stem is not None]\n')
edit('homr/note_detection.py', '        directions = [stem_direction(notehead, stem) for stem in found]\n', '''        directions = [
            direction for stem in found
            if (direction := stem_direction(notehead, stem)) is not None
        ]
''')
# Re-export is intentional compatibility for existing importers.
edit('homr/segmentation/inference_segnet.py', 'from homr.segmentation.patches import merge_patches as merge_patches\n', 'from homr.segmentation.patches import merge_patches as merge_patches  # noqa: PLC0414\n')
# Keep runtime lint policies; pytest asserts and in-memory/committed XML aren't untrusted input.
edit('pyproject.toml','[tool.ruff.lint.flake8-tidy-imports]\n', '''[tool.ruff.lint.per-file-ignores]
# Pytest assertions are executable checks, not production input validation.
# These modules parse only committed or test-built XML. Local imports allow
# fixtures to install monkeypatches before importing the component under test.
"tests/**/*.py" = ["S101", "S314", "PLC0415"]

[tool.ruff.lint.flake8-tidy-imports]
''')

import subprocess,sys
subprocess.run([sys.executable,'-m','black','homr','training','tests','validation'],check=True)
subprocess.run([sys.executable,'-m','isort','homr','training','tests','validation'],check=True)

assert subprocess.run([sys.executable,'-m','ruff','check','--fix','homr','training','tests','validation']).returncode in (0,1)

from pathlib import Path
import ast,io,tokenize
root=Path.cwd()/'tests'
# Typed test fixtures and helpers; the test bodies and their assertions stay intact.
common={'tmp_path':'Path','path':'Path','monkeypatch':'pytest.MonkeyPatch','capsys':'pytest.CaptureFixture[str]',
        'harness':'Harness','remembered':'Callable[[dict], None]','outcome':'str','case':'cases.Case'}
spec={
 'test_reread.py':{
  'note':({'rhythm':'str','pitch':'str','probability':'float','position':'str'},'EncodedSymbol'),
  'plain':({'rhythm':'str','position':'str | None'},'EncodedSymbol'),
  'tokens':({'symbols':'list[EncodedSymbol]'},'list[str]'),
 },
 'test_bar_arithmetic_repair.py':{
  'note':({'rhythm':'str','position':'str','stem':'str | None','alternatives':'tuple[str, ...]','pitch':'str'},'EncodedSymbol'),
  'moment':({'symbols':'EncodedSymbol'},'SymbolChord'), 'barline':({},'SymbolChord'),
  'even_bar':({'quarters':'int','positions':'tuple[str, ...]'},'list[SymbolChord]'),
  'rhythms':({'voice':'list[SymbolChord]','position':'str'},'list[str]'),
  'hanget_soi_bar_2':({'alternatives':'tuple[str, ...]'},'list[SymbolChord]'),
  'hanget_soi_bar_3':({},'list[SymbolChord]'),
  'system':({'bars':'list[SymbolChord]'},'list[SymbolChord]'),
  '_lower_beats':({'bar':'list[SymbolChord]'},'list[Fraction]'),
 },
 'test_inferred_meter_changes.py':{
  'chord':({'rhythm':'str','positions':'str'},'SymbolChord'),
  'bar':({'quarters':'int','staves':'tuple[str, ...]'},'list[SymbolChord]'),
  'signatures_of':({'voice':'list[SymbolChord]'},'list[int]'),
  'tokens':({'voice':'list[SymbolChord]'},'list[EncodedSymbol]'),
 },
 'test_time_signature_nominator.py':{
  'chord':({'rhythm':'str','position':'str'},'SymbolChord'),
  'bar':({'quarters':'int'},'list[SymbolChord]'),
 },
 'test_fixturecheck_memory.py':{
  '_counts':({'over':'int'},'dict[str, int]'),
  'record':({'name':'str','over':'int'},'series.CaseRecord'),
  'manifest':({'memories':'dict'},'Path'),
  'reading':({'score':'float','structure':'int','meter':'int'},'dict[str, float]'),
  'json_memory':({},'dict'),
  '_case':({'name':'str','image':'bytes','reference':'bytes'},'cases.Case'),
  '_source':({},'cases.Case'),
  'harness':({},'Harness'),
  'score_it':({'agree':'int','pitch':'int'},'None'),
 },
 'test_fixturecheck_pod.py':{'_isolated':({},'None'),'case':({},'cases.Case'),'reader':({},'str')},
 'test_fixturecheck_meter.py':{
  'score':({'meters':'list[str | None]','name':'str'},'Path'),
  'kinds':({'rows':'list[compare.Row]'},'list[str]'),
 },
 'test_fixturecheck_bars.py':{
  'score':({'text':'str','name':'str'},'Path'),
  '_geo':({'lines':'list[float]','staves':'int'},'dict'),
  '_mpos':({'count':'int','width':'float','y':'float','sy':'float','page':'int'},'list[dict]'),
 },
 'test_fixturecheck_report.py':{'_index':({'entries':'list[dict]'},'str')},
 'test_fixturecheck_series.py':{
  'case':({'name':'str','counts':'int'},'series.CaseRecord'),
  '_stands':({'name':'str','ok':'bool'},'series.CaseRecord'),
  '_accepted':({'names':'str','score':'float'},'dict'),
  'remembered':({},'Callable[[dict], None]'),
  'use':({'mapping':'dict'},'None'),
  '_fixture_case':({'name':'str','image':'bytes','reference':'bytes'},'cases.Case'),
  '_fixture_run':({'cases_':'list[series.CaseRecord]','at_committed':'tuple[str, ...]','kw':'Any'},'dict'),
  '_under':({'cases_':'list[series.CaseRecord]','homr':'str','references':'str','tier':'str','roster':'tuple[str, ...]'},'dict'),
  '_line':({'name':'str','at':'str'},'dict'),
  'commit':({'message':'str'},'str'),
 },
}
unknown=[]; changed=[]
for p in root.glob('*.py'):
 text=p.read_text(); tree=ast.parse(text); lines=text.splitlines(keepends=True)
 offsets=[0]
 for line in lines: offsets.append(offsets[-1]+len(line))
 def pos(l,c):return offsets[l-1]+len(lines[l-1].encode()[:c].decode())
 def tpos(pair):return offsets[pair[0]-1]+pair[1]
 toks=list(tokenize.generate_tokens(io.StringIO(text).readline))
 edits=[]; used=[]
 for f in ast.walk(tree):
  if not isinstance(f,ast.FunctionDef):continue
  args=f.args.posonlyargs+f.args.args+f.args.kwonlyargs+[x for x in [f.args.vararg,f.args.kwarg] if x]
  need=[a for a in args if a.arg not in ('self','cls') and a.annotation is None]
  if f.returns is not None and not need:continue
  amap, ret=spec.get(p.name,{}).get(f.name,({},'None' if f.name.startswith('test_') or any(isinstance(d,ast.Name) and d.id=='pytest' for d in []) else None))
  if ret is None and any(isinstance(d,(ast.Attribute,ast.Call)) and 'fixture' in ast.unparse(d) for d in f.decorator_list):ret='None'
  # Helpers that only act via side effects may safely declare None.
  if ret is None and not any(isinstance(n,(ast.Return,ast.Yield,ast.YieldFrom)) and getattr(n,'value',None) is not None for n in ast.walk(f)):ret='None'
  for a in need:
   hint=amap.get(a.arg,common.get(a.arg))
   if hint is None:unknown.append((p.name,f.name,a.arg));continue
   edits.append((pos(a.lineno,a.col_offset)+len(a.arg),': '+hint)); used.append(hint)
  if f.returns is None:
   if ret is None:unknown.append((p.name,f.name,'RETURN'));continue
   i=next(i for i,t in enumerate(toks) if t.start==(f.lineno,f.col_offset) and t.string=='def')
   depth=0
   for t in toks[i+1:]:
    if t.string in ('(','[','{'):depth+=1
    elif t.string in (')',']','}'):depth-=1
    elif t.string==':' and depth==0:
     edits.append((tpos(t.start),' -> '+ret));used.append(ret);break
 for off,add in sorted(edits,reverse=True):text=text[:off]+add+text[off:]
 if not edits:continue
 imports=[]
 if any('Path' in s for s in used):imports.append('from pathlib import Path')
 if any('pytest.' in s for s in used):imports.append('import pytest')
 if any('Callable' in s for s in used):imports.append('from collections.abc import Callable')
 if any('Any' in s for s in used):imports.append('from typing import Any')
 if any('cases.Case' in s for s in used):imports.append('from fixturecheck import cases')
 if any('SymbolChord' in s for s in used):imports.append('from homr.music_xml_generator import SymbolChord')
 if any('compare.Row' in s for s in used):imports.append('from fixturecheck import compare')
 if any('Harness' in s for s in used):
  imports.extend(['from collections.abc import Callable','from types import ModuleType','from pathlib import Path'])
 tree=ast.parse(text); body=tree.body; insertion=0
 if body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str):insertion=body[0].end_lineno
 future=[n for n in body if isinstance(n,ast.ImportFrom) and n.module=='__future__']
 if future:insertion=max(n.end_lineno for n in future)
 else:imports.insert(0,'from __future__ import annotations')
 txtlines=text.splitlines(keepends=True);txtlines[insertion:insertion]=['\n'+'\n'.join(imports)+'\n']
 text=''.join(txtlines)
 if any('Harness' in s for s in used):
  # Keep a typed fixture alias rather than repeating a three-part signature.
  idx=text.index('\ndef ')
  text=text[:idx]+'\nHarness = tuple[ModuleType, Path, Callable[..., None]]\n\n'+text[idx:]
 p.write_text(text);changed.append(p.name)
print('Changed',len(changed),'test modules')
print('Still need annotations:');print(*unknown,sep='\n')

import subprocess,sys
subprocess.run([sys.executable,'-m','black','homr','training','tests','validation'],check=True)
subprocess.run([sys.executable,'-m','isort','homr','training','tests','validation'],check=True)

from pathlib import Path
r=Path('.')
def edit(name,old,new,count=1):
 p=r/name;s=p.read_text();assert s.count(old)==count,(name,old,s.count(old));p.write_text(s.replace(old,new))
edit('homr/stem_voice_hints.py', '''        x = _note_coordinates(first)[0]
        clef = clefs[members[0]]
        claimed = expected_position(first.pitch, clef)''', '''        first_coordinates = _note_coordinates(first)
        second_coordinates = _note_coordinates(second)
        clef = clefs[members[0]]
        if first_coordinates is None or second_coordinates is None or clef is None:
            continue
        x = first_coordinates[0]
        claimed = expected_position(first.pitch, clef)''')
edit('homr/stem_voice_hints.py','abs(note.center[1] - _note_coordinates(first)[1])','abs(note.center[1] - first_coordinates[1])')
edit('homr/stem_voice_hints.py','abs(note.center[1] - _note_coordinates(second)[1])','abs(note.center[1] - second_coordinates[1])')
edit('homr/stem_voice_hints.py','(_note_coordinates(second)[1] > _note_coordinates(first)[1])','(second_coordinates[1] > first_coordinates[1])')
edit('homr/stem_voice_hints.py', '''            clef = clefs[members[0]]
            position = expected_position(first.pitch, clef)''','''            clef = clefs[members[0]]
            if clef is None:
                continue
            position = expected_position(first.pitch, clef)''',count=2)
edit('homr/stem_voice_hints.py', '''    x, y = _note_coordinates(first)
    here = [''','''    first_coordinates = _note_coordinates(first)
    second_coordinates = _note_coordinates(second)
    if first_coordinates is None or second_coordinates is None:
        return 0
    x, y = first_coordinates
    here = [''')
edit('homr/stem_voice_hints.py', '''    tokens = sorted((first, second), key=lambda s: _note_coordinates(s)[0])''','''    tokens = (
        (first, second) if first_coordinates[0] <= second_coordinates[0] else (second, first)
    )''')
edit('homr/stem_voice_hints.py', '''    above, below = sorted((first, second), key=lambda s: _note_coordinates(s)[1])
    top = _note_coordinates(above)[1]
    bottom = _note_coordinates(below)[1]
    middle = (top + bottom) / 2
    x = (_note_coordinates(first)[0] + _note_coordinates(second)[0]) / 2''', '''    first_coordinates = _note_coordinates(first)
    second_coordinates = _note_coordinates(second)
    if first_coordinates is None or second_coordinates is None:
        return 0
    above, below = (
        (first, second) if first_coordinates[1] <= second_coordinates[1] else (second, first)
    )
    top, bottom = sorted((first_coordinates[1], second_coordinates[1]))
    middle = (top + bottom) / 2
    x = (first_coordinates[0] + second_coordinates[0]) / 2''')
# Fixture-derived bar indexes are integers; preserve numeric messages exactly.
edit('tests/fixture_matching.py', '    moment: tuple | None = None','    moment: tuple[int, float] | None = None')
edit('tests/fixture_matching.py', '''        else:
            pairs.extend((None, other) for other in detected[column].heads)''','''        elif column is not None:
            pairs.extend((None, other) for other in detected[column].heads)''')
p=r/'tests/fixture_matching.py';s=p.read_text().replace('bars: dict[object,','bars: dict[int,');p.write_text(s)
# Test doubles deliberately expose just the attributes these tests exercise.
p=r/'tests/test_stem_voice_hints.py';s=p.read_text();s=s.replace('from types import SimpleNamespace','from types import SimpleNamespace\nfrom typing import cast').replace('from homr.model import StemDirection','from homr.model import Note, StemDirection');
s=s.replace(' -> SimpleNamespace:', ' -> Note:')
s=s.replace('return SimpleNamespace(center=(x, y), stem_directions=directions)','return cast(Note, SimpleNamespace(center=(x, y), stem_directions=directions))')
s=s.replace('return SimpleNamespace(center=(x, y), position=position, stem_directions=directions)','return cast(Note, SimpleNamespace(center=(x, y), position=position, stem_directions=directions))')
p.write_text(s)
# Module-internal imports aren't part of its public API: patch the imported modules directly.
p=r/'tests/test_recognition_failures.py';s=p.read_text().replace('from pathlib import Path','import sys\nfrom pathlib import Path').replace('import numpy as np','import numpy as np\nimport onnxruntime as ort');s=s.replace('cli.sys','sys').replace('cli.ort','ort').replace('parsing.reread','reread');s=s.replace('from homr import main as cli','from homr import reread\nfrom homr import main as cli')
s=s.replace('    upper, lower, fused, chosen = [Mock()], [Mock()], [Mock()], [Mock()]','''    upper: list[EncodedSymbol] = [Mock()]
    lower: list[EncodedSymbol] = [Mock()]
    fused: list[EncodedSymbol] = [Mock()]
    chosen: list[EncodedSymbol] = [Mock()]''')
s=s.replace('    detection = ([], np.zeros((1, 1)), debug, Mock(), [])','    detection: tuple = ([], np.zeros((1, 1)), debug, Mock(), [])')
s=s.replace('    assert cli.main() is None','    cli.main()  # Success returns normally; failures raise SystemExit.')
p.write_text(s)
# The new controls ensure missing coordinates never trigger guessing or a crash.
p=r/'tests/test_stem_voice_hints.py';s=p.read_text();s+='''\n\ndef test_unison_pairing_requires_coordinates_for_both_notes() -> None:
    first = _symbol(20, 40)
    missing = EncodedSymbol("note_4", pitch="C4", position="upper")
    from homr.stem_voice_hints import _pair, _pair_by_attention

    for left, right in ((first, missing), (missing, first)):
        assert _pair(left, right, [], 1) == 0
        assert _pair_by_attention(left, right, [], 1) == 0
    assert first.stem_direction is None
    assert missing.stem_direction is None
''';p.write_text(s)

import subprocess,sys
subprocess.run([sys.executable,'-m','black','homr','training','tests','validation'],check=True)
subprocess.run([sys.executable,'-m','isort','homr','training','tests','validation'],check=True)

from pathlib import Path
import ast
r=Path('.')
def replace(name, old,new):
 p=r/name;s=p.read_text();assert old in s,(name,old);p.write_text(s.replace(old,new))
for name in ['test_fixturecheck_memory.py','test_fixturecheck_series.py']:
 p=r/'tests'/name;s=p.read_text()
 old='''class Fake:
    def __init__(self, name, image, reference) -> None:
        self.name, self.image, self.reference = name, image, reference


'''
 assert old in s;s=s.replace(old,'').replace('return Fake(name, picture, score)','return cases.Case(name, picture, score, "fixture")');p.write_text(s)
replace('tests/test_fixturecheck_meter.py','def kinds(rows: list[compare.Row]) -> list[str]:','def kinds(rows: list[compare.Row]) -> list[tuple[str, str, str]]:')
replace('tests/test_fixturecheck_report.py','def _entry(name: str, **over) -> dict:','def _entry(name: str, **over: object) -> dict:')
replace('tests/test_segmentation_patches.py','    patches = []','    patches: list[np.ndarray] = []')
replace('tests/test_segmentation_inference.py','from homr.segmentation import inference_segnet as segnet','from homr.segmentation import inference_segnet as segnet\nfrom homr.segmentation.config import segmentation_version')
replace('tests/test_segmentation_inference.py','segnet.segmentation_version','segmentation_version')
replace('tests/test_stem_recovery_attachment.py','    ink = np.zeros((80, 80), dtype=np.uint8)','    ink: np.ndarray = np.zeros((80, 80), dtype=np.uint8)')
replace('tests/test_stem_direction_fixtures.py','    notes = [\n','    notes: list[dict] = [\n')
replace('tests/test_recognition_failures.py','from homr.errors import IncompleteRecognitionError','from homr.errors import IncompleteRecognitionError\nfrom homr.model import MultiStaff')
replace('tests/test_recognition_failures.py','    rows = [Mock(staffs=[Mock()]), Mock(staffs=[Mock()])]','    rows: list[MultiStaff] = [Mock(staffs=[Mock()]), Mock(staffs=[Mock()])]')
replace('tests/test_bar_arithmetic_repair.py','    voice[4].symbols[0].confidence["rhythm"]["alternatives"] = [','    confidence = voice[4].symbols[0].confidence\n    assert confidence is not None\n    confidence["rhythm"]["alternatives"] = [')
# Explicitly prove optional results exist before examining them, rather than using casts.
for file,var,prefix in [('tests/test_reread.py','spliced','reread.splice'), ('tests/test_fixturecheck_series.py','gate','series.published_gate'),('tests/test_fixturecheck_bars.py','kept','bars.one_bar')]:
 p=r/file;s=p.read_text();t=ast.parse(s);lines=s.splitlines(keepends=True);inserts=[]
 for n in ast.walk(t):
  if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id==var and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)==prefix:
   inserts.append((n.end_lineno,' '*n.col_offset+f'assert {var} is not None\n'))
 for i,txt in sorted(inserts,reverse=True):lines.insert(i,txt)
 p.write_text(''.join(lines))
replace('tests/test_fixturecheck_bars.py','    for box in (printed, engraved):\n        assert box is not None','    assert printed is not None\n    assert engraved is not None')
replace('tests/test_fixturecheck_series.py','    assert series.latest("fixturecheck", path)["headline"]["percent"] == 80.0\n    assert series.latest("choir-bench", path)["headline"]["percent"] == 50.0', '''    fixture_run = series.latest("fixturecheck", path)
    benchmark_run = series.latest("choir-bench", path)
    assert fixture_run is not None and benchmark_run is not None
    assert fixture_run["headline"]["percent"] == 80.0
    assert benchmark_run["headline"]["percent"] == 50.0''')
replace('tests/test_fixturecheck_series.py','    assert series.published_gate(series.runs(path))["passed"]','    gate = series.published_gate(series.runs(path))\n    assert gate is not None\n    assert gate["passed"]')
# Checking a subprocess success explicitly strengthens the test's precondition.
replace('tests/test_fixturecheck_series.py','cwd=tree, capture_output=True, text=True','cwd=tree, capture_output=True, text=True, check=True')
# These arguments are literal test-only git commands, not production user inputs.
p=r/'tests/test_fixturecheck_series.py';s=p.read_text();s='# ruff: noqa: S607 -- fixed git commands in disposable test repositories\n'+s
s=s.replace('''        sp.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", message],''','''        sp.run(  # noqa: S603 -- message is a fixed test string, not shell input
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", message],''')
p.write_text(s)
replace('tests/test_time_signature_nominator.py','zip(found, (4, 4, 2))','zip(found, (4, 4, 2), strict=True)')
replace('tests/test_inferred_meter_changes.py','[c for c in voice if c.symbols[0].rhythm.startswith("timeSignature")],\n        )','[c for c in voice if c.symbols[0].rhythm.startswith("timeSignature")],\n            strict=True,\n        )')

import subprocess,sys
subprocess.run([sys.executable,'-m','black','homr','training','tests','validation'],check=True)
subprocess.run([sys.executable,'-m','isort','homr','training','tests','validation'],check=True)

assert subprocess.run([sys.executable,'-m','ruff','check','--fix','homr','training','tests','validation']).returncode in (0,1)

p=Path('tests/test_segmentation_patches.py')
p.write_text(p.read_text().replace('    patches = [np.full((2, 2), label, dtype=dtype) for label in (2, 4)]','    patches: list[np.ndarray] = [np.full((2, 2), label, dtype=dtype) for label in (2, 4)]'))
p=Path('homr/model.py');p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))

# Refuse to publish anything except the exact files reviewed and tested locally.
import hashlib
expected_paths = ['homr/main.py', 'homr/model.py', 'homr/music_xml_generator.py', 'homr/note_detection.py', 'homr/reread.py', 'homr/segmentation/inference_segnet.py', 'homr/stem_voice_hints.py', 'homr/transformer/decoder_inference.py', 'homr/transformer/score_settings.py', 'pyproject.toml', 'tests/fixture_matching.py', 'tests/fixture_reference.py', 'tests/test_bar_arithmetic_repair.py', 'tests/test_decoder_confidence.py', 'tests/test_fixturecheck_bars.py', 'tests/test_fixturecheck_compare.py', 'tests/test_fixturecheck_memory.py', 'tests/test_fixturecheck_meter.py', 'tests/test_fixturecheck_pod.py', 'tests/test_fixturecheck_report.py', 'tests/test_fixturecheck_series.py', 'tests/test_fixturecheck_serve.py', 'tests/test_fixturecheck_warnings.py', 'tests/test_inferred_meter_changes.py', 'tests/test_music_xml_generator.py', 'tests/test_note_detection.py', 'tests/test_recognition_failures.py', 'tests/test_recognition_fixtures.py', 'tests/test_reread.py', 'tests/test_score_settings.py', 'tests/test_segmentation_inference.py', 'tests/test_segmentation_patches.py', 'tests/test_stem_direction_fixtures.py', 'tests/test_stem_recovery_attachment.py', 'tests/test_stem_voice_hints.py', 'tests/test_time_signature_nominator.py']
changed = subprocess.check_output(['git','diff','--name-only','HEAD'],text=True).splitlines()
assert changed == expected_paths, ('unexpected changed paths', changed)
digest = hashlib.sha256()
for name in expected_paths:
    digest.update(name.encode() + b'\0' + Path(name).read_bytes() + b'\0')
assert digest.hexdigest() == '3123df059998acda6201378643f1ae64d43832612ca0b883f4f5068bb7ef49c0', 'Result differs from the locally tested source'
subprocess.run(['git','diff','--check'],check=True)
subprocess.run(['git','diff','--exit-code','--','fixtures','fixturecheck'],check=True)
subprocess.run(['git','add','--',*expected_paths],check=True)
print('Verified 36 exact source/test files; original fixtures and reference criteria unchanged')
