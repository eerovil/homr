from pathlib import Path

path = Path("homr/staff_parsing.py")
text = path.read_text()

old = '''    staffs = _ensure_same_number_of_staffs(staffs)\n    # For simplicity we call every staff in a multi staff a voice,\n'''
new = '''    staffs = _ensure_same_number_of_staffs(staffs)\n    if selected_staff >= len(staffs):\n        raise IncompleteRecognitionError(\n            f"Selected staff {selected_staff} does not exist; page has {len(staffs)} staff rows"\n        )\n    # For simplicity we call every staff in a multi staff a voice,\n'''
assert text.count(old) == 1
text = text.replace(old, new)

old = '''            if len(result_staff) == 0:\n                raise IncompleteRecognitionError(f"Staff {i}: no symbols were recognized")\n            result_staff.append(EncodedSymbol("newline"))\n'''
new = '''            if len(result_staff) == 0:\n                raise IncompleteRecognitionError(f"Staff {i}: no symbols were recognized")\n            if not any(\n                symbol.rhythm.startswith(("note", "rest")) for symbol in result_staff\n            ):\n                raise IncompleteRecognitionError(\n                    f"Staff {i}: no notes or rests were recognized"\n                )\n            result_staff.append(EncodedSymbol("newline"))\n'''
assert text.count(old) == 1
text = text.replace(old, new)
path.write_text(text)
print("Applied remaining incomplete-recognition guards")
