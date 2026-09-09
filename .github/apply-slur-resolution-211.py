from pathlib import Path

path = Path("homr/music_xml_generator.py")
text = path.read_text()

import_needle = "from homr.simple_logging import eprint\n"
import_replacement = import_needle + "from homr.slur_resolution import resolve_slurs\n"
assert text.count(import_needle) == 1
assert "from homr.slur_resolution import resolve_slurs\n" not in text
text = text.replace(import_needle, import_replacement, 1)

call_needle = "    convert_ties(part)\n    return part\n"
call_replacement = "    convert_ties(part)\n    resolve_slurs(part)\n    return part\n"
assert text.count(call_needle) == 1
text = text.replace(call_needle, call_replacement, 1)

path.write_text(text)
