from pathlib import Path

path = Path("homr/main.py")
text = path.read_text()
needle = "from homr.image_prediction import PredictedSymbols, get_predictions, predict_symbols\n"
replacement = "from homr.image_prediction import get_predictions, predict_symbols\n"
if text.count(needle) != 1:
    raise SystemExit("prediction compatibility import changed")
path.write_text(text.replace(needle, replacement))
