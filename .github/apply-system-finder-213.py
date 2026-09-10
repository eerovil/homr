from pathlib import Path

path = Path("homr/main.py")
text = path.read_text()

# The finder and full recognizer share these exact pixel-level primitives. Move
# the implementation once rather than making system_finder import main lazily or
# duplicating the pipeline.
text = text.replace(
    '''from homr.bar_line_detection import (\n    detect_bar_lines,\n    prepare_bar_line_image,\n)\nfrom homr.bounding_boxes import (\n    BoundingEllipse,\n    RotatedBoundingBox,\n    create_bounding_ellipses,\n    create_rotated_bounding_boxes,\n)\n''',
    '''from homr.bar_line_detection import detect_bar_lines\nfrom homr.bounding_boxes import create_rotated_bounding_boxes\n''',
)
text = text.replace(
    '''from homr.errors import IncompleteRecognitionError\nfrom homr.model import InputPredictions, MultiStaff, Staff\n''',
    '''from homr.errors import IncompleteRecognitionError\nfrom homr.image_prediction import PredictedSymbols, get_predictions, predict_symbols\nfrom homr.model import InputPredictions, MultiStaff, Staff\n''',
)
text = text.replace(
    '''from homr.segmentation.inference_segnet import extract\nfrom homr.simple_logging import eprint\n''',
    '''from homr.simple_logging import eprint\n''',
)
text = text.replace(
    '''from homr.system_crops import (\n    DEFAULT_SYSTEM_DPI,\n    DEFAULT_SYSTEM_PAD,\n    load_system_bounds,\n    render_system_crops,\n    select_system,\n)\n''',
    '''from homr.system_crops import (\n    DEFAULT_SYSTEM_DPI,\n    DEFAULT_SYSTEM_PAD,\n    load_system_bounds,\n    render_system_crops,\n    select_system,\n)\nfrom homr.system_finder import bounds_payload, find_system_bounds\n''',
)

class_block = '''class PredictedSymbols:\n    def __init__(\n        self,\n        noteheads: list[BoundingEllipse],\n        staff_fragments: list[RotatedBoundingBox],\n        clefs_keys: list[RotatedBoundingBox],\n        stems_rest: list[RotatedBoundingBox],\n        bar_lines: list[RotatedBoundingBox],\n    ) -> None:\n        self.noteheads = noteheads\n        self.staff_fragments = staff_fragments\n        self.clefs_keys = clefs_keys\n        self.stems_rest = stems_rest\n        self.bar_lines = bar_lines\n\n\n'''
if text.count(class_block) != 1:
    raise SystemExit("PredictedSymbols anchor changed")
text = text.replace(class_block, "")

predictions_block = '''def get_predictions(\n    original: NDArray,\n    preprocessed: NDArray,\n    img_path: str,\n    enable_cache: bool,\n    segnet_use_gpu: bool,\n) -> InputPredictions:\n    result = extract(\n        preprocessed,\n        img_path,\n        step_size=320,\n        use_cache=enable_cache,\n        use_gpu_inference=segnet_use_gpu,\n    )\n    original_image = cv2.resize(original, (result.staff.shape[1], result.staff.shape[0]))\n    preprocessed_image = cv2.resize(preprocessed, (result.staff.shape[1], result.staff.shape[0]))\n    return InputPredictions(\n        original=original_image,\n        preprocessed=preprocessed_image,\n        notehead=result.notehead.astype(np.uint8),\n        symbols=result.symbols.astype(np.uint8),\n        staff=result.staff.astype(np.uint8),\n        clefs_keys=result.clefs_keys.astype(np.uint8),\n        stems_rest=result.stems_rests.astype(np.uint8),\n    )\n\n\n'''
if text.count(predictions_block) != 1:
    raise SystemExit("get_predictions anchor changed")
text = text.replace(predictions_block, "")

symbols_block = '''def predict_symbols(debug: Debug, predictions: InputPredictions) -> PredictedSymbols:\n    eprint("Creating bounds for noteheads")\n    noteheads = create_bounding_ellipses(predictions.notehead, min_size=(4, 4))\n    eprint("Creating bounds for staff_fragments")\n    staff_fragments = create_rotated_bounding_boxes(\n        predictions.staff, skip_merging=True, min_size=(5, 1), max_size=(10000, 100)\n    )\n\n    eprint("Creating bounds for clefs_keys")\n    clefs_keys = create_rotated_bounding_boxes(\n        predictions.clefs_keys, min_size=(20, 40), max_size=(1000, 1000)\n    )\n    eprint("Creating bounds for stems_rest")\n    stems_rest = create_rotated_bounding_boxes(predictions.stems_rest)\n    eprint("Creating bounds for bar_lines")\n    bar_line_img = prepare_bar_line_image(predictions.stems_rest)\n    debug.write_threshold_image("bar_line_img", bar_line_img)\n    bar_lines = create_rotated_bounding_boxes(bar_line_img, skip_merging=True, min_size=(1, 5))\n\n    return PredictedSymbols(noteheads, staff_fragments, clefs_keys, stems_rest, bar_lines)\n\n\n'''
if text.count(symbols_block) != 1:
    raise SystemExit("predict_symbols anchor changed")
text = text.replace(symbols_block, "")

needle = '''    parser.add_argument(\n        "--system-pad",\n        type=float,\n        default=DEFAULT_SYSTEM_PAD,\n        help=f"Page-height padding on each system edge (default {DEFAULT_SYSTEM_PAD})",\n    )\n\n    args = parser.parse_args()\n'''
replacement = '''    parser.add_argument(\n        "--system-pad",\n        type=float,\n        default=DEFAULT_SYSTEM_PAD,\n        help=f"Page-height padding on each system edge (default {DEFAULT_SYSTEM_PAD})",\n    )\n    parser.add_argument(\n        "--find-system-bounds",\n        action="store_true",\n        help="Propose printed-system bounds for a PDF as JSON without decoding music",\n    )\n    parser.add_argument(\n        "--system-page",\n        type=int,\n        help="With --find-system-bounds, propose only this 1-based PDF page",\n    )\n\n    args = parser.parse_args()\n'''
if text.count(needle) != 1:
    raise SystemExit("system-pad parser anchor changed")
text = text.replace(needle, replacement)

needle = '''        if args.system_index is not None and not args.system_bounds:\n            parser.error("--system-index requires --system-bounds")\n        if args.system_bounds and (not args.image or not args.image.lower().endswith(".pdf")):\n            parser.error("--system-bounds requires a PDF input")\n        if args.system_dpi < 1:\n            parser.error("--system-dpi must be at least 1")\n'''
replacement = '''        if args.system_index is not None and not args.system_bounds:\n            parser.error("--system-index requires --system-bounds")\n        if args.system_page is not None and not args.find_system_bounds:\n            parser.error("--system-page requires --find-system-bounds")\n        if args.system_bounds and args.find_system_bounds:\n            parser.error("--system-bounds and --find-system-bounds are mutually exclusive")\n        if args.system_bounds and (not args.image or not args.image.lower().endswith(".pdf")):\n            parser.error("--system-bounds requires a PDF input")\n        if args.find_system_bounds and (not args.image or not args.image.lower().endswith(".pdf")):\n            parser.error("--find-system-bounds requires a PDF input")\n        if args.system_page is not None and args.system_page < 1:\n            parser.error("--system-page must be at least 1")\n        if args.system_dpi < 1:\n            parser.error("--system-dpi must be at least 1")\n'''
if text.count(needle) != 1:
    raise SystemExit("argument-validation anchor changed")
text = text.replace(needle, replacement)

needle = '''    if not args.image:\n        eprint("No image provided")\n        parser.print_help()\n        sys.exit(1)\n    elif os.path.isfile(args.image):\n'''
replacement = '''    if args.find_system_bounds:\n        try:\n            proposal = find_system_bounds(\n                args.image,\n                use_gpu=segnet_use_gpu,\n                dpi=args.system_dpi,\n                log=eprint,\n                page=args.system_page,\n            )\n        except (OSError, ValueError) as error:\n            eprint(str(error))\n            sys.exit(2)\n        json.dump(bounds_payload(proposal), sys.stdout)\n        sys.stdout.write("\\n")\n        return\n\n    if not args.image:\n        eprint("No image provided")\n        parser.print_help()\n        sys.exit(1)\n    elif os.path.isfile(args.image):\n'''
if text.count(needle) != 1:
    raise SystemExit("processing-dispatch anchor changed")
text = text.replace(needle, replacement)

path.write_text(text)
