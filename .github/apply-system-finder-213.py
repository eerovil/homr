from pathlib import Path

path = Path("homr/main.py")
text = path.read_text()

needle = '''    parser.add_argument(\n        "--system-pad",\n        type=float,\n        default=DEFAULT_SYSTEM_PAD,\n        help=f"Page-height padding on each system edge (default {DEFAULT_SYSTEM_PAD})",\n    )\n\n    args = parser.parse_args()\n'''
replacement = '''    parser.add_argument(\n        "--system-pad",\n        type=float,\n        default=DEFAULT_SYSTEM_PAD,\n        help=f"Page-height padding on each system edge (default {DEFAULT_SYSTEM_PAD})",\n    )\n    parser.add_argument(\n        "--find-system-bounds",\n        action="store_true",\n        help="Propose printed-system bounds for a PDF as JSON without decoding music",\n    )\n\n    args = parser.parse_args()\n'''
if text.count(needle) != 1:
    raise SystemExit("system-pad parser anchor changed")
text = text.replace(needle, replacement)

needle = '''        if args.system_index is not None and not args.system_bounds:\n            parser.error("--system-index requires --system-bounds")\n        if args.system_bounds and (not args.image or not args.image.lower().endswith(".pdf")):\n            parser.error("--system-bounds requires a PDF input")\n        if args.system_dpi < 1:\n            parser.error("--system-dpi must be at least 1")\n'''
replacement = '''        if args.system_index is not None and not args.system_bounds:\n            parser.error("--system-index requires --system-bounds")\n        if args.system_bounds and args.find_system_bounds:\n            parser.error("--system-bounds and --find-system-bounds are mutually exclusive")\n        if args.system_bounds and (not args.image or not args.image.lower().endswith(".pdf")):\n            parser.error("--system-bounds requires a PDF input")\n        if args.find_system_bounds and (not args.image or not args.image.lower().endswith(".pdf")):\n            parser.error("--find-system-bounds requires a PDF input")\n        if args.system_dpi < 1:\n            parser.error("--system-dpi must be at least 1")\n'''
if text.count(needle) != 1:
    raise SystemExit("argument-validation anchor changed")
text = text.replace(needle, replacement)

needle = '''    if not args.image:\n        eprint("No image provided")\n        parser.print_help()\n        sys.exit(1)\n    elif os.path.isfile(args.image):\n'''
replacement = '''    if args.find_system_bounds:\n        from homr.system_finder import bounds_payload, find_system_bounds\n\n        try:\n            proposal = find_system_bounds(\n                args.image, use_gpu=segnet_use_gpu, dpi=args.system_dpi, log=eprint\n            )\n        except (OSError, ValueError) as error:\n            eprint(str(error))\n            sys.exit(2)\n        json.dump(bounds_payload(proposal), sys.stdout)\n        sys.stdout.write("\\n")\n        return\n\n    if not args.image:\n        eprint("No image provided")\n        parser.print_help()\n        sys.exit(1)\n    elif os.path.isfile(args.image):\n'''
if text.count(needle) != 1:
    raise SystemExit("processing-dispatch anchor changed")
text = text.replace(needle, replacement)

path.write_text(text)
