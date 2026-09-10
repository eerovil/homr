from pathlib import Path

path = Path("homr/main.py")
text = path.read_text()

header = '''def download_weights(segnet_use_gpu: bool, transformer_use_gpu: bool, coreml_encoder: bool) -> None:\n'''
replacement = '''def download_weights(\n    segnet_use_gpu: bool,\n    transformer_use_gpu: bool,\n    coreml_encoder: bool,\n    include_transformer: bool = True,\n) -> None:\n'''
if text.count(header) != 1:
    raise SystemExit("download_weights header changed")
text = text.replace(header, replacement)

models = '''    models = [segnet_path_onnx_fp16 if segnet_use_gpu else segnet_path_onnx]\n    if transformer_use_gpu:\n        # CUDA runs the whole transformer on the fp16 models.\n        models.append(default_config.filepaths.encoder_path_fp16)\n        models.append(default_config.filepaths.decoder_path_fp16)\n    else:\n        # On the CPU EP the fp32 models are faster, and the CoreML EP cannot run\n        # the decoder, so the decoder always uses fp32. The CoreML encoder, when\n        # enabled, uses the fp16 encoder instead of the fp32 one.\n        if coreml_encoder:\n            models.append(default_config.filepaths.encoder_path_fp16)\n        else:\n            models.append(default_config.filepaths.encoder_path)\n        models.append(default_config.filepaths.decoder_path)\n'''
replacement = '''    models = [segnet_path_onnx_fp16 if segnet_use_gpu else segnet_path_onnx]\n    if include_transformer:\n        if transformer_use_gpu:\n            # CUDA runs the whole transformer on the fp16 models.\n            models.append(default_config.filepaths.encoder_path_fp16)\n            models.append(default_config.filepaths.decoder_path_fp16)\n        else:\n            # On the CPU EP the fp32 models are faster, and the CoreML EP cannot run\n            # the decoder, so the decoder always uses fp32. The CoreML encoder, when\n            # enabled, uses the fp16 encoder instead of the fp32 one.\n            if coreml_encoder:\n                models.append(default_config.filepaths.encoder_path_fp16)\n            else:\n                models.append(default_config.filepaths.encoder_path)\n            models.append(default_config.filepaths.decoder_path)\n'''
if text.count(models) != 1:
    raise SystemExit("download_weights model block changed")
text = text.replace(models, replacement)

normal_download = '''    download_weights(segnet_use_gpu, transformer_use_gpu, coreml_encoder)\n    if args.init:\n'''
early_finder = '''    if args.find_system_bounds:\n        # This mode stops before transformer decoding, so do not require or\n        # download encoder/decoder models just to locate printed systems.\n        download_weights(segnet_use_gpu, transformer_use_gpu, coreml_encoder, False)\n        if args.debug:\n            ort.set_default_logger_severity(2)\n        else:\n            ort.set_default_logger_severity(3)\n        try:\n            proposal = find_system_bounds(\n                args.image,\n                use_gpu=segnet_use_gpu,\n                dpi=args.system_dpi,\n                log=eprint,\n                page=args.system_page,\n            )\n        except (OSError, ValueError) as error:\n            eprint(str(error))\n            sys.exit(2)\n        json.dump(bounds_payload(proposal), sys.stdout)\n        sys.stdout.write("\\n")\n        return\n\n    download_weights(segnet_use_gpu, transformer_use_gpu, coreml_encoder)\n    if args.init:\n'''
if text.count(normal_download) != 1:
    raise SystemExit("normal model-download dispatch changed")
text = text.replace(normal_download, early_finder)

late_finder = '''    if args.find_system_bounds:\n        try:\n            proposal = find_system_bounds(\n                args.image,\n                use_gpu=segnet_use_gpu,\n                dpi=args.system_dpi,\n                log=eprint,\n                page=args.system_page,\n            )\n        except (OSError, ValueError) as error:\n            eprint(str(error))\n            sys.exit(2)\n        json.dump(bounds_payload(proposal), sys.stdout)\n        sys.stdout.write("\\n")\n        return\n\n'''
if text.count(late_finder) != 1:
    raise SystemExit("late system-finder dispatch changed")
text = text.replace(late_finder, "")

path.write_text(text)
