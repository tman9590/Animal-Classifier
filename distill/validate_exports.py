#!/usr/bin/env python3
"""Run exported artifacts with Scrypted's input conventions and compare raw YOLO outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parents[1]


def validate_output(output, classes: int, size: int):
    output = np.asarray(output)
    anchors = sum((size // stride) ** 2 for stride in (8, 16, 32))
    if output.shape != (1, classes + 4, anchors):
        raise ValueError(f'Unexpected YOLO tensor: {output.shape}; expected {(1, classes + 4, anchors)}')
    if not np.isfinite(output).all():
        raise ValueError('Non-finite model output')
    if output[:, 4:].min() < -1e-6 or output[:, 4:].max() > 1.000001:
        raise ValueError('Class scores are not probabilities')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weights', type=Path)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--models', type=Path, default=ROOT / 'models')
    parser.add_argument('--backends', nargs='+', choices=['onnx', 'openvino', 'coreml', 'ncnn'],
                        default=['onnx', 'openvino', 'coreml', 'ncnn'])
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--report', type=Path, default=ROOT / 'work/export-validation.json')
    args = parser.parse_args()
    import torch
    from ultralytics import YOLO

    model = YOLO(args.weights).model.cpu().float().eval()
    labels = [model.names[i] for i in range(len(model.names))]
    reports = {}
    runners = {}
    size = None
    for backend in args.backends:
        root = args.models / backend
        config = json.loads((root / 'config.json').read_text())
        if config['model'] != 'yolov9' or list(config['labels'].values()) != labels:
            raise ValueError(f'{backend}: labels/parser disagree with checkpoint')
        size = config['input_shape'][2]
        if config['input_shape'] != [1, 3, size, size]:
            raise ValueError('Only square batch-one RGB models are supported')
        files = [root / name for name in config['files']]
        for path in files:
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f'Missing/empty configured artifact: {path}')
        reports[backend] = {'artifacts': {str(p.relative_to(root)): {
            'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()
        } for p in files}, 'samples': []}
        if backend == 'coreml':
            import coremltools as ct
            package = next(p.parent for p in files if p.name == 'Manifest.json')
            coreml = ct.models.MLModel(str(package))
            feature = coreml.get_spec().description.input[0]
            if feature.type.WhichOneof('Type') != 'imageType':
                raise ValueError('Scrypted config without mean/std requires a CoreML image input')
            def coreml_run(image, tensor, cm=coreml, key=feature.name):
                return next(iter(cm.predict({key: image}).values()))
            runners[backend] = coreml_run
        elif backend == 'openvino':
            import openvino as ov
            compiled = ov.Core().compile_model(str(next(p for p in files if p.suffix == '.xml')), 'CPU')
            def openvino_run(image, tensor, cm=compiled):
                return cm(tensor)[0]
            runners[backend] = openvino_run
        elif backend == 'onnx':
            import onnxruntime as ort
            session = ort.InferenceSession(str(next(p for p in files if p.suffix == '.onnx')),
                                           providers=['CPUExecutionProvider'])
            input_name = session.get_inputs()[0].name
            def onnx_run(image, tensor, cm=session, key=input_name):
                return cm.run(None, {key: tensor})[0]
            runners[backend] = onnx_run
        else:
            import ncnn
            network = ncnn.Net()
            network.load_param(str(next(p for p in files if p.suffix == '.param')))
            network.load_model(str(next(p for p in files if p.suffix == '.bin')))
            def ncnn_run(image, tensor, net=network):
                with net.create_extractor() as extractor:
                    extractor.input('in0', ncnn.Mat(tensor[0]).clone())
                    status, output = extractor.extract('out0')
                    if status != 0:
                        raise RuntimeError(f'NCNN inference failed with status {status}')
                    return np.asarray(output)[None]
            runners[backend] = ncnn_run
    paths = sorted(args.images.rglob('*.jpg'))
    if not paths:
        raise ValueError('No validation images')
    indexes = np.linspace(0, len(paths) - 1, min(args.samples, len(paths)), dtype=int)
    for index in indexes:
        path = paths[index]
        with Image.open(path) as original:
            image = original.convert('RGB').resize((size, size))
        tensor = np.ascontiguousarray(np.asarray(image).transpose(2, 0, 1)[None].astype(np.float32) / 255)
        with torch.inference_mode():
            raw = model(torch.from_numpy(tensor))
            reference = validate_output((raw[0] if isinstance(raw, tuple) else raw).numpy(), len(labels), size)
        for backend, run in runners.items():
            start = time.perf_counter()
            output = validate_output(run(image, tensor), len(labels), size)
            elapsed = time.perf_counter() - start
            score_error = float(np.max(np.abs(output[:, 4:] - reference[:, 4:])))
            box_error = float(np.max(np.abs(output[:, :4] - reference[:, :4])))
            reports[backend]['samples'].append(dict(image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                latency_seconds=elapsed, max_score_error=score_error, max_box_error_pixels=box_error))
            if score_error > .03 or box_error > 4:
                raise ValueError(f'{backend} parity failed: scores={score_error}, box pixels={box_error}')
    result = dict(passed=True, class_count=len(labels), input_shape=[1, 3, size, size], backends=reports,
                  scope='Backend runtime and raw-output parity; not camera accuracy or server integration')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(f"Validated {', '.join(args.backends)} artifacts on {len(indexes)} images: {args.report}")


if __name__ == '__main__':
    main()
