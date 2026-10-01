#!/usr/bin/env python3
"""Validate SpeciesNet Scrypted exports against the source checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from speciesnet_scrypted import DEFAULT_HANDLE, IMAGE_SIZE, build_adapter, load_checkpoint


ROOT = Path(__file__).parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def probabilities(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(axis=1, keepdims=True)


def validate_config(config_path: Path, labels: list[str]) -> tuple[dict, list[Path]]:
    config = json.loads(config_path.read_text())
    expected = {
        "model": "resnet",
        "input_shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{config_path.name}: invalid {key}: {config.get(key)!r}")
    if list(config["labels"].values()) != labels:
        raise ValueError(f"{config_path.name}: labels differ from the source checkpoint")
    files = [config_path.parent / name for name in config["files"]]
    for path in files:
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing or empty configured artifact: {path}")
    return config, files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handle", default=DEFAULT_HANDLE)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--backends", nargs="+", choices=["coreml", "ncnn", "onnx", "openvino"], default=["coreml", "ncnn", "onnx", "openvino"])
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--report", type=Path, default=ROOT / "work" / "speciesnet-validation.json")
    args = parser.parse_args()

    import torch

    _, info, labels, checkpoint = load_checkpoint(args.handle, args.model_dir)
    reference_model = build_adapter(checkpoint).eval()
    runners = {}
    reports = {}

    for backend in args.backends:
        root = args.models / backend
        _, files = validate_config(root / "config.json", labels)
        reports[backend] = {
            "artifacts": {
                str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in files
            },
            "samples": [],
        }
        if backend == "openvino":
            import openvino as ov

            xml = next(path for path in files if path.suffix == ".xml")
            compiled = ov.Core().compile_model(str(xml), "CPU")
            runners[backend] = lambda tensor, model=compiled: model(tensor)[0]
        elif backend == "coreml":
            manifest = next(path for path in files if path.name == "Manifest.json")
            if sys.platform == "darwin":
                import coremltools as ct

                model = ct.models.MLModel(str(manifest.parent))
                spec = model.get_spec()
            else:
                from coremltools.proto import Model_pb2

                manifest_data = json.loads(manifest.read_text())
                root_id = manifest_data["rootModelIdentifier"]
                model_relative = manifest_data["itemInfoEntries"][root_id]["path"]
                model_path = manifest.parent / "Data" / model_relative
                spec = Model_pb2.Model()
                spec.ParseFromString(model_path.read_bytes())
            input_name = spec.description.input[0].name
            output_name = spec.description.output[0].name
            if input_name != "input" or output_name != "logits":
                raise ValueError(
                    f"coreml: unexpected model I/O: {input_name!r}, {output_name!r}"
                )
            input_shape = list(spec.description.input[0].type.multiArrayType.shape)
            output_shape = list(spec.description.output[0].type.multiArrayType.shape)
            if input_shape != [1, 3, 480, 480] or output_shape != [1, 2498]:
                raise ValueError(
                    f"coreml: unexpected tensor shapes: {input_shape}, {output_shape}"
                )
            if sys.platform == "darwin":
                reports[backend]["validation"] = "source/export inference parity"
                runners[backend] = lambda tensor, model=model, name=input_name: next(
                    iter(model.predict({name: tensor}).values())
                )
            else:
                reports[backend]["validation"] = (
                    "artifact integrity and Core ML specification; inference requires macOS"
                )
        elif backend == "onnx":
            import onnxruntime as ort

            model_path = next(path for path in files if path.suffix == ".onnx")
            model = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            input_name = model.get_inputs()[0].name
            runners[backend] = lambda tensor, model=model, name=input_name: model.run(None, {name: tensor})[0]
        elif backend == "ncnn":
            import ncnn

            param = next(path for path in files if path.name.endswith(".ncnn.param"))
            binary = next(path for path in files if path.name.endswith(".ncnn.bin"))
            model = ncnn.Net()
            model.load_param(str(param))
            model.load_model(str(binary))

            def run_ncnn(tensor, model=model):
                extractor = model.create_extractor()
                extractor.input("in0", ncnn.Mat(tensor.squeeze(0)))
                _, output = extractor.extract("out0")
                return np.asarray(output)[None]

            runners[backend] = run_ncnn

        if backend != "coreml":
            reports[backend]["validation"] = "source/export inference parity"

    paths = sorted(
        path
        for path in args.images.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        raise ValueError(f"No validation images under {args.images}")
    indexes = np.linspace(0, len(paths) - 1, min(args.samples, len(paths)), dtype=int)
    for index in indexes:
        path = paths[index]
        with Image.open(path) as source:
            image = source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        tensor = np.ascontiguousarray(np.asarray(image).transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        with torch.inference_mode():
            reference = reference_model(torch.from_numpy(tensor)).numpy()
        reference_probabilities = probabilities(reference)
        reference_top5 = np.argsort(reference_probabilities[0])[-5:][::-1]
        for backend, run in runners.items():
            started = time.perf_counter()
            output = np.asarray(run(tensor))
            elapsed = time.perf_counter() - started
            if output.shape != reference.shape or not np.isfinite(output).all():
                raise ValueError(f"{backend}: invalid output {output.shape}")
            output_probabilities = probabilities(output)
            output_top5 = np.argsort(output_probabilities[0])[-5:][::-1]
            top1_match = bool(output_top5[0] == reference_top5[0])
            source_top1_probability = float(reference_probabilities[0, reference_top5[0]])
            top5_overlap = len(set(output_top5) & set(reference_top5))
            max_probability_error = float(np.max(np.abs(output_probabilities - reference_probabilities)))
            # Scrypted does not emit classifier results below its default 0.50 threshold,
            # where tiny FP16 differences may legitimately swap nearly tied classes.
            if (not top1_match and source_top1_probability >= 0.50) or top5_overlap < 4 or max_probability_error > 0.10:
                raise ValueError(
                    f"{backend} parity failed for {path.name}: top1={top1_match}, "
                    f"source confidence={source_top1_probability:.4f}, top5 overlap={top5_overlap}, "
                    f"probability error={max_probability_error:.4f}"
                )
            reports[backend]["samples"].append(
                {
                    "image_sha256": sha256(path),
                    "latency_seconds": elapsed,
                    "source_top1": labels[int(reference_top5[0])],
                    "export_top1": labels[int(output_top5[0])],
                    "source_top1_probability": source_top1_probability,
                    "top1_match": top1_match,
                    "top5_overlap": top5_overlap,
                    "max_probability_error": max_probability_error,
                }
            )

    result = {
        "passed": True,
        "source_version": info["version"],
        "class_count": len(labels),
        "input_shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
        "sample_count": len(indexes),
        "backends": reports,
        "scope": "Artifact integrity and source/export inference parity; not site-specific species accuracy.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Validated {', '.join(args.backends)} on {len(indexes)} images: {args.report}")


if __name__ == "__main__":
    main()
