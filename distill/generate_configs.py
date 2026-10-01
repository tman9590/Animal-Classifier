#!/usr/bin/env python3
"""Generate Scrypted's backend-specific manifests from one canonical config."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
MODEL_BASENAME = "speciesnet-v4.0.3a"
MODEL_FILES = {
    "coreml": [
        f"{MODEL_BASENAME}.mlpackage/Data/com.apple.CoreML/model.mlmodel",
        f"{MODEL_BASENAME}.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
        f"{MODEL_BASENAME}.mlpackage/Manifest.json",
    ],
    "ncnn": [
        f"{MODEL_BASENAME}.ncnn.bin",
        f"{MODEL_BASENAME}.ncnn.param",
    ],
    "onnx": [f"{MODEL_BASENAME}.onnx"],
    "openvino": [
        f"{MODEL_BASENAME}.xml",
        f"{MODEL_BASENAME}.bin",
    ],
}
COMMON_KEYS = ("input_shape", "model", "mean", "std", "labels")


def common_config(labels: dict[str, str]) -> dict:
    return {
        "input_shape": [1, 3, 480, 480],
        "model": "resnet",
        # Each Scrypted backend first converts uint8 RGB to NCHW float [0, 1].
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "labels": labels,
    }


def canonical_config(labels: dict[str, str]) -> dict:
    config = common_config(labels)
    config["backends"] = {
        backend: {"files": files} for backend, files in MODEL_FILES.items()
    }
    return config


def validate_canonical(config: dict) -> None:
    if config.get("input_shape") != [1, 3, 480, 480]:
        raise ValueError("SpeciesNet must use a 480 x 480 NCHW input")
    if config.get("model") != "resnet":
        raise ValueError("Scrypted must parse SpeciesNet as a resnet classifier")
    labels = config.get("labels", {})
    if len(labels) != 2498 or list(labels) != [str(i) for i in range(2498)]:
        raise ValueError("Expected the ordered 2,498-label SpeciesNet taxonomy")
    if config.get("backends") != {
        backend: {"files": files} for backend, files in MODEL_FILES.items()
    }:
        raise ValueError("Canonical backend file map is missing or out of date")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_projections(config: dict, models: Path, verify_files: bool = True) -> None:
    validate_canonical(config)
    common = {key: config[key] for key in COMMON_KEYS}
    for backend, metadata in config["backends"].items():
        directory = models / backend
        files = metadata["files"]
        if verify_files:
            missing = [name for name in files if not (directory / name).is_file()]
            if missing:
                raise ValueError(f"{backend}: missing artifacts: {', '.join(missing)}")
        write_json(directory / "config.json", {**common, "files": files})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument(
        "--labels-from",
        type=Path,
        help="Bootstrap the canonical manifest from an existing Scrypted config",
    )
    parser.add_argument("--no-verify-files", action="store_true")
    args = parser.parse_args()

    if args.labels_from:
        source = json.loads(args.labels_from.read_text())
        config = canonical_config(source["labels"])
        write_json(args.config, config)
    else:
        config = json.loads(args.config.read_text())
    write_projections(config, args.models, verify_files=not args.no_verify_files)


if __name__ == "__main__":
    main()
