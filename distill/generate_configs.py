#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from catalog import read_catalog


ROOT = Path(__file__).parents[1]
ROOT_CONFIG = ROOT / "config.json"
MODEL_FILES = {
    "onnx": [
        "north-carolina-wildlife.onnx",
    ],
    "coreml": [
        "north-carolina-wildlife.mlpackage/Data/com.apple.CoreML/model.mlmodel",
        "north-carolina-wildlife.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
        "north-carolina-wildlife.mlpackage/Manifest.json",
    ],
    "openvino": [
        "north-carolina-wildlife.xml",
        "north-carolina-wildlife.bin",
    ],
    "ncnn": [
        "north-carolina-wildlife.ncnn.param",
        "north-carolina-wildlife.ncnn.bin",
    ],
}


def build_common_config(labels: list[str], image_size: int) -> dict:
    return {
        "input_shape": [1, 3, image_size, image_size],
        "model": "yolov9",
        "labels": {str(index): label for index, label in enumerate(labels)},
    }


def write_config(directory: Path, files: list[str], labels: list[str], image_size: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    config = build_common_config(labels, image_size)
    config["files"] = files
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def write_root_config(path: Path, labels: list[str], image_size: int) -> None:
    """Write the one human-maintained manifest for every Scrypted backend."""
    config = build_common_config(labels, image_size)
    config["backends"] = {
        backend: {"files": files}
        for backend, files in MODEL_FILES.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")


def read_root_config(path: Path, labels: list[str]) -> dict:
    config = json.loads(path.read_text())
    if config.get("model") != "yolov9":
        raise ValueError(f"{path}: model must be yolov9")
    if list(config.get("labels", {}).values()) != labels:
        raise ValueError(f"{path}: label order does not match the species catalog")
    if set(config.get("backends", {})) != set(MODEL_FILES):
        raise ValueError(f"{path}: backend set must be {sorted(MODEL_FILES)}")
    return config


def write_projection(directory: Path, backend: str, root_config: dict) -> None:
    config = {
        "input_shape": root_config["input_shape"],
        "model": root_config["model"],
        "files": root_config["backends"][backend]["files"],
        "labels": root_config["labels"],
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Scrypted backend config.json files")
    parser.add_argument("--catalog", type=Path, default=ROOT / "species" / "north-carolina.json")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--config", type=Path, default=ROOT_CONFIG)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--backends", nargs="+", choices=sorted(MODEL_FILES), default=sorted(MODEL_FILES))
    parser.add_argument(
        "--refresh-root",
        action="store_true",
        help="Rebuild the canonical config from the catalog and built-in backend filenames",
    )
    args = parser.parse_args()

    metadata, _ = read_catalog(args.catalog)
    if args.refresh_root or not args.config.exists():
        write_root_config(args.config, metadata["labels"], args.image_size)
    root_config = read_root_config(args.config, metadata["labels"])
    for backend in args.backends:
        write_projection(args.models_dir / backend, backend, root_config)


if __name__ == "__main__":
    main()
