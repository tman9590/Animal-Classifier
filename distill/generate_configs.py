#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from catalog import read_catalog


ROOT = Path(__file__).parents[1]
MODEL_FILES = {
    "coreml": [
        "north-carolina-wildlife.mlpackage/Data/com.apple.CoreML/model.mlmodel",
        "north-carolina-wildlife.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
        "north-carolina-wildlife.mlpackage/Manifest.json",
    ],
    "openvino": [
        "north-carolina-wildlife.xml",
        "north-carolina-wildlife.bin",
    ],
}


def write_config(directory: Path, files: list[str], labels: list[str], image_size: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "input_shape": [1, 3, image_size, image_size],
        "model": "yolov9",
        "files": files,
        "labels": {str(index): label for index, label in enumerate(labels)},
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Scrypted backend config.json files")
    parser.add_argument("--catalog", type=Path, default=ROOT / "species" / "north-carolina.json")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--backends", nargs="+", choices=sorted(MODEL_FILES), default=sorted(MODEL_FILES))
    args = parser.parse_args()

    metadata, _ = read_catalog(args.catalog)
    for backend in args.backends:
        write_config(args.models_dir / backend, MODEL_FILES[backend], metadata["labels"], args.image_size)


if __name__ == "__main__":
    main()
