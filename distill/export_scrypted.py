#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from catalog import read_catalog
from generate_configs import write_config


ROOT = Path(__file__).parents[1]


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def export_backend(model: YOLO, backend: str, labels: list[str], image_size: int) -> None:
    output = ROOT / "models" / backend
    output.mkdir(parents=True, exist_ok=True)
    formats = {"onnx": "onnx", "openvino": "openvino", "coreml": "coreml", "ncnn": "ncnn"}
    exported = Path(model.export(format=formats[backend], imgsz=image_size, dynamic=False, simplify=True, nms=False, batch=1))

    if backend == "onnx":
        destination = output / "north-carolina-wildlife.onnx"
        copy_file(exported, destination)
        write_config(output, [destination.name], labels, image_size)
    elif backend == "openvino":
        xml = next(exported.glob("*.xml"))
        binary = xml.with_suffix(".bin")
        copy_file(xml, output / "north-carolina-wildlife.xml")
        copy_file(binary, output / "north-carolina-wildlife.bin")
        write_config(output, ["north-carolina-wildlife.xml", "north-carolina-wildlife.bin"], labels, image_size)
    elif backend == "coreml":
        destination = output / "north-carolina-wildlife.mlpackage"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(exported, destination)
        files = [str(path.relative_to(output)) for path in sorted(destination.rglob("*")) if path.is_file()]
        write_config(output, files, labels, image_size)
    elif backend == "ncnn":
        param = next(exported.glob("*.param"))
        binary = next(exported.glob("*.bin"))
        copy_file(param, output / "north-carolina-wildlife.ncnn.param")
        copy_file(binary, output / "north-carolina-wildlife.ncnn.bin")
        write_config(output, ["north-carolina-wildlife.ncnn.param", "north-carolina-wildlife.ncnn.bin"], labels, image_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained species detector for Scrypted model backends")
    parser.add_argument("weights", type=Path)
    parser.add_argument("--catalog", type=Path, default=ROOT / "species" / "north-carolina.json")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--backends", nargs="+", choices=["onnx", "openvino", "coreml", "ncnn"], default=["onnx", "openvino", "coreml", "ncnn"])
    args = parser.parse_args()
    metadata, _ = read_catalog(args.catalog)
    labels = metadata["labels"]
    model = YOLO(args.weights)
    trained_labels = [model.names[index] for index in range(len(model.names))]
    if trained_labels != labels:
        raise ValueError("Checkpoint class order does not match the catalog; refusing mislabeled export")
    failures = []
    for backend in args.backends:
        print(f"Exporting {backend}")
        try:
            export_backend(model, backend, labels, args.image_size)
        except Exception as error:
            failures.append((backend, error))
            print(f"Failed to export {backend}: {error}")
    if failures:
        summary = ", ".join(f"{backend}: {error}" for backend, error in failures)
        raise RuntimeError(f"One or more backend exports failed ({summary})")


if __name__ == "__main__":
    main()
