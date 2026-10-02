#!/usr/bin/env python3
"""Export Google's SpeciesNet crop classifier for Scrypted classifier backends."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterable

from generate_configs import MODEL_FILES, canonical_config, write_json, write_projections


ROOT = Path(__file__).parents[1]
DEFAULT_HANDLE = "google/speciesnet/pyTorch/v4.0.3a/1"
IMAGE_SIZE = 480
MODEL_BASENAME = "speciesnet-v4.0.3a-north-carolina"
REGION_COUNTRY = "USA"
REGION_ADMIN1 = "NC"
SAFETY_LABELS = {"blank", "human", "vehicle"}
LIVESTOCK_LABELS = {
    "domestic cattle",
    "domestic chicken",
    "domestic donkey",
    "domestic goat",
    "domestic goose",
    "domestic horse",
    "domestic mule",
    "domestic pig",
    "domestic sheep",
    "domestic water buffalo",
    "helmeted guineafowl",
    "llama",
}


def parse_speciesnet_labels(lines: Iterable[str]) -> list[str]:
    """Convert SpeciesNet taxonomy rows into unique, readable Scrypted labels."""
    rows: list[tuple[str, str, str]] = []
    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split(";")
        if len(fields) != 7:
            raise ValueError(f"SpeciesNet label row {number} has {len(fields)} fields, expected 7")
        taxonomy_id, animal_class, order, family, genus, species, common_name = fields
        scientific = " ".join(part for part in (genus, species) if part)
        fallback = scientific or family or order or animal_class or f"taxonomy {fields[0]}"
        rows.append((common_name or fallback, scientific, taxonomy_id))

    counts = Counter(label.casefold() for label, _, _ in rows)
    labels = [
        f"{label} ({scientific})" if counts[label.casefold()] > 1 and scientific else label
        for label, scientific, _ in rows
    ]
    candidate_counts = Counter(label.casefold() for label in labels)
    labels = [
        f"{label} [{taxonomy_id[:8]}]" if candidate_counts[label.casefold()] > 1 else label
        for label, (_, _, taxonomy_id) in zip(labels, rows)
    ]
    return labels


def load_checkpoint(handle: str, model_dir: Path | None):
    import kagglehub
    import torch

    root = model_dir or Path(kagglehub.model_download(handle))
    info = json.loads((root / "info.json").read_text())
    labels = parse_speciesnet_labels((root / info["classifier_labels"]).read_text().splitlines())
    model = torch.load(root / info["classifier"], map_location="cpu", weights_only=False).eval()
    return root, info, labels, model


def should_geofence(rule: dict | None, country: str, admin1: str) -> bool:
    """Mirror SpeciesNet's country/admin1 allow/block geofence semantics."""
    if not rule:
        return False
    allow = rule.get("allow")
    if allow is not None:
        if country not in allow:
            return True
        states = allow[country]
        if states and admin1 not in states:
            return True
    block = rule.get("block")
    if block is not None and country in block:
        states = block[country]
        if not states or admin1 in states:
            return True
    return False


def select_regional_indices(lines: Iterable[str], geofence: dict, country: str, admin1: str) -> list[int]:
    """Select locally plausible species, their ancestors, and safety labels."""
    rows = []
    for index, raw in enumerate(lines):
        fields = raw.strip().split(";")
        if len(fields) != 7:
            raise ValueError(f"SpeciesNet label row {index + 1} has {len(fields)} fields, expected 7")
        rows.append(fields)

    species = set()
    for fields in rows:
        taxonomy = ";".join(fields[1:6])
        rule = geofence.get(taxonomy)
        # A state-scoped model should fail closed when the upstream geofence
        # has no range rule for a species.
        locally_allowed = rule is not None and not should_geofence(rule, country, admin1)
        livestock = fields[6].casefold() in LIVESTOCK_LABELS
        if fields[5] and (locally_allowed or livestock):
            species.add(taxonomy)
    selected = []
    for index, fields in enumerate(rows):
        taxonomy = ";".join(fields[1:6])
        common_name = fields[6].casefold()
        if fields[5]:
            keep = taxonomy in species
        else:
            prefix = taxonomy.rstrip(";")
            keep = common_name in SAFETY_LABELS or bool(prefix) and any(
                item == prefix or item.startswith(prefix + ";") for item in species
            )
        if keep:
            selected.append(index)
    return selected


def load_regional_selection(source: Path, info: dict, labels: list[str]) -> tuple[list[int], list[str]]:
    raw_lines = (source / info["classifier_labels"]).read_text().splitlines()
    geofence = json.loads((source / info["geofence"]).read_text())
    indices = select_regional_indices(raw_lines, geofence, REGION_COUNTRY, REGION_ADMIN1)
    return indices, [labels[index] for index in indices]


def prune_classifier_head(model, indices: list[int]):
    """Prune the source dense head so every backend natively emits the region."""
    import torch

    selected = torch.tensor(indices, dtype=torch.long)
    initializers = model.initializers
    initializers.onnx_initializer_135 = torch.index_select(
        initializers.onnx_initializer_135, 1, selected
    ).contiguous()
    initializers.onnx_initializer_136 = torch.index_select(
        initializers.onnx_initializer_136, 0, selected
    ).contiguous()
    return model


def build_adapter(model):
    import torch

    class NchwSpeciesNet(torch.nn.Module):
        """Adapt Scrypted's NCHW float crop to SpeciesNet's NHWC input."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, image):
            return self.inner(image.permute(0, 2, 3, 1))

    return NchwSpeciesNet(model).eval()


def export_onnx(model, example, destination: Path) -> None:
    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        destination,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes=None,
        dynamo=False,
    )


def export_openvino(onnx_path: Path, directory: Path, compress: bool) -> list[Path]:
    import openvino as ov

    directory.mkdir(parents=True, exist_ok=True)
    xml = directory / f"{MODEL_BASENAME}.xml"
    model = ov.convert_model(str(onnx_path))
    if compress:
        import nncf

        model = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM)
    ov.save_model(model, xml, compress_to_fp16=not compress)
    return [xml, xml.with_suffix(".bin")]


def export_coreml(model, example, directory: Path, compress: bool) -> list[Path]:
    import coremltools as ct
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{MODEL_BASENAME}.mlmodel"
    traced = torch.jit.trace(model, example, strict=True)
    converted = ct.convert(
        traced,
        convert_to="neuralnetwork",
        inputs=[ct.TensorType(name="input", shape=tuple(example.shape))],
        outputs=[ct.TensorType(name="logits")],
    )
    if compress:
        from coremltools.models.neural_network.quantization_utils import quantize_weights

        converted = quantize_weights(converted, nbits=8)
    converted.save(str(destination))
    return [destination]


def export_ncnn(onnx_path: Path, directory: Path) -> list[Path]:
    """Convert ONNX with pnnx using Scrypted's in0/out0 convention."""
    directory.mkdir(parents=True, exist_ok=True)
    command = shutil.which("pnnx")
    if not command:
        candidate = Path(__import__("sys").executable).with_name("pnnx.exe")
        command = str(candidate) if candidate.is_file() else None
    if not command:
        raise ValueError("pnnx executable was not found")
    subprocess.run(
        [command, str(onnx_path), f"inputshape=[1,3,{IMAGE_SIZE},{IMAGE_SIZE}]"],
        check=True,
    )
    generated_stem = onnx_path.stem.replace("-", "_")
    source_param = onnx_path.with_name(f"{generated_stem}.ncnn.param")
    source_binary = onnx_path.with_name(f"{generated_stem}.ncnn.bin")
    if not source_param.is_file() or not source_binary.is_file():
        raise ValueError("pnnx did not produce the expected NCNN artifacts")
    param = directory / f"{MODEL_BASENAME}.ncnn.param"
    binary = directory / f"{MODEL_BASENAME}.ncnn.bin"
    shutil.copy2(source_param, param)
    shutil.copy2(source_binary, binary)
    return [binary, param]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handle", default=DEFAULT_HANDLE)
    parser.add_argument("--model-dir", type=Path, help="Use an already downloaded SpeciesNet model directory")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--work", type=Path, default=ROOT / "work" / "speciesnet-export")
    parser.add_argument("--backends", nargs="+", choices=["onnx", "openvino", "coreml", "ncnn"], default=["onnx", "openvino", "coreml", "ncnn"])
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="Keep FP16 artifacts even when a file will exceed GitHub's 100 MB limit",
    )
    args = parser.parse_args()

    import torch

    source, info, source_labels, checkpoint = load_checkpoint(args.handle, args.model_dir)
    if len(source_labels) != 2498:
        raise ValueError(f"Unexpected SpeciesNet class count: {len(source_labels)}")
    indices, labels = load_regional_selection(source, info, source_labels)
    model = build_adapter(prune_classifier_head(checkpoint, indices))
    torch.manual_seed(7)
    example = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.inference_mode():
        output = model(example)
    if tuple(output.shape) != (1, len(labels)) or not torch.isfinite(output).all():
        raise ValueError(f"Invalid checkpoint output shape or values: {tuple(output.shape)}")

    args.work.mkdir(parents=True, exist_ok=True)
    onnx_path = args.work / f"{MODEL_BASENAME}.onnx"
    if any(backend in args.backends for backend in ("onnx", "openvino", "ncnn")):
        export_onnx(model, example, onnx_path)
    if "onnx" in args.backends:
        destination = args.models / "onnx" / onnx_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_path, destination)
    if "openvino" in args.backends:
        export_openvino(onnx_path, args.models / "openvino", not args.no_compression)
    if "coreml" in args.backends:
        export_coreml(model, example, args.models / "coreml", not args.no_compression)
    if "ncnn" in args.backends:
        export_ncnn(onnx_path, args.models / "ncnn")

    config = canonical_config({str(index): label for index, label in enumerate(labels)})
    write_json(args.config, config)
    # A GitHub repository URL is expanded by Scrypted to
    # models/<backend>/config.json, so every backend needs its own projection.
    write_projections(config, args.models)

    manifest = {
        "source": str(source),
        "version": info["version"],
        "handle": args.handle,
        "class_count": len(labels),
        "source_class_count": len(source_labels),
        "region": {"country": REGION_COUNTRY, "admin1": REGION_ADMIN1},
        "geofence": info["geofence"],
        "input_shape": list(example.shape),
        "backends": sorted(MODEL_FILES),
        "config": str(args.config),
        "weight_compression": "int8" if not args.no_compression else "fp16",
    }
    (args.work / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
