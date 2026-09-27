#!/usr/bin/env python3
"""Import labeled, grayscale nighttime Caltech Camera Traps images for training."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from catalog import read_catalog
from label_images import prune_exact_duplicates, write_dataset_yaml


ROOT = Path(__file__).parents[1]
LICENSE = "Community Data License Agreement - Permissive 2.0"
SOURCE_PAGE = "https://lila.science/datasets/caltech-camera-traps"
DEFAULT_BASE_URL = "https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/"
CATEGORY_MAP = {
    "opossum": "Virginia Opossum",
    "raccoon": "Common Raccoon",
    "coyote": "Coyote",
    "bobcat": "Bobcat",
    "cat": "Domestic Cat",
    "dog": "Domestic Dog",
}


def is_night(datetime_text: str) -> bool:
    hour = int(datetime_text[11:13])
    return hour >= 20 or hour < 6


def image_metrics(path: Path) -> dict[str, float]:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize((128, 96)), dtype=np.float32)
    maximum = array.max(axis=2)
    minimum = array.min(axis=2)
    saturation = float(np.mean((maximum - minimum) / (maximum + 1.0)))
    channel_delta = float(np.mean(
        np.abs(array[:, :, 0] - array[:, :, 1])
        + np.abs(array[:, :, 1] - array[:, :, 2])
        + np.abs(array[:, :, 0] - array[:, :, 2])
    ) / 3.0)
    luminance = array.mean(axis=2)
    return {
        "mean_saturation": saturation,
        "mean_channel_delta": channel_delta,
        "mean_luminance": float(luminance.mean()),
        "luminance_stddev": float(luminance.std()),
    }


def is_ir_like(metrics: dict[str, float], maximum_saturation: float) -> bool:
    return (
        metrics["mean_saturation"] <= maximum_saturation
        and metrics["mean_luminance"] >= 8.0
        and metrics["luminance_stddev"] >= 5.0
    )


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "scrypted-animal-classifier/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    with Image.open(partial) as image:
        image.verify()
    partial.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata", type=Path,
        default=ROOT / "work" / "external" / "caltech-camera-traps" / "caltech_bboxes_20200316.json",
    )
    parser.add_argument(
        "--cache", type=Path,
        default=ROOT / "work" / "external" / "caltech-camera-traps" / "images",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "dataset-production")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-boxes-per-class", type=int, default=200)
    parser.add_argument("--candidate-multiplier", type=int, default=3)
    parser.add_argument("--maximum-saturation", type=float, default=0.16)
    args = parser.parse_args()

    payload = json.loads(args.metadata.read_text())
    categories = {item["id"]: item["name"] for item in payload["categories"]}
    images = {str(item["id"]): item for item in payload["images"]}
    annotations_by_image: dict[str, list[dict]] = defaultdict(list)
    for annotation in payload["annotations"]:
        annotations_by_image[str(annotation["image_id"])].append(annotation)

    _, species = read_catalog(ROOT / "species" / "north-carolina.json")
    labels = [item.label for item in species] + ["unknown"]
    label_indexes = {label: index for index, label in enumerate(labels)}

    existing_records = []
    used_sources = set()
    used_sequences = set()
    for audit_name in ("caltech-audit.jsonl", "caltech-ir-audit.jsonl"):
        audit_path = args.output / audit_name
        if not audit_path.exists():
            continue
        records = [json.loads(line) for line in audit_path.read_text().splitlines() if line]
        used_sources.update(record["source"] for record in records)
        used_sequences.update(record.get("sequence_id") for record in records)
        if audit_name == "caltech-ir-audit.jsonl":
            existing_records = records

    imported_counts = Counter(
        box["label"] for record in existing_records for box in record["boxes"]
    )
    candidates_by_class: dict[str, list[tuple[str, list[dict], str]]] = defaultdict(list)
    for image_id, image_annotations in annotations_by_image.items():
        image = images[image_id]
        source_names = {categories[item["category_id"]] for item in image_annotations}
        if not source_names or not source_names.issubset(CATEGORY_MAP) or not is_night(image["date_captured"]):
            continue
        sequence_id = image.get("seq_id", image_id)
        if image["file_name"] in used_sources or sequence_id in used_sequences:
            continue
        key = sorted(source_names)[0]
        candidates_by_class[key].append((image_id, image_annotations, sequence_id))

    candidates = []
    reserved_sequences = set()
    for source_name in sorted(CATEGORY_MAP):
        target = CATEGORY_MAP[source_name]
        remaining = max(0, args.max_boxes_per_class - imported_counts[target])
        ordered = sorted(
            candidates_by_class[source_name],
            key=lambda item: hashlib.sha256(item[0].encode()).hexdigest(),
        )
        for item in ordered:
            if item[2] in reserved_sequences:
                continue
            candidates.append(item)
            reserved_sequences.add(item[2])
            if sum(len(value) for _, value, _ in candidates if source_name in {categories[a["category_id"]] for a in value}) >= remaining * args.candidate_multiplier:
                break

    output_images = args.output / "images" / "train"
    output_labels = args.output / "labels" / "train"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    records = list(existing_records)
    rejected = Counter()
    failures = []
    for number, (image_id, image_annotations, sequence_id) in enumerate(candidates, 1):
        metadata = images[image_id]
        relative = metadata["file_name"].replace("\\", "/")
        destination = args.cache / relative
        url = urllib.parse.urljoin(args.base_url, urllib.parse.quote(relative))
        try:
            if not destination.exists():
                download(url, destination)
            metrics = image_metrics(destination)
        except Exception as error:
            failures.append({"image_id": image_id, "error": str(error)})
            continue
        if not is_ir_like(metrics, args.maximum_saturation):
            rejected["not_grayscale_ir_like"] += 1
            continue

        width, height = int(metadata["width"]), int(metadata["height"])
        lines = []
        details = []
        for annotation in image_annotations:
            source_name = categories[annotation["category_id"]]
            target_name = CATEGORY_MAP[source_name]
            if imported_counts[target_name] >= args.max_boxes_per_class:
                continue
            x, y, box_width, box_height = map(float, annotation["bbox"])
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(width), x + box_width), min(float(height), y + box_height)
            if x2 <= x1 or y2 <= y1:
                continue
            normalized = (
                (x1 + x2) / 2 / width,
                (y1 + y2) / 2 / height,
                (x2 - x1) / width,
                (y2 - y1) / height,
            )
            lines.append(f"{label_indexes[target_name]} " + " ".join(f"{value:.8f}" for value in normalized) + "\n")
            details.append({"source_label": source_name, "label": target_name, "bbox": [x1, y1, x2 - x1, y2 - y1]})
            imported_counts[target_name] += 1
        if not lines:
            rejected["class_quota_filled"] += 1
            continue

        stem = f"caltech-ir-{image_id}"
        shutil.copy2(destination, output_images / f"{stem}.jpg")
        (output_labels / f"{stem}.txt").write_text("".join(lines))
        records.append({
            "source": metadata["file_name"],
            "source_url": url,
            "source_page": SOURCE_PAGE,
            "rights_holder": metadata.get("rights_holder"),
            "license": LICENSE,
            "sequence_id": sequence_id,
            "datetime": metadata["date_captured"],
            "split": "train",
            "selection": "nighttime plus grayscale/low-saturation IR heuristic",
            "image_metrics": metrics,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "boxes": details,
        })
        if number % 100 == 0 or number == len(candidates):
            print(f"Checked {number}/{len(candidates)} nighttime candidates; imported {len(records)} IR-like images")

    (args.output / "caltech-ir-audit.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    (args.output / "caltech-ir-failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    write_dataset_yaml(args.output, labels)
    removed = prune_exact_duplicates(args.output)
    print(f"Caltech IR subset now contributes {len(records)} images and {sum(imported_counts.values())} boxes")
    print(f"Class counts: {dict(sorted(imported_counts.items()))}")
    print(f"Rejected: {dict(sorted(rejected.items()))}; failures: {len(failures)}")
    print(f"Removed {len(removed)} byte-identical copies across the merged dataset")
    if failures:
        raise RuntimeError(f"{len(failures)} Caltech IR downloads failed; rerun to resume")


if __name__ == "__main__":
    main()
