#!/usr/bin/env python3
"""Import a balanced, training-only subset of Caltech Camera Traps boxes."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from catalog import read_catalog
from label_images import prune_exact_duplicates, write_dataset_yaml


ROOT = Path(__file__).parents[1]
LICENSE = "Community Data License Agreement - Permissive 2.0"
SOURCE_PAGE = "https://lila.science/datasets/caltech-camera-traps"
DEFAULT_BASE_URL = "https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/"

# Coarse or geographically ambiguous source categories (deer, rabbit, bird,
# squirrel, skunk, fox, rodent, lizard, and bat) are deliberately excluded.
CATEGORY_MAP = {
    "opossum": "Virginia Opossum",
    "raccoon": "Common Raccoon",
    "coyote": "Coyote",
    "bobcat": "Bobcat",
    "cat": "Domestic Cat",
    "dog": "Domestic Dog",
}


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
    parser.add_argument("--max-boxes-per-class", type=int, default=250)
    parser.add_argument("--workers", type=int, default=24)
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
    missing = sorted(set(CATEGORY_MAP.values()) - set(label_indexes))
    if missing:
        raise ValueError(f"Caltech mappings are absent from the model catalog: {missing}")

    candidates = []
    for image_id, image_annotations in annotations_by_image.items():
        source_names = {categories[item["category_id"]] for item in image_annotations}
        if not source_names or not source_names.issubset(CATEGORY_MAP):
            continue
        image = images[image_id]
        candidates.append((hashlib.sha256(image_id.encode()).hexdigest(), image_id, image_annotations, image.get("seq_id", image_id)))

    selected = []
    selected_counts = Counter()
    used_sequences = set()
    for _, image_id, image_annotations, sequence_id in sorted(candidates):
        if sequence_id in used_sequences:
            continue
        target_names = [CATEGORY_MAP[categories[item["category_id"]]] for item in image_annotations]
        if not any(selected_counts[name] < args.max_boxes_per_class for name in target_names):
            continue
        selected.append((image_id, image_annotations))
        selected_counts.update(target_names)
        used_sequences.add(sequence_id)

    def fetch(item: tuple[str, list[dict]]) -> tuple[str, list[dict], Path, str]:
        image_id, image_annotations = item
        image = images[image_id]
        relative = image["file_name"].replace("\\", "/")
        destination = args.cache / relative
        url = urllib.parse.urljoin(args.base_url, urllib.parse.quote(relative))
        if not destination.exists():
            download(url, destination)
        return image_id, image_annotations, destination, url

    fetched = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch, item): item[0] for item in selected}
        for number, future in enumerate(as_completed(futures), 1):
            try:
                fetched.append(future.result())
            except Exception as error:
                failures.append({"image_id": futures[future], "error": str(error)})
            if number % 100 == 0 or number == len(futures):
                print(f"Downloaded/verified {number}/{len(futures)} Caltech images; {len(failures)} failures")

    output_images = args.output / "images" / "train"
    output_labels = args.output / "labels" / "train"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    audit_records = []
    imported_counts = Counter()
    for image_id, image_annotations, source, url in sorted(fetched):
        metadata = images[image_id]
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
            normalized = ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)
            lines.append(f"{label_indexes[target_name]} " + " ".join(f"{value:.8f}" for value in normalized) + "\n")
            details.append({"source_label": source_name, "label": target_name, "bbox": [x1, y1, x2 - x1, y2 - y1]})
            imported_counts[target_name] += 1
        if not lines:
            continue
        stem = f"caltech-{image_id}"
        shutil.copy2(source, output_images / f"{stem}.jpg")
        (output_labels / f"{stem}.txt").write_text("".join(lines))
        audit_records.append({
            "source": metadata["file_name"],
            "source_url": url,
            "source_page": SOURCE_PAGE,
            "rights_holder": metadata.get("rights_holder"),
            "license": LICENSE,
            "sequence_id": metadata.get("seq_id"),
            "location": metadata.get("location"),
            "split": "train",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "boxes": details,
        })

    (args.output / "caltech-audit.jsonl").write_text("".join(json.dumps(record) + "\n" for record in audit_records))
    (args.output / "caltech-failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    write_dataset_yaml(args.output, labels)
    removed = prune_exact_duplicates(args.output)
    print(f"Imported {len(audit_records)} Caltech images with {sum(imported_counts.values())} boxes")
    print(f"Class counts: {dict(sorted(imported_counts.items()))}")
    print(f"Removed {len(removed)} byte-identical copies across the merged dataset")
    if failures:
        raise RuntimeError(f"{len(failures)} Caltech downloads failed; rerun to resume")


if __name__ == "__main__":
    main()
