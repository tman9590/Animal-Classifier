#!/usr/bin/env python3
"""Stage a bounded, taxonomy-matched subset from a LILA COCO dataset.

This script never modifies the active training dataset. Exact species mappings
come from LILA's iNaturalist taxonomy table. Source bounding boxes are retained
when present; otherwise the manifest marks images for later localization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from catalog import USER_AGENT, read_catalog


ROOT = Path(__file__).parents[1]


def stable_split(location: str) -> str:
    digest = hashlib.sha256(f"lila-location:{location}".encode()).hexdigest()
    return "val" if int(digest, 16) % 5 == 0 else "train"


def normalize_split(image: dict) -> str | None:
    source_split = str(image.get("split") or "").lower()
    if source_split in {"train", "training"}:
        return "train"
    if source_split in {"val", "validation"}:
        return "val"
    if source_split in {"test", "testing"}:
        return None
    return stable_split(str(image.get("location") or image.get("seq_id") or image["id"]))


def download_jpeg(url: str, destination: Path) -> tuple[int, int]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    with Image.open(io.BytesIO(data)) as image:
        rgb = image.convert("RGB")
        destination.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(destination, format="JPEG", quality=95)
        return rgb.size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True, help="Exact dataset_name in LILA taxonomy mapping")
    parser.add_argument("--slug", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image-base-url", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--citation", required=True)
    parser.add_argument("--target", choices=("low", "all"), default="low")
    parser.add_argument("--catalog", type=Path, default=ROOT / "species" / "north-carolina.json")
    parser.add_argument(
        "--coverage-report",
        type=Path,
        default=ROOT / "work" / "dataset-production" / "dataset-audit-rare-final.json",
    )
    parser.add_argument(
        "--taxonomy-mapping",
        type=Path,
        default=ROOT / "work" / "external" / "lila-meta" / "lila-taxonomy-mapping.csv",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "external" / "lila-staged")
    parser.add_argument("--max-train", type=int, default=120)
    parser.add_argument("--max-val", type=int, default=30)
    parser.add_argument("--download-workers", type=int, default=8)
    args = parser.parse_args()

    _, species = read_catalog(args.catalog)
    by_scientific = {item.scientific_name.lower(): (index, item.label) for index, item in enumerate(species)}
    report = json.loads(args.coverage_report.read_text())
    low_labels = set(report["classes_under_5_boxes"]) | set(report["classes_missing_from_validation"])

    category_to_scientific = {}
    with args.taxonomy_mapping.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["dataset_name"] == args.dataset_name and row.get("species"):
                category_to_scientific[row["query"]] = row["species"].lower()

    data = json.loads(args.metadata.read_text())
    images = {image["id"]: image for image in data["images"]}
    categories = {category["id"]: category["name"] for category in data["categories"]}
    class_by_category = {}
    for category_id, category_name in categories.items():
        mapped = by_scientific.get(category_to_scientific.get(category_name, ""))
        if mapped is not None:
            class_by_category[category_id] = mapped

    annotations_by_image = defaultdict(list)
    for annotation in data["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    candidates = []
    for image_id, annotations in annotations_by_image.items():
        if not annotations or any(annotation["category_id"] not in class_by_category for annotation in annotations):
            continue
        mapped = [class_by_category[annotation["category_id"]] for annotation in annotations]
        if args.target == "low" and not any(label in low_labels for _, label in mapped):
            continue
        split = normalize_split(images[image_id])
        if split is None:
            continue
        candidates.append((images[image_id], annotations, mapped, split))

    # Reproducible ordering prevents API/cloud listing order from changing the subset.
    candidates.sort(key=lambda row: hashlib.sha256(str(row[0]["id"]).encode()).hexdigest())
    quotas = {"train": args.max_train, "val": args.max_val}
    selected_counts = Counter()
    selected = []
    for image, annotations, mapped, split in candidates:
        target_labels = {label for _, label in mapped if args.target == "all" or label in low_labels}
        if not target_labels or all(selected_counts[(split, label)] >= quotas[split] for label in target_labels):
            continue
        selected.append((image, annotations, mapped, split))
        for label in target_labels:
            selected_counts[(split, label)] += 1

    destination_root = args.output / args.slug
    existing_manifest = destination_root / "manifest.jsonl"
    completed = {}
    if existing_manifest.exists():
        for line in existing_manifest.read_text().splitlines():
            record = json.loads(line)
            completed[record["source_image_id"]] = record

    def fetch(row):
        image, annotations, mapped, split = row
        source_name = image["file_name"].replace("\\", "/")
        url = args.image_base_url.rstrip("/") + "/" + urllib.parse.quote(source_name, safe="/")
        digest = hashlib.sha256(str(image["id"]).encode()).hexdigest()[:20]
        stem = f"{args.slug}-{digest}"
        destination = destination_root / "images" / split / f"{stem}.jpg"
        if destination.exists():
            with Image.open(destination) as existing:
                width, height = existing.size
        else:
            width, height = download_jpeg(url, destination)
        boxes = []
        for annotation, (class_index, label) in zip(annotations, mapped):
            bbox = annotation.get("bbox")
            boxes.append({"class_index": class_index, "label": label, "bbox_xywh": bbox})
        if all(box["bbox_xywh"] is not None for box in boxes):
            label_path = destination_root / "labels" / split / f"{stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for box in boxes:
                x, y, w, h = box["bbox_xywh"]
                # A small number of source annotations extend beyond the JPEG
                # dimensions. Retain the visible portion rather than emitting
                # invalid YOLO coordinates.
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(float(width), x + w), min(float(height), y + h)
                if x2 <= x1 or y2 <= y1:
                    continue
                normalized = (
                    (x1 + x2) / 2 / width,
                    (y1 + y2) / 2 / height,
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                )
                lines.append(f"{box['class_index']} " + " ".join(f"{value:.8f}" for value in normalized))
            label_path.write_text("\n".join(lines) + "\n")
        return {
            "dataset": args.dataset_name,
            "source_image_id": image["id"],
            "source_file": source_name,
            "source_url": url,
            "location": image.get("location"),
            "source_split": image.get("split"),
            "split": split,
            "output_image": destination.relative_to(destination_root).as_posix(),
            "width": width,
            "height": height,
            "boxes": boxes,
            "requires_localization": any(box["bbox_xywh"] is None for box in boxes),
            "license": args.license,
            "citation": args.citation,
        }

    pending = [row for row in selected if row[0]["id"] not in completed]
    failures = []
    missing_source_images = []
    with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
        futures = {executor.submit(fetch, row): row for row in pending}
        for number, future in enumerate(as_completed(futures), 1):
            try:
                record = future.result()
                completed[record["source_image_id"]] = record
            except Exception as error:
                image = futures[future][0]
                record = {"source_image_id": image["id"], "error": str(error)}
                if isinstance(error, HTTPError) and error.code == 404:
                    missing_source_images.append(record)
                else:
                    failures.append(record)
            if number % 200 == 0 or number == len(futures):
                print(f"Staged {number}/{len(futures)}; failures={len(failures)}")

    destination_root.mkdir(parents=True, exist_ok=True)
    existing_manifest.write_text(
        "".join(json.dumps(completed[key]) + "\n" for key in sorted(completed, key=str))
    )
    summary_counts = Counter()
    localization_count = 0
    for record in completed.values():
        localization_count += int(record["requires_localization"])
        for box in record["boxes"]:
            summary_counts[(record["split"], box["label"])] += 1
    summary = {
        "dataset": args.dataset_name,
        "license": args.license,
        "citation": args.citation,
        "images_staged": len(completed),
        "images_requiring_localization": localization_count,
        "annotations_by_split_and_class": {
            f"{split}:{label}": count for (split, label), count in sorted(summary_counts.items())
        },
        "download_failures": failures,
        "missing_source_images": missing_source_images,
    }
    (destination_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} downloads failed; rerun to resume")


if __name__ == "__main__":
    main()
