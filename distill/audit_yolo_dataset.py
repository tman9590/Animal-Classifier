#!/usr/bin/env python3
"""Validate YOLO dataset structure, coordinates, class IDs, and exact leakage."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def read_labels(dataset_yaml: Path) -> list[str]:
    labels = []
    for line in dataset_yaml.read_text().splitlines():
        stripped = line.strip()
        if stripped and stripped[0].isdigit() and ":" in stripped:
            labels.append(json.loads(stripped.split(":", 1)[1].strip()))
    return labels


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    labels = read_labels(args.dataset / "dataset.yaml")
    errors = []
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    class_counts = Counter()
    source_counts = Counter()
    image_counts = Counter()
    box_counts = Counter()

    for split in ("train", "val"):
        image_dir = args.dataset / "images" / split
        label_dir = args.dataset / "labels" / split
        images = {path.stem: path for path in image_dir.glob("*.jpg")}
        annotations = {path.stem: path for path in label_dir.glob("*.txt")}
        for stem in sorted(images.keys() - annotations.keys()):
            errors.append(f"{split}: image without label: {stem}")
        for stem in sorted(annotations.keys() - images.keys()):
            errors.append(f"{split}: label without image: {stem}")
        for stem in sorted(images.keys() & annotations.keys()):
            image_path = images[stem]
            try:
                with Image.open(image_path) as image:
                    image_format = image.format
                    image.load()
            except Exception as error:
                errors.append(f"{split}: unreadable image {image_path.name}: {error}")
                continue
            if image_format not in {"JPEG", "MPO"}:
                errors.append(f"{split}: {image_path.name}: extension is .jpg but content is {image_format}")
                continue
            # Pillow can decode some truncated JPEGs that Ultralytics will later
            # rewrite in-place.  Catch those before training mutates an audited
            # dataset and invalidates provenance hashes.
            with image_path.open("rb") as stream:
                stream.seek(-2, 2)
                if stream.read() != b"\xff\xd9":
                    errors.append(f"{split}: {image_path.name}: JPEG is missing its end marker")
                    continue
            hashes[file_hash(image_path)].append((split, image_path.name))
            prefix = stem.split("-", 1)[0]
            source_counts[(split, prefix)] += 1
            image_counts[split] += 1
            for line_number, line in enumerate(annotations[stem].read_text().splitlines(), 1):
                fields = line.split()
                if len(fields) != 5:
                    errors.append(f"{split}: {annotations[stem].name}:{line_number}: expected 5 fields")
                    continue
                try:
                    class_id = int(fields[0])
                    x, y, width, height = map(float, fields[1:])
                except ValueError as error:
                    errors.append(f"{split}: {annotations[stem].name}:{line_number}: {error}")
                    continue
                if not 0 <= class_id < len(labels):
                    errors.append(f"{split}: {annotations[stem].name}:{line_number}: class {class_id} out of range")
                    continue
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    errors.append(f"{split}: {annotations[stem].name}:{line_number}: invalid normalized box")
                    continue
                if x - width / 2 < -1e-6 or x + width / 2 > 1 + 1e-6 or y - height / 2 < -1e-6 or y + height / 2 > 1 + 1e-6:
                    errors.append(f"{split}: {annotations[stem].name}:{line_number}: box exceeds image")
                    continue
                class_counts[(split, labels[class_id])] += 1
                box_counts[split] += 1

    duplicate_groups = [items for items in hashes.values() if len(items) > 1]
    cross_split_duplicates = [items for items in duplicate_groups if len({item[0] for item in items}) > 1]
    if cross_split_duplicates:
        errors.append(f"{len(cross_split_duplicates)} exact duplicate groups cross train/validation")

    known_labels = [label for label in labels if label != "unknown"]
    total_class_counts = Counter()
    for (_, label), count in class_counts.items():
        total_class_counts[label] += count
    report = {
        "dataset": str(args.dataset.resolve()),
        "labels_including_unknown": len(labels),
        "images": dict(image_counts),
        "boxes": dict(box_counts),
        "images_by_source": {f"{split}:{prefix}": count for (split, prefix), count in sorted(source_counts.items())},
        "unknown_boxes": total_class_counts["unknown"],
        "unknown_box_rate": total_class_counts["unknown"] / sum(total_class_counts.values()),
        "known_classes_with_boxes": sum(total_class_counts[label] > 0 for label in known_labels),
        "zero_support_classes": [label for label in known_labels if total_class_counts[label] == 0],
        "classes_under_5_boxes": [label for label in known_labels if total_class_counts[label] < 5],
        "classes_missing_from_validation": [
            label for label in known_labels
            if total_class_counts[label] and class_counts[("val", label)] == 0
        ],
        "top_30_classes": total_class_counts.most_common(30),
        "exact_duplicate_groups": len(duplicate_groups),
        "exact_cross_split_duplicate_groups": len(cross_split_duplicates),
        "errors": errors,
    }
    report_path = args.report or args.dataset / "dataset-audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
