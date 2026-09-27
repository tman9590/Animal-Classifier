#!/usr/bin/env python3
"""Import a training-only subset of annotated Lindenthal infrared frames.

The source taxonomy is intentionally not projected onto the North Carolina
catalog: deer, goat, donkey, and goose are too coarse or geographically
incompatible.  Their boxes still provide useful IR animal-localization examples
and are therefore assigned to the model's explicit ``unknown`` animal class.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from remotezip import RemoteZip

from catalog import read_catalog
from label_images import prune_exact_duplicates, write_dataset_yaml


ROOT = Path(__file__).parents[1]
ARCHIVE_URL = "https://storage.googleapis.com/public-datasets-lila/lindenthal-camera-traps/lindenthal-camera-traps.zip"
ARCHIVE_ROOT = "lindenthal-camera-traps/lindenthal_coco/"
SOURCE_PAGE = "https://lila.science/datasets/lindenthal-camera-traps/"
LICENSE = "Community Data License Agreement - Permissive 2.0"


def is_night(datetime_text: str) -> bool:
    hour = int(datetime_text[11:13])
    return hour >= 20 or hour < 6


def selected_images(payload: dict, frame_stride: int, max_images: int) -> list[dict]:
    """Select temporally separated annotated images from nighttime sequences."""
    annotations_by_image: dict[str, list[dict]] = defaultdict(list)
    for annotation in payload["annotations"]:
        annotations_by_image[str(annotation["image_id"])].append(annotation)

    by_sequence: dict[str, list[dict]] = defaultdict(list)
    for image in payload["images"]:
        image_id = str(image["id"])
        if image_id in annotations_by_image and is_night(image["datetime"]):
            by_sequence[str(image["seq_id"])].append(image)

    selected = []
    for sequence_id in sorted(by_sequence):
        previous_frame = -frame_stride
        for image in sorted(by_sequence[sequence_id], key=lambda item: int(item["frame_num"])):
            frame = int(image["frame_num"])
            if frame - previous_frame < frame_stride:
                continue
            selected.append(image)
            previous_frame = frame
    selected.sort(key=lambda item: hashlib.sha256(str(item["id"]).encode()).hexdigest())
    return selected[:max_images]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-url", default=ARCHIVE_URL)
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "dataset-production")
    parser.add_argument("--cache", type=Path, default=ROOT / "work" / "external" / "lindenthal-ir")
    parser.add_argument("--frame-stride", type=int, default=40)
    parser.add_argument("--max-images", type=int, default=180)
    args = parser.parse_args()

    _, species = read_catalog(ROOT / "species" / "north-carolina.json")
    labels = [item.label for item in species] + ["unknown"]
    unknown_index = len(species)
    args.cache.mkdir(parents=True, exist_ok=True)

    audit_path = args.output / "lindenthal-ir-audit.jsonl"
    existing_records = [json.loads(line) for line in audit_path.read_text().splitlines()] if audit_path.exists() else []
    completed = {record["source"] for record in existing_records}

    with RemoteZip(args.archive_url) as archive:
        metadata_name = ARCHIVE_ROOT + "train.json"
        payload = json.loads(archive.read(metadata_name))
        categories = {item["id"]: item["name"] for item in payload["categories"]}
        annotations_by_image: dict[str, list[dict]] = defaultdict(list)
        for annotation in payload["annotations"]:
            annotations_by_image[str(annotation["image_id"])].append(annotation)

        candidates = selected_images(payload, args.frame_stride, args.max_images)
        output_images = args.output / "images" / "train"
        output_labels = args.output / "labels" / "train"
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        records = list(existing_records)
        counts = Counter(record["source_label"] for record in records for _ in record["boxes"])

        for number, image_record in enumerate(candidates, 1):
            relative = image_record["file_name"].replace("\\", "/")
            source_name = ARCHIVE_ROOT + "images/" + relative
            if source_name in completed:
                continue
            data = archive.read(source_name)
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            cache_path = args.cache / f"{image_record['id']}.jpg"
            cache_path.write_bytes(data)

            width, height = int(image_record["width"]), int(image_record["height"])
            lines = []
            details = []
            source_labels = []
            for annotation in annotations_by_image[str(image_record["id"])]:
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
                lines.append(f"{unknown_index} " + " ".join(f"{value:.8f}" for value in normalized) + "\n")
                source_label = categories[annotation["category_id"]]
                source_labels.append(source_label)
                details.append({
                    "source_label": source_label,
                    "label": "unknown",
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "track_id": annotation.get("attributes", {}).get("track_id"),
                })
            if not lines:
                continue

            stem = "lindenthal-ir-" + str(image_record["id"]).replace("-", "_")
            (output_images / f"{stem}.jpg").write_bytes(data)
            (output_labels / f"{stem}.txt").write_text("".join(lines))
            counts.update(source_labels)
            records.append({
                "source": source_name,
                "source_url": args.archive_url,
                "source_page": SOURCE_PAGE,
                "license": LICENSE,
                "split": "train",
                "capture_mode": "infrared intensity",
                "datetime": image_record["datetime"],
                "sequence_id": image_record["seq_id"],
                "frame_number": image_record["frame_num"],
                "source_label": source_labels[0] if len(set(source_labels)) == 1 else "mixed",
                "sha256": hashlib.sha256(data).hexdigest(),
                "boxes": details,
            })
            if number % 25 == 0 or number == len(candidates):
                print(f"Imported/verified {number}/{len(candidates)} selected Lindenthal IR images")

    audit_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    write_dataset_yaml(args.output, labels)
    removed = prune_exact_duplicates(args.output)
    print(f"Lindenthal IR dataset now contributes {len(records)} images and {sum(len(r['boxes']) for r in records)} boxes")
    print(f"Source-label counts: {dict(sorted(counts.items()))}")
    print(f"Removed {len(removed)} byte-identical copies across the merged dataset")


if __name__ == "__main__":
    main()
