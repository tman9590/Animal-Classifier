#!/usr/bin/env python3
"""Prepare the official Lindenthal test split as an isolated IR holdout set."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image
from remotezip import RemoteZip

from catalog import read_catalog
from import_lindenthal_ir import ARCHIVE_ROOT, ARCHIVE_URL, LICENSE, SOURCE_PAGE, is_night
from label_images import prune_exact_duplicates, write_dataset_yaml


ROOT = Path(__file__).parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-url", default=ARCHIVE_URL)
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "evaluation" / "lindenthal-ir-holdout")
    args = parser.parse_args()

    _, species = read_catalog(ROOT / "species" / "north-carolina.json")
    labels = [item.label for item in species] + ["unknown"]
    unknown_index = len(species)
    image_dir = args.output / "images" / "val"
    label_dir = args.output / "labels" / "val"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with RemoteZip(args.archive_url) as archive:
        payload = json.loads(archive.read(ARCHIVE_ROOT + "test.json"))
        categories = {item["id"]: item["name"] for item in payload["categories"]}
        annotations_by_image: dict[str, list[dict]] = defaultdict(list)
        for annotation in payload["annotations"]:
            annotations_by_image[str(annotation["image_id"])].append(annotation)

        selected = [
            image for image in payload["images"]
            if str(image["id"]) in annotations_by_image and is_night(image["datetime"])
        ]
        for number, image_record in enumerate(selected, 1):
            source_name = ARCHIVE_ROOT + "images/" + image_record["file_name"].replace("\\", "/")
            data = archive.read(source_name)
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            width, height = int(image_record["width"]), int(image_record["height"])
            lines = []
            details = []
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
                details.append({
                    "source_label": categories[annotation["category_id"]],
                    "label": "unknown",
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                })
            if not lines:
                continue
            stem = "lindenthal-ir-test-" + str(image_record["id"]).replace("-", "_")
            (image_dir / f"{stem}.jpg").write_bytes(data)
            (label_dir / f"{stem}.txt").write_text("".join(lines))
            records.append({
                "source": source_name,
                "source_url": args.archive_url,
                "source_page": SOURCE_PAGE,
                "license": LICENSE,
                "official_split": "test",
                "capture_mode": "infrared intensity",
                "datetime": image_record["datetime"],
                "sequence_id": image_record["seq_id"],
                "sha256": hashlib.sha256(data).hexdigest(),
                "boxes": details,
            })
            if number % 20 == 0 or number == len(selected):
                print(f"Prepared {number}/{len(selected)} Lindenthal IR holdout images")

    write_dataset_yaml(args.output, labels)
    # The helper requires a train entry syntactically; alias it to the holdout,
    # while all evaluations explicitly request the validation split.
    dataset_yaml = args.output / "dataset.yaml"
    dataset_yaml.write_text(dataset_yaml.read_text().replace("train: images/train", "train: images/val"))
    (args.output / "holdout-audit.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    removed = prune_exact_duplicates(args.output)
    print(f"Prepared {len(records)} isolated IR holdout images with {sum(len(r['boxes']) for r in records)} boxes")
    print(f"Removed {len(removed)} byte-identical holdout copies")


if __name__ == "__main__":
    main()
