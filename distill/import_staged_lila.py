#!/usr/bin/env python3
"""Merge staged LILA subsets into the production YOLO dataset.

Manual source boxes are copied directly. Images without boxes are localized
with MegaDetector while retaining their exact source species label. The script
is idempotent and is intended to run only after active training has stopped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from catalog import read_catalog
from label_images import prune_exact_duplicates, write_dataset_yaml
from teacher import MegaDetector


ROOT = Path(__file__).parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "staged",
        nargs="+",
        type=Path,
        help="Staged dataset directories containing manifest.jsonl",
    )
    parser.add_argument("--catalog", type=Path, default=ROOT / "species" / "north-carolina.json")
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "dataset-production")
    parser.add_argument("--megadetector", type=Path, default=ROOT / "work" / "models" / "MDV6-yolov9-c.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    _, species = read_catalog(args.catalog)
    labels = [item.label for item in species] + ["unknown"]
    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    records = []
    for staged_root in args.staged:
        manifest = staged_root / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        for line in manifest.read_text().splitlines():
            record = json.loads(line)
            record["_staged_root"] = str(staged_root)
            records.append(record)

    audit_path = args.output / "lila-staged-import-audit.jsonl"
    audit_by_key = {}
    if audit_path.exists():
        for line in audit_path.read_text().splitlines():
            record = json.loads(line)
            audit_by_key[(record["dataset"], record["source_image_id"])] = record

    counts = Counter()
    failures = []

    # First copy datasets that already provide manual boxes.
    for record in records:
        if record["requires_localization"]:
            continue
        staged_root = Path(record["_staged_root"])
        source_image = staged_root / record["output_image"]
        split = record["split"]
        destination_image = args.output / "images" / split / source_image.name
        source_label = staged_root / "labels" / split / f"{source_image.stem}.txt"
        destination_label = args.output / "labels" / split / source_label.name
        if not source_image.exists() or not source_label.exists():
            failures.append({"dataset": record["dataset"], "source_image_id": record["source_image_id"], "error": "staged image or label missing"})
            continue
        if not destination_image.exists():
            shutil.copy2(source_image, destination_image)
            shutil.copy2(source_label, destination_label)
        key = (record["dataset"], record["source_image_id"])
        audit_by_key[key] = {
            **{k: v for k, v in record.items() if not k.startswith("_")},
            "status": "ok",
            "output_image": destination_image.relative_to(args.output).as_posix(),
            "output_label": destination_label.relative_to(args.output).as_posix(),
            "localization": "manual source bounding boxes",
            "sha256": hashlib.sha256(source_image.read_bytes()).hexdigest(),
        }
        for box in record["boxes"]:
            counts[(record["dataset"], split, box["label"])] += 1

    localization_records = [record for record in records if record["requires_localization"]]
    detector = MegaDetector(args.megadetector, args.device) if localization_records else None
    for offset in range(0, len(localization_records), args.batch_size):
        batch_records = localization_records[offset : offset + args.batch_size]
        images = []
        valid = []
        for record in batch_records:
            source_image = Path(record["_staged_root"]) / record["output_image"]
            try:
                images.append(Image.open(source_image).convert("RGB"))
                valid.append((record, source_image))
            except Exception as error:
                failures.append({"dataset": record["dataset"], "source_image_id": record["source_image_id"], "error": str(error)})
        detections = detector.detect_many(images, args.threshold) if detector else []
        for (record, source_image), image, boxes in zip(valid, images, detections):
            source_classes = {(box["class_index"], box["label"]) for box in record["boxes"]}
            if len(source_classes) != 1:
                failures.append({"dataset": record["dataset"], "source_image_id": record["source_image_id"], "error": "localization requires exactly one source species"})
                continue
            class_index, label = next(iter(source_classes))
            split = record["split"]
            destination_image = args.output / "images" / split / source_image.name
            destination_label = args.output / "labels" / split / f"{source_image.stem}.txt"
            lines = []
            details = []
            for box, confidence in boxes:
                x1, y1, x2, y2 = box
                normalized = (
                    (x1 + x2) / 2 / image.width,
                    (y1 + y2) / 2 / image.height,
                    (x2 - x1) / image.width,
                    (y2 - y1) / image.height,
                )
                lines.append(f"{class_index} " + " ".join(f"{value:.8f}" for value in normalized))
                details.append({"bbox_xyxy": box, "confidence": confidence, "class_index": class_index, "label": label})
            status = "ok" if lines else "no_detection"
            if lines:
                shutil.copy2(source_image, destination_image)
                destination_label.write_text("\n".join(lines) + "\n")
                counts[(record["dataset"], split, label)] += len(lines)
            key = (record["dataset"], record["source_image_id"])
            audit_by_key[key] = {
                **{k: v for k, v in record.items() if not k.startswith("_")},
                "status": status,
                "output_image": destination_image.relative_to(args.output).as_posix() if lines else None,
                "output_label": destination_label.relative_to(args.output).as_posix() if lines else None,
                "localization": "MegaDetector animal box; species from LILA source annotation",
                "localized_boxes": details,
                "sha256": hashlib.sha256(source_image.read_bytes()).hexdigest(),
            }
        completed = min(offset + args.batch_size, len(localization_records))
        if completed % 160 == 0 or completed == len(localization_records):
            print(f"Localized {completed}/{len(localization_records)} staged images")

    audit_path.write_text(
        "".join(json.dumps(audit_by_key[key]) + "\n" for key in sorted(audit_by_key))
    )
    write_dataset_yaml(args.output, labels)
    removed = prune_exact_duplicates(args.output)
    summary = {
        "source_images": len(records),
        "images_imported": sum(1 for record in audit_by_key.values() if record["status"] == "ok"),
        "images_without_detection": sum(1 for record in audit_by_key.values() if record["status"] == "no_detection"),
        "boxes_by_dataset_split_and_class": {
            f"{dataset}:{split}:{label}": count
            for (dataset, split, label), count in sorted(counts.items())
        },
        "duplicates_removed": removed,
        "failures": failures,
    }
    (args.output / "lila-staged-import-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} staged records failed")


if __name__ == "__main__":
    main()
