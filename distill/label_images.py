#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import hashlib
from collections import defaultdict
import shutil
from pathlib import Path

from PIL import Image

from catalog import read_catalog
from teacher import BioClipClassifier, MegaDetector, crop_box


ROOT = Path(__file__).parents[1]


def yolo_box(box, width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)


def write_dataset_yaml(output: Path, labels: list[str]) -> None:
    yaml = [f"path: {output.resolve()}", "train: images/train", "val: images/val", "names:"]
    yaml.extend(f"  {index}: {json.dumps(label)}" for index, label in enumerate(labels))
    (output / "dataset.yaml").write_text("\n".join(yaml) + "\n")


def prune_exact_duplicates(output: Path, pattern: str = "*.jpg") -> list[dict[str, str]]:
    """Keep one copy of byte-identical images, preferring validation copies.

    Validation wins when a duplicate crosses partitions so no validation image is
    also visible during training.  A machine-readable filter log keeps the
    removal auditable and makes repeated runs deterministic.
    """
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for split in ("train", "val"):
        for path in sorted((output / "images" / split).glob(pattern)):
            by_hash[hashlib.sha256(path.read_bytes()).hexdigest()].append(path)
    removed = []
    for digest, paths in sorted(by_hash.items()):
        if len(paths) < 2:
            continue
        keep = min(paths, key=lambda path: (path.parent.name != "val", path.name))
        for path in paths:
            if path == keep:
                continue
            label = output / "labels" / path.parent.name / f"{path.stem}.txt"
            path.unlink(missing_ok=True)
            label.unlink(missing_ok=True)
            removed.append({
                "sha256": digest,
                "kept": keep.relative_to(output).as_posix(),
                "removed": path.relative_to(output).as_posix(),
            })
    log_path = output / "exact-duplicate-filter.jsonl"
    history = []
    if log_path.exists():
        history = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    seen = {(record["sha256"], record["kept"], record["removed"]) for record in history}
    history.extend(
        record for record in removed
        if (record["sha256"], record["kept"], record["removed"]) not in seen
    )
    log_path.write_text("".join(json.dumps(record) + "\n" for record in history))
    return removed


def observation_split(stem: str) -> str:
    observation = stem.split("-")[0]
    return "val" if int(hashlib.sha256(observation.encode()).hexdigest(), 16) % 5 == 0 else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo-label licensed iNaturalist images with MDV6 and BioCLIP 2")
    parser.add_argument("--catalog", type=Path, default=ROOT / "species" / "north-carolina.json")
    parser.add_argument("--source", type=Path, default=ROOT / "work" / "source-images")
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "yolo-dataset")
    parser.add_argument("--megadetector", type=Path, default=ROOT / "work" / "models" / "MDV6-yolov9-c.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--detection-threshold", type=float, default=0.25)
    parser.add_argument("--unknown-threshold", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Limit this run to a reproducible smoke-test subset")
    args = parser.parse_args()

    metadata, species = read_catalog(args.catalog)
    labels = [item.label for item in species] + ["unknown"]
    unknown_index = len(species)
    rows = list(csv.DictReader((args.source / "attribution.csv").open()))
    by_photo = defaultdict(list)
    for row in rows:
        by_photo[Path(row["file"]).stem.split("-")[-1]].append(row)
    conflicting_groups = [group for group in by_photo.values() if len({row["class_index"] for row in group}) > 1]
    conflicts = len(conflicting_groups)
    # Newly recovered metadata can expose conflicts in an earlier partial pass.
    # Remove only generated derivatives; the licensed originals remain untouched.
    for group in conflicting_groups:
        for row in group:
            stem = "public-" + Path(row["file"]).stem
            for split in ("train", "val"):
                for folder, suffix in (("images", ".jpg"), ("labels", ".txt")):
                    (args.output / folder / split / (stem + suffix)).unlink(missing_ok=True)
    rows = [group[0] for group in by_photo.values() if len({row["class_index"] for row in group}) == 1]
    eligible_sources = {row["file"] for row in rows}
    print(f"Excluded {conflicts} photos with conflicting source taxon assignments")
    random.Random(360).shuffle(rows)
    if args.limit is not None:
        rows = rows[:args.limit]

    args.output.mkdir(parents=True, exist_ok=True)
    signature = {
        "catalog_sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        "megadetector_sha256": hashlib.sha256(args.megadetector.read_bytes()).hexdigest(),
        "detection_threshold": args.detection_threshold,
        "unknown_threshold": args.unknown_threshold,
        "pipeline_version": 3,
    }
    run_path = args.output / "public-run.json"
    if run_path.exists() and json.loads(run_path.read_text()) != signature:
        raise ValueError("Teacher settings changed; choose a fresh output directory")
    run_path.write_text(json.dumps(signature, indent=2) + "\n")
    audit_path = args.output / "public-audit.jsonl"
    completed = set()
    if audit_path.exists():
        records_by_source = {}
        for line in audit_path.read_text().splitlines():
            record = json.loads(line)
            if record.get("status") == "ok" and record.get("source") in eligible_sources:
                records_by_source[record["source"]] = record
        audit_path.write_text("".join(json.dumps(record) + "\n" for record in records_by_source.values()))
        completed.update(records_by_source)
    rows = [row for row in rows if row["file"] not in completed]
    print(f"Resuming with {len(rows)} public images; {len(completed)} already processed")
    write_dataset_yaml(args.output, labels)
    if not rows:
        removed = prune_exact_duplicates(args.output, "public-*.jpg")
        print(f"Removed {len(removed)} byte-identical public image copies")
        return
    detector = MegaDetector(args.megadetector, args.device)
    classifier = BioClipClassifier(species, args.device, ROOT / "work" / "models")
    audit = audit_path.open("a")
    for batch_start in range(0, len(rows), args.batch_size):
        batch = []
        for row_number, row in enumerate(rows[batch_start : batch_start + args.batch_size], batch_start):
            source = args.source / row["file"]
            try:
                batch.append((row_number, row, source, Image.open(source).convert("RGB")))
            except Exception as error:
                print(f"skip {source}: {error}")
        detections = detector.detect_many([item[3] for item in batch], args.detection_threshold)
        crops = [crop_box(image, box) for (_, _, _, image), boxes in zip(batch, detections) for box, _ in boxes]
        predictions = iter(classifier.classify(crops))
        for (row_number, row, source, image), boxes in zip(batch, detections):
            # Keep every photo from an observation in the same stable partition.
            split = observation_split(source.stem)
            image_dir = args.output / "images" / split
            label_dir = args.output / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            annotations = []
            details = []
            for box, _ in boxes:
                probabilities = next(predictions)
                predicted_label, confidence = max(probabilities.items(), key=lambda item: item[1])
                expected_index = int(row["class_index"])
                class_index = (
                    expected_index
                    if predicted_label == species[expected_index].label and confidence >= args.unknown_threshold
                    else unknown_index
                )
                annotations.append((class_index, *yolo_box(box, image.width, image.height)))
                details.append({"xyxy": box, "label": labels[class_index], "teacher_label": predicted_label, "confidence": confidence})
            stem = "public-" + source.stem
            if annotations:
                shutil.copy2(source, image_dir / f"{stem}.jpg")
                (label_dir / f"{stem}.txt").write_text(
                    "".join(f"{index} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n" for index, x, y, w, h in annotations)
                )
            audit.write(json.dumps({"source": row["file"], "status": "ok", "split": split, "boxes": details}) + "\n")
            audit.flush()
        print(f"Labeled {min(batch_start + args.batch_size, len(rows))}/{len(rows)} public images")

    audit.close()
    removed = prune_exact_duplicates(args.output, "public-*.jpg")
    print(f"Removed {len(removed)} byte-identical public image copies")
    write_dataset_yaml(args.output, labels)
    print(f"Wrote training dataset to {args.output}")


if __name__ == "__main__":
    main()
