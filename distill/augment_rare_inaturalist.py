#!/usr/bin/env python3
"""Add licensed, research-grade iNaturalist examples for unsupported classes.

The observation identification supplies the species label; MegaDetector is used
only to localize animals.  This avoids turning rare but correctly identified
species into ``unknown`` merely because a zero-shot classifier confuses two
visually similar taxa.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from catalog import USER_AGENT, api_get, read_catalog
from label_images import prune_exact_duplicates, write_dataset_yaml
from teacher import MegaDetector


ROOT = Path(__file__).parents[1]
SOURCE_PAGE = "https://www.inaturalist.org"
ALLOWED_LICENSES = {"cc0", "cc-by"}


def observation_split(observation_id: int) -> str:
    return "val" if int(hashlib.sha256(str(observation_id).encode()).hexdigest(), 16) % 5 == 0 else "train"


def image_url(photo: dict) -> str:
    return photo["url"].replace("square", "large")


def download_jpeg(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read()
    with Image.open(io.BytesIO(data)) as image:
        image.convert("RGB").save(destination, format="JPEG", quality=95)


def exact_or_descendant(observation: dict, species_taxon_id: int) -> bool:
    taxon = observation.get("taxon") or {}
    return int(taxon.get("id") or -1) == species_taxon_id or species_taxon_id in taxon.get("ancestor_ids", [])


def select_observations(
    item,
    existing_observation_ids: set[int],
    train_candidates: int,
    val_candidates: int,
) -> list[dict]:
    payload = api_get(
        "observations",
        {
            "taxon_id": item.taxon_id,
            "quality_grade": "research",
            "photos": "true",
            "photo_license": "cc0,cc-by",
            "per_page": 200,
            "order_by": "votes",
            "order": "desc",
            "locale": "en",
        },
    )
    quotas = {"train": train_candidates, "val": val_candidates}
    selected = []
    for observation in payload.get("results", []):
        observation_id = int(observation["id"])
        if observation_id in existing_observation_ids or not exact_or_descendant(observation, item.taxon_id):
            continue
        photos = observation.get("photos") or []
        if not photos:
            continue
        photo = photos[0]
        license_code = (photo.get("license_code") or "").lower()
        if license_code not in ALLOWED_LICENSES:
            continue
        split = observation_split(observation_id)
        if quotas[split] <= 0:
            continue
        selected.append({
            "observation_id": observation_id,
            "photo_id": int(photo["id"]),
            "photo_url": image_url(photo),
            "license": license_code,
            "attribution": photo.get("attribution") or "",
            "split": split,
        })
        quotas[split] -= 1
        if not any(quotas.values()):
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "species" / "north-carolina.json")
    parser.add_argument("--coverage-report", type=Path, default=ROOT / "work" / "dataset-production" / "dataset-audit-ir-final.json")
    parser.add_argument("--target", choices=("zero", "missing-val", "under-five"), default="zero")
    parser.add_argument("--cache", type=Path, default=ROOT / "work" / "external" / "inaturalist-rare")
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "dataset-production")
    parser.add_argument("--megadetector", type=Path, default=ROOT / "work" / "models" / "MDV6-yolov9-c.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-candidates", type=int, default=40)
    parser.add_argument("--val-candidates", type=int, default=10)
    parser.add_argument("--download-workers", type=int, default=8)
    args = parser.parse_args()

    _, species = read_catalog(args.catalog)
    labels = [item.label for item in species] + ["unknown"]
    label_indexes = {label: index for index, label in enumerate(labels)}
    report = json.loads(args.coverage_report.read_text())
    target_key = {
        "zero": "zero_support_classes",
        "missing-val": "classes_missing_from_validation",
        "under-five": "classes_under_5_boxes",
    }[args.target]
    target_labels = set(report[target_key])
    targets = [(index, item) for index, item in enumerate(species) if item.label in target_labels]

    existing_observation_ids = set()
    attribution_path = ROOT / "work" / "source-images" / "attribution.csv"
    if attribution_path.exists():
        for row in csv.DictReader(attribution_path.open()):
            try:
                existing_observation_ids.add(int(Path(row["file"]).stem.split("-")[0]))
            except (ValueError, IndexError):
                pass

    audit_path = args.output / "rare-inaturalist-audit.jsonl"
    records = [json.loads(line) for line in audit_path.read_text().splitlines() if line] if audit_path.exists() else []
    existing_observation_ids.update(int(record["observation_id"]) for record in records)

    candidates = []
    availability = {}
    for number, (class_index, item) in enumerate(targets, 1):
        chosen = select_observations(
            item,
            existing_observation_ids,
            args.train_candidates,
            args.val_candidates,
        )
        availability[item.label] = Counter(row["split"] for row in chosen)
        for row in chosen:
            row.update({
                "class_index": class_index,
                "label": item.label,
                "taxon_id": item.taxon_id,
                "scientific_name": item.scientific_name,
            })
            candidates.append(row)
            existing_observation_ids.add(row["observation_id"])
        print(f"Queried {number}/{len(targets)}: {item.label}: {dict(availability[item.label])}")
        # iNaturalist asks clients to remain below 60 API requests per minute.
        time.sleep(1.05)

    def fetch(row: dict) -> tuple[dict, Path]:
        destination = args.cache / f"{row['class_index']:03d}-{row['taxon_id']}" / f"{row['observation_id']}-{row['photo_id']}.jpg"
        if not destination.exists():
            download_jpeg(row["photo_url"], destination)
        else:
            with Image.open(destination) as image:
                image.load()
        return row, destination

    downloaded = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
        futures = {executor.submit(fetch, row): row for row in candidates}
        for number, future in enumerate(as_completed(futures), 1):
            try:
                downloaded.append(future.result())
            except Exception as error:
                row = futures[future]
                failures.append({"observation_id": row["observation_id"], "error": str(error)})
            if number % 200 == 0 or number == len(futures):
                print(f"Downloaded/verified {number}/{len(futures)} targeted photos; {len(failures)} failures")

    detector = MegaDetector(args.megadetector, args.device)
    image_dirs = {split: args.output / "images" / split for split in ("train", "val")}
    label_dirs = {split: args.output / "labels" / split for split in ("train", "val")}
    for directory in [*image_dirs.values(), *label_dirs.values()]:
        directory.mkdir(parents=True, exist_ok=True)

    added_counts = Counter()
    no_detection_counts = Counter()
    audit = audit_path.open("a")
    downloaded.sort(key=lambda item: (item[0]["class_index"], item[0]["observation_id"]))
    for offset in range(0, len(downloaded), args.batch_size):
        batch_rows = downloaded[offset : offset + args.batch_size]
        images = []
        valid = []
        for row, path in batch_rows:
            try:
                images.append(Image.open(path).convert("RGB"))
                valid.append((row, path))
            except Exception as error:
                failures.append({"observation_id": row["observation_id"], "error": str(error)})
        detections = detector.detect_many(images, args.threshold)
        for (row, path), image, boxes in zip(valid, images, detections):
            stem = f"rareinat-{row['class_index']:03d}-{row['observation_id']}-{row['photo_id']}"
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
                lines.append(f"{row['class_index']} " + " ".join(f"{value:.8f}" for value in normalized) + "\n")
                details.append({"bbox_xyxy": box, "confidence": confidence, "label": row["label"]})
            status = "ok" if lines else "no_detection"
            if lines:
                shutil.copy2(path, image_dirs[row["split"]] / f"{stem}.jpg")
                (label_dirs[row["split"]] / f"{stem}.txt").write_text("".join(lines))
                added_counts[(row["split"], row["label"])] += len(lines)
            else:
                no_detection_counts[row["label"]] += 1
            record = {
                **row,
                "source": path.relative_to(args.cache).as_posix(),
                "source_page": SOURCE_PAGE,
                "observation_url": f"https://www.inaturalist.org/observations/{row['observation_id']}",
                "status": status,
                "localization": "MegaDetector animal box; species from research-grade observation",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "boxes": details,
            }
            audit.write(json.dumps(record) + "\n")
            audit.flush()
        completed = min(offset + args.batch_size, len(downloaded))
        if completed % 160 == 0 or completed == len(downloaded):
            print(f"Localized {completed}/{len(downloaded)} targeted photos")
    audit.close()

    (args.output / "rare-inaturalist-failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    write_dataset_yaml(args.output, labels)
    removed = prune_exact_duplicates(args.output)
    summary = {
        "target": args.target,
        "target_classes": len(targets),
        "candidate_photos": len(candidates),
        "downloaded_photos": len(downloaded),
        "added_boxes": {f"{split}:{label}": count for (split, label), count in sorted(added_counts.items())},
        "classes_with_new_boxes": len({label for (_, label) in added_counts}),
        "no_detection_photos": dict(sorted(no_detection_counts.items())),
        "download_failures": failures,
        "duplicates_removed": removed,
    }
    (args.output / "rare-inaturalist-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} targeted downloads failed; rerun to resume")


if __name__ == "__main__":
    main()
