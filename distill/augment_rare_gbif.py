#!/usr/bin/env python3
"""Add explicitly CC0/CC-BY GBIF images for still-unsupported species.

GBIF occurrence identification supplies the species label. MegaDetector is used
only to localize animals. One image is retained per occurrence, and all media
license checks are made on the media item itself rather than the occurrence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from catalog import USER_AGENT, read_catalog
from label_images import prune_exact_duplicates, write_dataset_yaml
from teacher import MegaDetector


ROOT = Path(__file__).parents[1]
GBIF_API = "https://api.gbif.org/v1"
GBIF_HOME = "https://www.gbif.org"


def api_get(path: str, params: dict) -> dict:
    url = f"{GBIF_API}/{path}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def occurrence_split(occurrence_key: int) -> str:
    digest = hashlib.sha256(f"gbif:{occurrence_key}".encode()).hexdigest()
    return "val" if int(digest, 16) % 5 == 0 else "train"


def allowed_license(value: str) -> bool:
    license_url = (value or "").lower().replace("https://", "http://")
    return (
        "creativecommons.org/publicdomain/zero/" in license_url
        or "creativecommons.org/licenses/by/" in license_url
    )


def direct_image_url(value: str) -> bool:
    path = urllib.parse.urlparse(value or "").path.lower()
    return path.endswith((".jpg", ".jpeg", ".png", ".webp"))


def observation_id(record: dict) -> int | None:
    text = " ".join(str(record.get(key) or "") for key in ("references", "occurrenceID"))
    match = re.search(r"inaturalist\.org/observations/(\d+)", text)
    return int(match.group(1)) if match else None


def exact_species(record: dict, scientific_name: str) -> bool:
    names = {str(record.get(key) or "").strip().lower() for key in ("species", "acceptedScientificName")}
    target = scientific_name.strip().lower()
    return any(name == target or name.startswith(target + " ") for name in names)


def select_occurrences(
    item,
    existing_references: set[str],
    existing_observation_ids: set[int],
    train_candidates: int,
    val_candidates: int,
) -> list[dict]:
    payload = api_get(
        "occurrence/search",
        {
            "scientific_name": item.scientific_name,
            "media_type": "StillImage",
            "occurrence_status": "PRESENT",
            "limit": 300,
        },
    )
    quotas = {"train": train_candidates, "val": val_candidates}
    selected = []
    for record in payload.get("results", []):
        if not exact_species(record, item.scientific_name):
            continue
        key = int(record["key"])
        obs_id = observation_id(record)
        reference = str(record.get("references") or f"{GBIF_HOME}/occurrence/{key}")
        if reference in existing_references or (obs_id is not None and obs_id in existing_observation_ids):
            continue
        media = next(
            (
                entry
                for entry in record.get("media", [])
                if direct_image_url(entry.get("identifier") or "")
                and allowed_license(entry.get("license") or "")
            ),
            None,
        )
        if media is None:
            continue
        split = occurrence_split(key)
        if quotas[split] <= 0:
            continue
        selected.append(
            {
                "occurrence_key": key,
                "observation_id": obs_id,
                "occurrence_id": record.get("occurrenceID") or "",
                "references": reference,
                "dataset_key": record.get("datasetKey") or "",
                "publishing_org_key": record.get("publishingOrgKey") or "",
                "media_url": media["identifier"],
                "license": media.get("license") or "",
                "creator": media.get("creator") or record.get("recordedBy") or "",
                "rights_holder": media.get("rightsHolder") or "",
                "media_reference": media.get("references") or "",
                "split": split,
            }
        )
        quotas[split] -= 1
        existing_references.add(reference)
        if obs_id is not None:
            existing_observation_ids.add(obs_id)
        if not any(quotas.values()):
            break
    return selected


def download_jpeg(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read()
    with Image.open(io.BytesIO(data)) as image:
        image.convert("RGB").save(destination, format="JPEG", quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "species" / "north-carolina.json")
    parser.add_argument(
        "--coverage-report",
        type=Path,
        default=ROOT / "work" / "dataset-production" / "dataset-audit-rare-v1.json",
    )
    parser.add_argument("--cache", type=Path, default=ROOT / "work" / "external" / "gbif-rare")
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "dataset-production")
    parser.add_argument("--megadetector", type=Path, default=ROOT / "work" / "models" / "MDV6-yolov9-c.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-candidates", type=int, default=40)
    parser.add_argument("--val-candidates", type=int, default=10)
    parser.add_argument("--download-workers", type=int, default=6)
    args = parser.parse_args()

    _, species = read_catalog(args.catalog)
    labels = [item.label for item in species] + ["unknown"]
    report = json.loads(args.coverage_report.read_text())
    target_labels = set(report["zero_support_classes"])
    targets = [(index, item) for index, item in enumerate(species) if item.label in target_labels]

    existing_references: set[str] = set()
    existing_observation_ids: set[int] = set()
    attribution_path = ROOT / "work" / "source-images" / "attribution.csv"
    if attribution_path.exists():
        for row in csv.DictReader(attribution_path.open()):
            try:
                existing_observation_ids.add(int(Path(row["file"]).stem.split("-")[0]))
            except (ValueError, IndexError):
                pass
    inat_audit = args.output / "rare-inaturalist-audit.jsonl"
    if inat_audit.exists():
        for line in inat_audit.read_text().splitlines():
            record = json.loads(line)
            if record.get("status") == "ok":
                existing_references.add(record.get("observation_url") or "")
                existing_observation_ids.add(int(record["observation_id"]))

    audit_path = args.output / "rare-gbif-audit.jsonl"
    prior_records = [json.loads(line) for line in audit_path.read_text().splitlines() if line] if audit_path.exists() else []
    successful_records = [record for record in prior_records if record.get("status") == "ok"]
    existing_keys = {int(record["occurrence_key"]) for record in successful_records}
    existing_references.update(record.get("references") or "" for record in successful_records)
    existing_observation_ids.update(
        int(record["observation_id"])
        for record in successful_records
        if record.get("observation_id") is not None
    )

    candidates = []
    availability = {}
    for number, (class_index, item) in enumerate(targets, 1):
        chosen = select_occurrences(
            item,
            existing_references,
            existing_observation_ids,
            args.train_candidates,
            args.val_candidates,
        )
        chosen = [row for row in chosen if row["occurrence_key"] not in existing_keys]
        availability[item.label] = Counter(row["split"] for row in chosen)
        for row in chosen:
            row.update(
                {
                    "class_index": class_index,
                    "label": item.label,
                    "taxon_id": item.taxon_id,
                    "scientific_name": item.scientific_name,
                }
            )
            candidates.append(row)
            existing_keys.add(row["occurrence_key"])
        print(f"Queried {number}/{len(targets)}: {item.label}: {dict(availability[item.label])}")
        time.sleep(0.25)

    def fetch(row: dict) -> tuple[dict, Path]:
        destination = args.cache / f"{row['class_index']:03d}" / f"{row['occurrence_key']}.jpg"
        if not destination.exists():
            download_jpeg(row["media_url"], destination)
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
                failures.append({"occurrence_key": row["occurrence_key"], "error": str(error)})
            if number % 50 == 0 or number == len(futures):
                print(f"Downloaded/verified {number}/{len(futures)} targeted photos; {len(failures)} failures")

    detector = MegaDetector(args.megadetector, args.device)
    image_dirs = {split: args.output / "images" / split for split in ("train", "val")}
    label_dirs = {split: args.output / "labels" / split for split in ("train", "val")}
    for directory in [*image_dirs.values(), *label_dirs.values()]:
        directory.mkdir(parents=True, exist_ok=True)

    added_counts = Counter()
    no_detection_counts = Counter()
    with audit_path.open("a") as audit:
        downloaded.sort(key=lambda item: (item[0]["class_index"], item[0]["occurrence_key"]))
        for offset in range(0, len(downloaded), args.batch_size):
            batch_rows = downloaded[offset : offset + args.batch_size]
            images = []
            valid = []
            for row, path in batch_rows:
                try:
                    images.append(Image.open(path).convert("RGB"))
                    valid.append((row, path))
                except Exception as error:
                    failures.append({"occurrence_key": row["occurrence_key"], "error": str(error)})
            detections = detector.detect_many(images, args.threshold)
            for (row, path), image, boxes in zip(valid, images, detections):
                stem = f"gbifrare-{row['class_index']:03d}-{row['occurrence_key']}"
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
                    "source_page": GBIF_HOME,
                    "occurrence_url": f"{GBIF_HOME}/occurrence/{row['occurrence_key']}",
                    "status": status,
                    "localization": "MegaDetector animal box; species from exact GBIF occurrence identification",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "boxes": details,
                }
                audit.write(json.dumps(record) + "\n")
                audit.flush()
            completed = min(offset + args.batch_size, len(downloaded))
            if completed % 80 == 0 or completed == len(downloaded):
                print(f"Localized {completed}/{len(downloaded)} targeted photos")

    (args.output / "rare-gbif-failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    write_dataset_yaml(args.output, labels)
    removed = prune_exact_duplicates(args.output)
    summary = {
        "target_classes": len(targets),
        "candidate_photos": len(candidates),
        "downloaded_photos": len(downloaded),
        "added_boxes": {f"{split}:{label}": count for (split, label), count in sorted(added_counts.items())},
        "classes_with_new_boxes": len({label for (_, label) in added_counts}),
        "no_detection_photos": dict(sorted(no_detection_counts.items())),
        "download_failures": failures,
        "duplicates_removed": removed,
    }
    (args.output / "rare-gbif-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} targeted downloads failed; rerun to resume")


if __name__ == "__main__":
    main()
