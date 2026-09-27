#!/usr/bin/env python3
"""Recover exact species labels for selected research-grade public images.

The original public pipeline localized animals with MegaDetector but changed a
box to ``unknown`` whenever BioCLIP did not repeat the observation identity.
For explicitly selected rare classes, restore those boxes to the licensed
research-grade iNaturalist observation's catalog class. Existing known labels
and box coordinates are left unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from catalog import read_catalog


ROOT = Path(__file__).parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "species" / "north-carolina.json")
    parser.add_argument(
        "--target-report",
        type=Path,
        default=ROOT / "work" / "dataset-production" / "dataset-audit-ir-final.json",
    )
    parser.add_argument("--source", type=Path, default=ROOT / "work" / "source-images")
    parser.add_argument("--dataset", type=Path, default=ROOT / "work" / "dataset-production")
    args = parser.parse_args()

    _, species = read_catalog(args.catalog)
    unknown_index = len(species)
    target_labels = set(json.loads(args.target_report.read_text())["zero_support_classes"])
    target_indexes = {index for index, item in enumerate(species) if item.label in target_labels}

    attribution = {row["file"]: row for row in csv.DictReader((args.source / "attribution.csv").open())}
    audit_path = args.dataset / "public-audit.jsonl"
    changes = []
    counts = Counter()
    for line in audit_path.read_text().splitlines():
        record = json.loads(line)
        source = record.get("source")
        row = attribution.get(source)
        if row is None or record.get("status") != "ok":
            continue
        class_index = int(row["class_index"])
        if class_index not in target_indexes:
            continue
        split = record["split"]
        stem = "public-" + Path(source).stem
        label_path = args.dataset / "labels" / split / f"{stem}.txt"
        if not label_path.exists():
            if record.get("boxes"):
                raise FileNotFoundError(f"Audit has boxes but label is missing: {label_path}")
            continue
        label_lines = label_path.read_text().splitlines()
        if len(label_lines) != len(record.get("boxes", [])):
            raise ValueError(f"Audit/label box count mismatch: {label_path}")
        rewritten = []
        changed = 0
        for label_line in label_lines:
            fields = label_line.split()
            if int(fields[0]) == unknown_index:
                fields[0] = str(class_index)
                changed += 1
            rewritten.append(" ".join(fields))
        if not changed:
            continue
        label_path.write_text("\n".join(rewritten) + "\n")
        counts[(split, species[class_index].label)] += changed
        changes.append(
            {
                "source": source,
                "output_label": label_path.relative_to(args.dataset).as_posix(),
                "split": split,
                "class_index": class_index,
                "label": species[class_index].label,
                "boxes_restored": changed,
                "basis": "research-grade iNaturalist observation identity; existing MegaDetector localization",
                "observation_url": row["observation_url"],
                "license": row["license"],
                "attribution": row["attribution"],
            }
        )

    log_path = args.dataset / "public-identity-relabels.jsonl"
    log_path.write_text("".join(json.dumps(record) + "\n" for record in changes))
    summary = {
        "target_classes": len(target_indexes),
        "images_relabelled": len(changes),
        "boxes_relabelled": sum(counts.values()),
        "boxes_by_split_and_class": {
            f"{split}:{label}": count for (split, label), count in sorted(counts.items())
        },
    }
    (args.dataset / "public-identity-relabels-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
