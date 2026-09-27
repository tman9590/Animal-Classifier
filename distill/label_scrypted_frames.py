#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

from catalog import read_catalog
from label_images import write_dataset_yaml, yolo_box
from smoothing import TemporalSmoother
from teacher import BioClipClassifier, MegaDetector, crop_box


ROOT = Path(__file__).parents[1]


def frame_order(path: Path) -> tuple[int, str]:
    suffix = path.stem.rsplit("-", 1)[-1]
    return (int(suffix) if suffix.isdigit() else 0, path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo-label Scrypted NVR event frames")
    parser.add_argument("frames", type=Path)
    parser.add_argument("--catalog", type=Path, default=ROOT / "species" / "north-carolina.json")
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "yolo-dataset")
    parser.add_argument("--megadetector", type=Path, default=ROOT / "work" / "models" / "MDV6-yolov9-c.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--detection-threshold", type=float, default=0.20)
    parser.add_argument("--unknown-threshold", type=float, default=0.25)
    parser.add_argument("--smoothing-alpha", type=float, default=0.35)
    parser.add_argument(
        "--poultry-cameras",
        nargs="*",
        default=["136", "292", "58"],
        help="Scrypted camera ids whose bird detections should be classified against the poultry subset",
    )
    args = parser.parse_args()

    metadata, species = read_catalog(args.catalog)
    labels = metadata["labels"]
    label_indexes = {label: index for index, label in enumerate(labels)}
    build_config = json.loads((ROOT / "distill" / "config.json").read_text())
    poultry_names = {item["scientific_name"] for item in build_config["required_species"]}
    poultry_labels = {item.label for item in species if item.scientific_name in poultry_names}
    detector = MegaDetector(args.megadetector, args.device)
    classifier = BioClipClassifier(species, args.device, ROOT / "work" / "models")

    # Scrypted directories are time buckets, not individual object tracks.
    groups: dict[tuple[Path, str], list[Path]] = defaultdict(list)
    for path in args.frames.rglob("*.jpg"):
        if path.stem.isdigit():
            sequence = "snapshots"
        elif "-" in path.stem and path.stem.rsplit("-", 1)[1].isdigit():
            sequence = path.stem.rsplit("-", 1)[0]
        else:
            sequence = path.stem  # Standalone crops have no trustworthy temporal order.
        groups[(path.parent, sequence)].append(path)

    args.output.mkdir(parents=True, exist_ok=True)
    audit = (args.output / "scrypted-audit.jsonl").open("w")
    written = 0
    for group_number, ((event, sequence), paths) in enumerate(sorted(groups.items())):
        camera_directory = next((part for part in event.parts if part.startswith("scrypted-") and part.endswith(".events")), "")
        camera_id = camera_directory.removeprefix("scrypted-").removesuffix(".events")
        poultry_camera = camera_id in set(args.poultry_cameras)
        smoother = TemporalSmoother(args.smoothing_alpha, 0.3, 30, args.unknown_threshold)
        split = "val" if int(hashlib.sha256(event.relative_to(args.frames).as_posix().encode()).hexdigest(), 16) % 5 == 0 else "train"
        image_dir = args.output / "images" / split
        label_dir = args.output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        previous_timestamp = None
        for frame_number, source in enumerate(sorted(paths, key=frame_order)):
            if source.stem.isdigit():
                timestamp = int(source.stem)
                if previous_timestamp is not None and timestamp - previous_timestamp > 30_000:
                    smoother = TemporalSmoother(args.smoothing_alpha, 0.3, 30, args.unknown_threshold)
                previous_timestamp = timestamp
            try:
                image = Image.open(source).convert("RGB")
            except Exception as error:
                print(f"skip {source}: {error}")
                continue
            detections = detector.detect(image, args.detection_threshold)
            probabilities = classifier.classify([crop_box(image, box) for box, _ in detections])
            if poultry_camera:
                for scores in probabilities:
                    # Retain regional mammals/reptiles instead of forcing every animal into poultry.
                    best_label = max(scores, key=scores.get)
                    bird_labels = {item.label for item in species if item.iconic_taxon == "Aves"}
                    if best_label not in bird_labels:
                        continue
                    total = sum(score for label, score in scores.items() if label in poultry_labels)
                    if total:
                        scores.update(
                            {label: score / total if label in poultry_labels else 0.0 for label, score in scores.items()}
                        )
            smoothed = smoother.update(frame_number, [(box, scores) for (box, _), scores in zip(detections, probabilities)])
            annotations = [
                (label_indexes[label], *yolo_box(box, image.width, image.height))
                for ((box, _), (_, label, _)) in zip(detections, smoothed)
            ]
            audit.write(json.dumps({
                "source": source.relative_to(args.frames).as_posix(), "split": split,
                "boxes": [{"xyxy": box, "label": label, "confidence": confidence}
                          for ((box, _), (_, label, confidence)) in zip(detections, smoothed)],
            }) + "\n")
            audit.flush()
            if not annotations:
                continue
            relative = source.relative_to(args.frames)
            stem = "scrypted-" + hashlib.sha1(relative.as_posix().encode()).hexdigest()[:16]
            shutil.copy2(source, image_dir / f"{stem}.jpg")
            (label_dir / f"{stem}.txt").write_text(
                "".join(f"{index} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n" for index, x, y, w, h in annotations)
            )
            written += 1
        print(f"Labeled Scrypted event {group_number + 1}/{len(groups)}")

    audit.close()
    write_dataset_yaml(args.output, labels)
    print(f"Added {written} Scrypted frames to {args.output}")


if __name__ == "__main__":
    main()
