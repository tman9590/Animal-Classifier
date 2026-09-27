#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image

from catalog import read_catalog
from label_images import yolo_box
from smoothing import TemporalSmoother
from teacher import BioClipClassifier, MegaDetector, crop_box


ROOT = Path(__file__).parents[1]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo-label camera clips with the complete teacher pipeline")
    parser.add_argument("clips", type=Path, help="Directory containing representative camera clips")
    parser.add_argument("--catalog", type=Path, default=ROOT / "species" / "north-carolina.json")
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "yolo-dataset")
    parser.add_argument("--megadetector", type=Path, default=ROOT / "work" / "models" / "MDV6-yolov10-c.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--sample-fps", type=float, default=2)
    parser.add_argument("--detection-threshold", type=float, default=0.25)
    parser.add_argument("--unknown-threshold", type=float, default=0.25)
    parser.add_argument("--smoothing-alpha", type=float, default=0.35)
    args = parser.parse_args()

    _, species = read_catalog(args.catalog)
    label_indexes = {item.label: index for index, item in enumerate(species)}
    label_indexes["unknown"] = len(species)
    detector = MegaDetector(args.megadetector, args.device)
    classifier = BioClipClassifier(species, args.device, ROOT / "work" / "models")

    videos = sorted(path for path in args.clips.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)
    for video_number, video in enumerate(videos):
        capture = cv2.VideoCapture(str(video))
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 30
        stride = max(1, round(source_fps / args.sample_fps))
        smoother = TemporalSmoother(args.smoothing_alpha, 0.3, max(1, round(args.sample_fps * 10)), args.unknown_threshold)
        split = "val" if video_number % 5 == 0 else "train"
        image_dir = args.output / "images" / split
        label_dir = args.output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        frame_number = 0
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frame_number += 1
            if frame_number % stride:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            detections = detector.detect(image, args.detection_threshold)
            probabilities = classifier.classify([crop_box(image, box) for box, _ in detections])
            smoothed = smoother.update(frame_number, [(box, scores) for (box, _), scores in zip(detections, probabilities)])
            annotations = []
            for ((box, _), (_, label, _)) in zip(detections, smoothed):
                annotations.append((label_indexes[label], *yolo_box(box, image.width, image.height)))
            if not annotations:
                continue
            stem = f"{video.stem}-{frame_number:08d}"
            image.save(image_dir / f"{stem}.jpg", quality=92)
            (label_dir / f"{stem}.txt").write_text(
                "".join(f"{index} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n" for index, x, y, w, h in annotations)
            )
        capture.release()
        print(f"Labeled {video}")


if __name__ == "__main__":
    main()
