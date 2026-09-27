#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the single-stage Scrypted student model")
    parser.add_argument("--dataset", type=Path, default=ROOT / "work" / "yolo-dataset" / "dataset.yaml")
    parser.add_argument("--base", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--name", default="north-carolina-wildlife")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", type=Path, help="Resume an interrupted Ultralytics run from last.pt")
    args = parser.parse_args()
    if args.resume:
        model = YOLO(args.resume)
        # Ultralytics explicitly permits batch overrides on resume. This is
        # needed when a checkpoint was saved before a validation-time OOM.
        model.train(resume=True, batch=args.batch)
        return
    model = YOLO(args.base)
    model.train(
        data=str(args.dataset),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "work" / "training"),
        name=args.name,
        patience=15,
        seed=360,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
