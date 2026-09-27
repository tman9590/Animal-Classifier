#!/usr/bin/env python3
"""Clip YOLO boxes to image boundaries for a selected filename prefix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

    changed_files = 0
    changed_boxes = 0
    dropped_boxes = 0
    for split in ("train", "val"):
        for path in sorted((args.dataset / "labels" / split).glob(f"{args.prefix}*.txt")):
            output = []
            changed = False
            for line in path.read_text().splitlines():
                fields = line.split()
                class_id = fields[0]
                x, y, w, h = map(float, fields[1:5])
                x1, y1 = max(0.0, x - w / 2), max(0.0, y - h / 2)
                x2, y2 = min(1.0, x + w / 2), min(1.0, y + h / 2)
                if x2 <= x1 or y2 <= y1:
                    dropped_boxes += 1
                    changed = True
                    continue
                clipped = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
                if any(abs(a - b) > 1e-9 for a, b in zip((x, y, w, h), clipped)):
                    changed_boxes += 1
                    changed = True
                output.append(f"{class_id} " + " ".join(f"{value:.8f}" for value in clipped))
            if changed:
                path.write_text("\n".join(output) + ("\n" if output else ""))
                changed_files += 1
    summary = {"changed_files": changed_files, "changed_boxes": changed_boxes, "dropped_boxes": dropped_boxes}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
