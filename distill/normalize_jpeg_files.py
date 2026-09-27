#!/usr/bin/env python3
"""Convert files named .jpg but encoded as another format into real JPEGs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from label_images import prune_exact_duplicates


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    records = []
    for path in sorted((args.dataset / "images").rglob("*.jpg")):
        with Image.open(path) as image:
            source_format = image.format
            image.load()
            if source_format in {"JPEG", "MPO"}:
                continue
            before = digest(path)
            temporary = path.with_name(path.stem + ".normalized.jpg")
            image.convert("RGB").save(temporary, format="JPEG", quality=95)
        temporary.replace(path)
        records.append({
            "path": path.relative_to(args.dataset).as_posix(),
            "source_format": source_format,
            "source_sha256": before,
            "normalized_sha256": digest(path),
        })
    log_path = args.dataset / "jpeg-normalization.jsonl"
    log_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    removed = prune_exact_duplicates(args.dataset)
    print(f"Normalized {len(records)} mislabeled image files to JPEG")
    print(f"Removed {len(removed)} duplicates after normalization")


if __name__ == "__main__":
    main()
