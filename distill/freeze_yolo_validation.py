#!/usr/bin/env python3
"""Freeze the current validation membership as an immutable image list."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source_yaml = yaml.safe_load((args.dataset / "dataset.yaml").read_text())
    val_images = sorted((args.dataset / "images" / "val").glob("*.jpg"))
    args.output.mkdir(parents=True, exist_ok=True)
    val_list = args.output / "val.txt"
    val_list.write_text("".join(str(path.resolve()) + "\n" for path in val_images))
    frozen = {
        "path": str(args.dataset.resolve()),
        "train": str(val_list.resolve()),
        "val": str(val_list.resolve()),
        "names": source_yaml["names"],
    }
    (args.output / "dataset.yaml").write_text(yaml.safe_dump(frozen, sort_keys=False, allow_unicode=True))
    manifest = {
        "source_dataset": str(args.dataset.resolve()),
        "validation_images": len(val_images),
        "validation_list": str(val_list.resolve()),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
