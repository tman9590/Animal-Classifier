#!/usr/bin/env python3
"""Create a linear model soup from two compatible Ultralytics checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def ordered_names(model: YOLO) -> list[str]:
    names = model.names
    return [names[index] for index in range(len(names))]


def blend_checkpoints(first_path: Path, second_path: Path, output: Path, alpha: float) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    first = YOLO(first_path)
    second = YOLO(second_path)
    if ordered_names(first) != ordered_names(second):
        raise ValueError("Checkpoint class orders do not match")

    first_state = first.model.state_dict()
    second_state = second.model.state_dict()
    if first_state.keys() != second_state.keys():
        raise ValueError("Checkpoint architectures do not match")

    blended = {}
    for name, first_value in first_state.items():
        second_value = second_state[name]
        if first_value.shape != second_value.shape or first_value.dtype != second_value.dtype:
            raise ValueError(f"Incompatible tensor {name}")
        if torch.is_floating_point(first_value):
            blended[name] = torch.lerp(first_value.float(), second_value.float(), alpha).to(second_value.dtype)
        else:
            blended[name] = second_value.clone()

    second.model.load_state_dict(blended, strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    second.save(output)

    saved = YOLO(output)
    if ordered_names(saved) != ordered_names(second):
        raise ValueError("Saved checkpoint class order changed")
    print(f"Saved blend alpha={alpha:g} (0={first_path.name}, 1={second_path.name}) to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight assigned to the second checkpoint (default: 0.5)")
    args = parser.parse_args()
    blend_checkpoints(args.first, args.second, args.output, args.alpha)


if __name__ == "__main__":
    main()
