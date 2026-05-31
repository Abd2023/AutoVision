"""
Export a model-only fp32 checkpoint for inference/submission.

This removes optimizer and scheduler state without changing model weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from train_resnet50 import project_path


KEEP_KEYS = [
    "epoch",
    "phase",
    "model_source",
    "model_name",
    "model_state_dict",
    "class_names",
    "class_to_idx",
    "image_size",
    "imagenet_mean",
    "imagenet_std",
    "train_counts",
    "val_counts",
    "test_counts",
    "best_val_balanced_acc",
    "best_val_acc",
    "args",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an inference-only fp32 checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-size-mb", type=float, default=95.0)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def fp32_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    exported: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        tensor = value.detach().cpu() if torch.is_tensor(value) else value
        if torch.is_tensor(tensor) and tensor.is_floating_point():
            tensor = tensor.float()
        exported[key] = tensor
    return exported


def main() -> None:
    args = parse_args()
    checkpoint_path = project_path(args.checkpoint)
    output_path = project_path(args.output)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path)
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_state_dict.")

    exported = {key: checkpoint[key] for key in KEEP_KEYS if key in checkpoint}
    exported["model_state_dict"] = fp32_state_dict(checkpoint["model_state_dict"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, output_path)

    size_mb = output_path.stat().st_size / 1_000_000
    print(f"Exported fp32 inference checkpoint: {output_path}")
    print(f"Size: {size_mb:.2f} MB")
    if size_mb > args.max_size_mb:
        print(f"WARNING: exported checkpoint is larger than {args.max_size_mb:.2f} MB.")


if __name__ == "__main__":
    main()
