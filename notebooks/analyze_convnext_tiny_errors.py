"""
Generate TTA-aware error analysis for a ConvNeXt-Tiny AutoVision checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import convnext_tiny

from analyze_resnet50_errors import (
    DEFAULT_PAIRS,
    PathImageFolder,
    make_dataset,
    parse_pairs,
    write_contact_sheets,
    write_prediction_tables,
)
from train_resnet50 import PROJECT_CLASSES, choose_device, load_checkpoint, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ConvNeXt-Tiny prediction and mistake reports.")
    parser.add_argument("--data-root", default="data/processed_convnext_tiny_320")
    parser.add_argument("--checkpoint", default="notebooks/outputs/convnext_tiny_highres_320_v1/best_convnext_tiny.pt")
    parser.add_argument("--output-dir", default="notebooks/outputs/error_analysis/convnext_tiny_highres_320_v1")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tta", choices=["none", "hflip"], default="hflip")
    parser.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS, help="Pairs formatted as ACTUAL:PREDICTED.")
    parser.add_argument("--max-per-pair", type=int, default=80)
    parser.add_argument("--max-all-mistakes", type=int, default=240)
    parser.add_argument("--sheet-cols", type=int, default=5)
    parser.add_argument("--thumb-width", type=int, default=180)
    parser.add_argument("--thumb-height", type=int, default=135)
    return parser.parse_args()


def build_convnext_tiny(num_classes: int, dropout: float) -> nn.Module:
    model = convnext_tiny(weights=None)
    norm_layer = model.classifier[0]
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        norm_layer,
        nn.Flatten(start_dim=1),
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any], list[str]]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = checkpoint.get("class_names", PROJECT_CLASSES)
    checkpoint_args = checkpoint.get("args", {})
    model_name = checkpoint.get("model_name", checkpoint_args.get("model_name", "convnext_tiny"))
    if model_name != "convnext_tiny":
        raise ValueError(f"This analyzer expects convnext_tiny, but checkpoint model_name is {model_name!r}.")

    model = build_convnext_tiny(len(class_names), float(checkpoint_args.get("dropout", 0.30)))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, class_names


@torch.no_grad()
def predict_rows(
    model: nn.Module,
    dataset: PathImageFolder,
    class_names: list[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    tta: str,
) -> list[dict[str, Any]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    rows: list[dict[str, Any]] = []

    for images, labels, paths in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        if tta == "hflip":
            logits = (logits + model(torch.flip(images, dims=[3]))) / 2.0
        probabilities = torch.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=min(3, len(class_names)), dim=1)

        for batch_index in range(images.size(0)):
            actual_idx = int(labels[batch_index].item())
            predicted_idx = int(top_indices[batch_index, 0].item())
            row = {
                "actual_class": class_names[actual_idx],
                "predicted_class": class_names[predicted_idx],
                "confidence": float(top_probs[batch_index, 0].item()),
                "correct": actual_idx == predicted_idx,
                "image_path": str(paths[batch_index]),
            }
            for rank in range(top_indices.size(1)):
                row[f"top{rank + 1}_class"] = class_names[int(top_indices[batch_index, rank].item())]
                row[f"top{rank + 1}_prob"] = float(top_probs[batch_index, rank].item())
            rows.append(row)
    return rows


def write_confusion_csv(path: Path, matrix: Any, class_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *class_names])
        for class_name, row in zip(class_names, matrix.tolist()):
            writer.writerow([class_name, *row])


def write_metric_reports(output_dir: Path, split: str, rows: list[dict[str, Any]], class_names: list[str]) -> None:
    y_true = [class_names.index(row["actual_class"]) for row in rows]
    y_pred = [class_names.index(row["predicted_class"]) for row in rows]
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": report_dict,
    }
    (output_dir / f"{split}_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / f"{split}_classification_report.txt").write_text(report_text, encoding="utf-8")
    write_confusion_csv(output_dir / f"{split}_confusion_matrix.csv", matrix, class_names)


def main() -> None:
    args = parse_args()
    data_root = project_path(args.data_root)
    checkpoint_path = project_path(args.checkpoint)
    output_dir = project_path(args.output_dir)
    device = choose_device(args.device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    model, checkpoint, class_names = load_model(checkpoint_path, device)
    image_size = int(checkpoint.get("image_size", 320))
    dataset = make_dataset(data_root, args.split, image_size, class_names)
    rows = predict_rows(model, dataset, class_names, args.batch_size, args.num_workers, device, args.tta)
    mistakes = write_prediction_tables(output_dir, args.split, rows)
    write_metric_reports(output_dir, args.split, rows, class_names)
    write_contact_sheets(
        output_dir=output_dir,
        split=args.split,
        mistakes=mistakes,
        pairs=parse_pairs(args.pairs),
        max_per_pair=args.max_per_pair,
        max_all_mistakes=args.max_all_mistakes,
        cols=args.sheet_cols,
        thumb_width=args.thumb_width,
        thumb_height=args.thumb_height,
    )

    print(f"Analyzed {len(rows)} {args.split} images.")
    print(f"Mistakes: {len(mistakes)}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
