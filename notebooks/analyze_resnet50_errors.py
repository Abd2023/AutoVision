"""
Generate per-image error analysis for an AutoVision checkpoint.

Outputs:
    predictions_<split>.csv
    mistakes_<split>.csv
    confusion_counts_<split>.csv
    contact_sheets/<split>_<ACTUAL>_to_<PREDICTED>.jpg
    contact_sheets/<split>_all_mistakes_page_XX.jpg
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from train_resnet50 import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PROJECT_CLASSES,
    build_model,
    choose_device,
    load_checkpoint,
    project_path,
)


DEFAULT_PAIRS = [
    "HATCHBACK:MICRO",
    "HATCHBACK:SEDAN",
    "HATCHBACK:VAN",
    "SUV:STATION_WAGON",
    "STATION_WAGON:SEDAN",
]


class PathImageFolder(datasets.ImageFolder):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        image, target = super().__getitem__(index)
        path = self.samples[index][0]
        return image, target, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create AutoVision per-image prediction and mistake reports.")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--checkpoint", default="notebooks/outputs/resnet50/best_resnet50.pt")
    parser.add_argument("--output-dir", default="notebooks/outputs/error_analysis/resnet50")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tta", choices=["none", "hflip"], default="none")
    parser.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS, help="Pairs formatted as ACTUAL:PREDICTED.")
    parser.add_argument("--max-per-pair", type=int, default=80)
    parser.add_argument("--max-all-mistakes", type=int, default=240)
    parser.add_argument("--sheet-cols", type=int, default=5)
    parser.add_argument("--thumb-width", type=int, default=180)
    parser.add_argument("--thumb-height", type=int, default=135)
    return parser.parse_args()


def eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], list[str]]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = checkpoint.get("class_names", PROJECT_CLASSES)
    checkpoint_args = checkpoint.get("args", {})
    model_args = SimpleNamespace(
        model_source=checkpoint.get("model_source", checkpoint_args.get("model_source", "torchvision")),
        model_name=checkpoint.get("model_name", checkpoint_args.get("model_name", "resnet50")),
        pretrained=False,
        dropout=float(checkpoint_args.get("dropout", 0.35)),
    )
    model = build_model(model_args, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, class_names


def make_dataset(data_root: Path, split: str, image_size: int, class_names: list[str]) -> PathImageFolder:
    split_root = data_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Missing split folder: {split_root}")
    dataset = PathImageFolder(split_root, transform=eval_transform(image_size))
    if dataset.classes != class_names:
        raise ValueError(f"Dataset class order does not match checkpoint. Dataset={dataset.classes}, checkpoint={class_names}")
    return dataset


@torch.no_grad()
def predict_rows(
    model: torch.nn.Module,
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
            flipped_logits = model(torch.flip(images, dims=[3]))
            logits = (logits + flipped_logits) / 2.0
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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_prediction_tables(output_dir: Path, split: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fieldnames = [
        "actual_class",
        "predicted_class",
        "confidence",
        "correct",
        "top1_class",
        "top1_prob",
        "top2_class",
        "top2_prob",
        "top3_class",
        "top3_prob",
        "image_path",
    ]
    mistakes = [row for row in rows if not row["correct"]]
    write_csv(output_dir / f"predictions_{split}.csv", rows, fieldnames)
    write_csv(output_dir / f"mistakes_{split}.csv", mistakes, fieldnames)

    counts = Counter((row["actual_class"], row["predicted_class"]) for row in mistakes)
    confusion_rows = [
        {"actual_class": actual, "predicted_class": predicted, "count": count}
        for (actual, predicted), count in sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    ]
    write_csv(output_dir / f"confusion_counts_{split}.csv", confusion_rows, ["actual_class", "predicted_class", "count"])
    return mistakes


def parse_pairs(values: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"Invalid pair '{value}'. Use ACTUAL:PREDICTED.")
        actual, predicted = [part.strip() for part in value.split(":", 1)]
        if actual not in PROJECT_CLASSES or predicted not in PROJECT_CLASSES:
            raise ValueError(f"Invalid class in pair '{value}'.")
        pairs.append((actual, predicted))
    return pairs


def draw_wrapped_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, width_chars: int, font: ImageFont.ImageFont) -> None:
    x, y = xy
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line
        while len(line) > width_chars:
            lines.append(line[:width_chars])
            line = line[width_chars:]
        lines.append(line)
    for line in lines:
        draw.text((x, y), line, fill=(20, 20, 20), font=font)
        y += 12


def make_contact_sheet(
    rows: list[dict[str, Any]],
    output_path: Path,
    cols: int,
    thumb_width: int,
    thumb_height: int,
) -> None:
    if not rows:
        return

    font = ImageFont.load_default()
    label_height = 68
    rows_count = math.ceil(len(rows) / cols)
    sheet = Image.new("RGB", (cols * thumb_width, rows_count * (thumb_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)

    for index, row in enumerate(rows):
        col = index % cols
        row_index = index // cols
        x = col * thumb_width
        y = row_index * (thumb_height + label_height)
        image_path = Path(row["image_path"])

        try:
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
                paste_x = x + (thumb_width - image.width) // 2
                paste_y = y + (thumb_height - image.height) // 2
                sheet.paste(image, (paste_x, paste_y))
        except (OSError, UnidentifiedImageError, ValueError):
            draw.rectangle((x, y, x + thumb_width - 1, y + thumb_height - 1), outline=(180, 0, 0))
            draw.text((x + 4, y + 4), "unreadable", fill=(180, 0, 0), font=font)

        draw.rectangle((x, y, x + thumb_width - 1, y + thumb_height - 1), outline=(210, 210, 210))
        label = (
            f"{row['actual_class']} -> {row['predicted_class']}\n"
            f"conf={float(row['confidence']):.3f}\n"
            f"2:{row['top2_class']} {float(row['top2_prob']):.2f}  "
            f"3:{row['top3_class']} {float(row['top3_prob']):.2f}\n"
            f"{image_path.name}"
        )
        draw_wrapped_text(draw, (x + 4, y + thumb_height + 4), label, max(18, thumb_width // 7), font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def write_contact_sheets(
    output_dir: Path,
    split: str,
    mistakes: list[dict[str, Any]],
    pairs: list[tuple[str, str]],
    max_per_pair: int,
    max_all_mistakes: int,
    cols: int,
    thumb_width: int,
    thumb_height: int,
) -> None:
    contact_dir = output_dir / "contact_sheets"

    sorted_mistakes = sorted(mistakes, key=lambda row: float(row["confidence"]), reverse=True)
    all_pages = sorted_mistakes[:max_all_mistakes]
    page_size = max(1, cols * 8)
    for page_start in range(0, len(all_pages), page_size):
        page_rows = all_pages[page_start : page_start + page_size]
        page_number = page_start // page_size + 1
        make_contact_sheet(
            page_rows,
            contact_dir / f"{split}_all_mistakes_page_{page_number:02d}.jpg",
            cols,
            thumb_width,
            thumb_height,
        )

    for actual, predicted in pairs:
        pair_rows = [
            row
            for row in sorted_mistakes
            if row["actual_class"] == actual and row["predicted_class"] == predicted
        ][:max_per_pair]
        make_contact_sheet(
            pair_rows,
            contact_dir / f"{split}_{actual}_to_{predicted}.jpg",
            cols,
            thumb_width,
            thumb_height,
        )


def main() -> None:
    args = parse_args()
    data_root = project_path(args.data_root)
    checkpoint_path = project_path(args.checkpoint)
    output_dir = project_path(args.output_dir)
    device = choose_device(args.device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    model, checkpoint, class_names = load_model(checkpoint_path, device)
    image_size = int(checkpoint.get("image_size", 224))
    dataset = make_dataset(data_root, args.split, image_size, class_names)
    rows = predict_rows(model, dataset, class_names, args.batch_size, args.num_workers, device, args.tta)
    mistakes = write_prediction_tables(output_dir, args.split, rows)
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
