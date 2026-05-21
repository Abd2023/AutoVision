"""
Freeze data/raw into train/val/test folders under data/processed.

This script does not delete data/raw. It creates a reproducible training dataset:
    data/processed/train/<CLASS>/
    data/processed/val/<CLASS>/
    data/processed/test/<CLASS>/

Default behavior keeps every class except F1, which is capped to avoid swamping
the smaller body-type classes.
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError


PROJECT_CLASSES = [
    "F1",
    "HATCHBACK",
    "MICRO",
    "PICK_UP",
    "SEDAN",
    "STATION_WAGON",
    "SUV",
    "VAN",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze AutoVision raw data into processed splits.")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--output-root", default="data/processed")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-per-class", type=int, default=1000)
    parser.add_argument("--max-f1", type=int, default=1000)
    parser.add_argument("--clear", action="store_true", help="Clear output-root before writing.")
    parser.add_argument(
        "--background",
        choices=["none", "white", "black", "blur"],
        default="white",
        help="How to fill padded background around each resized image.",
    )
    return parser.parse_args()


def image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [path for path in sorted(folder.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]


def clear_output(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def split_files(files: list[Path], val_ratio: float, test_ratio: float) -> dict[str, list[Path]]:
    count = len(files)
    test_count = round(count * test_ratio)
    val_count = round(count * val_ratio)
    train_count = count - val_count - test_count
    return {
        "train": files[:train_count],
        "val": files[train_count : train_count + val_count],
        "test": files[train_count + val_count :],
    }


def make_canvas(image: Image.Image, image_size: int, background: str) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)

    if background == "none":
        canvas = Image.new("RGB", (image_size, image_size), "black")
    elif background == "black":
        canvas = Image.new("RGB", (image_size, image_size), "black")
    elif background == "blur":
        canvas = ImageOps.fit(image.copy(), (image_size, image_size), method=Image.Resampling.BICUBIC)
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=12))
    else:
        canvas = Image.new("RGB", (image_size, image_size), "white")

    x = (image_size - image.width) // 2
    y = (image_size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def save_processed(source: Path, destination: Path, image_size: int, background: str) -> bool:
    try:
        with Image.open(source) as image:
            processed = make_canvas(image, image_size, background)
            destination.parent.mkdir(parents=True, exist_ok=True)
            processed.save(destination.with_suffix(".jpg"), "JPEG", quality=95)
            return True
    except (OSError, UnidentifiedImageError, ValueError):
        return False


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    random.seed(args.seed)

    if args.clear:
        clear_output(output_root)
    else:
        output_root.mkdir(parents=True, exist_ok=True)

    written = Counter()
    skipped = Counter()
    selected_counts = Counter()

    for class_name in PROJECT_CLASSES:
        files = image_files(raw_root / class_name)
        random.shuffle(files)

        class_limit = args.max_f1 if class_name == "F1" else args.max_per_class
        if class_limit > 0:
            files = files[:class_limit]
        selected_counts[class_name] = len(files)

        for split_name, split_paths in split_files(files, args.val_ratio, args.test_ratio).items():
            for index, source in enumerate(split_paths):
                destination = output_root / split_name / class_name / f"{class_name.lower()}_{index:05d}.jpg"
                if save_processed(source, destination, args.image_size, args.background):
                    written[(split_name, class_name)] += 1
                else:
                    skipped[class_name] += 1

    print("Selected from data/raw:")
    for class_name in PROJECT_CLASSES:
        print(f"  {class_name:14s} {selected_counts[class_name]:5d}")

    print("\nWritten to data/processed:")
    for split_name in ["train", "val", "test"]:
        print(f"  {split_name}:")
        for class_name in PROJECT_CLASSES:
            print(f"    {class_name:14s} {written[(split_name, class_name)]:5d}")

    if skipped:
        print("\nSkipped unreadable during processing:")
        for class_name, count in sorted(skipped.items()):
            print(f"  {class_name:14s} {count:5d}")


if __name__ == "__main__":
    main()
