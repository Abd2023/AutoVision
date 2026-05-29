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
import hashlib
import random
import shutil
import warnings
from collections import Counter
from dataclasses import dataclass
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


@dataclass(frozen=True)
class SourceRecord:
    path: Path
    sha256: str
    duplicate_group_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze AutoVision raw data into processed splits.")
    parser.add_argument("--raw-root", default="data/new_synthetic_car_dataset_all_data")
    parser.add_argument("--output-root", default="data/processed")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-per-class", type=int, default=1000)
    parser.add_argument("--max-f1", type=int, default=1000)
    parser.add_argument(
        "--dedupe-exact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove exact duplicate files within each class before splitting.",
    )
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_records(files: list[Path], dedupe_exact: bool) -> tuple[list[SourceRecord], int]:
    by_hash: dict[str, list[Path]] = {}
    for path in files:
        file_hash = sha256_file(path)
        by_hash.setdefault(file_hash, []).append(path)

    records: list[SourceRecord] = []
    duplicate_count = 0
    for file_hash, grouped_paths in by_hash.items():
        duplicate_group_size = len(grouped_paths)
        if duplicate_group_size > 1:
            duplicate_count += duplicate_group_size - 1
        kept_paths = grouped_paths[:1] if dedupe_exact else grouped_paths
        for kept_path in kept_paths:
            records.append(
                SourceRecord(
                    path=kept_path,
                    sha256=file_hash,
                    duplicate_group_size=duplicate_group_size,
                )
            )
    return records, duplicate_count


def split_files(records: list[SourceRecord], val_ratio: float, test_ratio: float) -> dict[str, list[SourceRecord]]:
    count = len(records)
    test_count = round(count * test_ratio)
    val_count = round(count * val_ratio)
    train_count = count - val_count - test_count
    return {
        "train": records[:train_count],
        "val": records[train_count : train_count + val_count],
        "test": records[train_count + val_count :],
    }


def make_canvas(image: Image.Image, image_size: int, background: str) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    has_alpha = image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info)
    if has_alpha:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rgba_image = image.convert("RGBA")
        if background == "black":
            alpha_background = (0, 0, 0, 255)
        else:
            alpha_background = (255, 255, 255, 255)
        composited = Image.new("RGBA", rgba_image.size, alpha_background)
        image = Image.alpha_composite(composited, rgba_image).convert("RGB")
    else:
        image = image.convert("RGB")
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


def write_manifest(output_root: Path, rows: list[dict[str, str | int]]) -> None:
    manifest_path = output_root / "split_manifest.csv"
    fieldnames = [
        "split",
        "class_name",
        "processed_path",
        "source_path",
        "source_sha256",
        "duplicate_group_size",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    duplicate_counts = Counter()
    manifest_rows: list[dict[str, str | int]] = []

    for class_name in PROJECT_CLASSES:
        files = image_files(raw_root / class_name)
        records, duplicate_count = build_source_records(files, args.dedupe_exact)
        random.shuffle(records)

        class_limit = args.max_f1 if class_name == "F1" else args.max_per_class
        if class_limit > 0:
            records = records[:class_limit]
        selected_counts[class_name] = len(records)
        duplicate_counts[class_name] = duplicate_count

        for split_name, split_records in split_files(records, args.val_ratio, args.test_ratio).items():
            for index, record in enumerate(split_records):
                destination = output_root / split_name / class_name / f"{class_name.lower()}_{index:05d}.jpg"
                if save_processed(record.path, destination, args.image_size, args.background):
                    written[(split_name, class_name)] += 1
                    manifest_rows.append(
                        {
                            "split": split_name,
                            "class_name": class_name,
                            "processed_path": str(destination.with_suffix(".jpg")),
                            "source_path": str(record.path),
                            "source_sha256": record.sha256,
                            "duplicate_group_size": record.duplicate_group_size,
                        }
                    )
                else:
                    skipped[class_name] += 1

    write_manifest(output_root, manifest_rows)

    print("Selected from data/new_synthetic_car_dataset_all_data:")
    for class_name in PROJECT_CLASSES:
        print(f"  {class_name:14s} {selected_counts[class_name]:5d}")

    print("\nExact duplicates removed before split:")
    for class_name in PROJECT_CLASSES:
        print(f"  {class_name:14s} {duplicate_counts[class_name]:5d}")

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
