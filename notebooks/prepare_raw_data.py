"""
Build data/raw from downloaded source datasets.

Sources expected under data/:
    - Cars_Body_Type/
    - F1_new_upd_car_data/
    - HF_MICRO_car_images/ (optional, created by download_hf_micro.py)
    - HF_STATION_WAGON_car_images/ (optional, created by download_hf_station_wagon.py)
    - HF_BODY_car_images/<CLASS>/ (optional, created by download_hf_body_classes.py)
    - archive/stanford_cars_type/
    - archive/stanford_cars_type.csv

The source datasets are kept intact. Files are copied into the project class
folders under data/raw.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter
from pathlib import Path


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

CARS_BODY_TYPE_MAP = {
    "Hatchback": "HATCHBACK",
    "Pick-Up": "PICK_UP",
    "Sedan": "SEDAN",
    "SUV": "SUV",
    "VAN": "VAN",
}

STANFORD_FOLDER_MAP = {
    "Cab": "PICK_UP",
    "Hatchback": "HATCHBACK",
    "Minivan": "VAN",
    "Sedan": "SEDAN",
    "SUV": "SUV",
    "Van": "VAN",
    "Wagon": "STATION_WAGON",
}

MICRO_PATTERN = re.compile(
    r"(smart fortwo|fiat 500|mini cooper|geo metro|toyota iq|chevrolet spark|"
    r"hyundai i10|kia picanto|twingo|aygo|citroen c1|peugeot 108|volkswagen up|vw up)",
    re.IGNORECASE,
)

PICKUP_OTHER_PATTERN = re.compile(
    r"(ford ranger supercab|supercab)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AutoVision raw dataset folders.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--clear", action="store_true", help="Clear data/raw class folders before copying.")
    parser.add_argument(
        "--allow-partial-rebuild",
        action="store_true",
        help="Allow --clear even when existing raw classes have no available source folder.",
    )
    return parser.parse_args()


def ensure_raw_dirs(raw_root: Path) -> None:
    for class_name in PROJECT_CLASSES:
        (raw_root / class_name).mkdir(parents=True, exist_ok=True)


def clear_raw_dirs(raw_root: Path) -> None:
    for class_name in PROJECT_CLASSES:
        folder = raw_root / class_name
        folder.mkdir(parents=True, exist_ok=True)
        for path in folder.iterdir():
            if path.is_file():
                path.unlink()


def image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [path for path in sorted(folder.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]


def raw_class_counts(raw_root: Path) -> Counter[str]:
    return Counter({class_name: len(image_files(raw_root / class_name)) for class_name in PROJECT_CLASSES})


def available_source_classes(data_root: Path) -> set[str]:
    classes: set[str] = set()

    cars_body_type_root = data_root / "Cars_Body_Type"
    for split in ["train", "valid", "test"]:
        for source_class, target_class in CARS_BODY_TYPE_MAP.items():
            if image_files(cars_body_type_root / split / source_class):
                classes.add(target_class)

    if image_files(data_root / "F1_new_upd_car_data"):
        classes.add("F1")

    stanford_root = data_root / "archive" / "stanford_cars_type"
    for source_class, target_class in STANFORD_FOLDER_MAP.items():
        if image_files(stanford_root / source_class):
            classes.add(target_class)

    if (data_root / "archive" / "stanford_cars_type.csv").exists():
        classes.update({"MICRO", "PICK_UP"})

    if image_files(data_root / "HF_MICRO_car_images"):
        classes.add("MICRO")

    if image_files(data_root / "HF_STATION_WAGON_car_images"):
        classes.add("STATION_WAGON")

    hf_body_root = data_root / "HF_BODY_car_images"
    for class_name in PROJECT_CLASSES:
        if image_files(hf_body_root / class_name):
            classes.add(class_name)

    return classes


def guard_clear(raw_root: Path, data_root: Path, allow_partial_rebuild: bool) -> None:
    if allow_partial_rebuild:
        return

    existing = {class_name for class_name, count in raw_class_counts(raw_root).items() if count}
    if not existing:
        return

    available = available_source_classes(data_root)
    missing = sorted(existing - available)
    if missing:
        missing_text = ", ".join(missing)
        raise SystemExit(
            "Refusing to clear data/raw because these existing classes have no source data available: "
            f"{missing_text}. Restore/download the sources first, or pass --allow-partial-rebuild intentionally."
        )


def unique_destination(folder: Path, filename: str) -> Path:
    candidate = folder / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    index = 2
    while True:
        candidate = folder / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def copy_image(source: Path, raw_root: Path, target_class: str, prefix: str, counts: Counter[str]) -> None:
    safe_name = source.name.replace("/", "_").replace("\\", "_")
    destination = unique_destination(raw_root / target_class, f"{prefix}__{safe_name}")
    shutil.copy2(source, destination)
    counts[target_class] += 1


def target_for_stanford_folder(source_class: str, source: Path) -> str:
    filename_lower = source.name.lower()
    if source_class == "Wagon" and "ford e-series wagon van" in filename_lower:
        return "VAN"
    return STANFORD_FOLDER_MAP[source_class]


def copy_cars_body_type(data_root: Path, raw_root: Path, counts: Counter[str]) -> None:
    source_root = data_root / "Cars_Body_Type"
    for split in ["train", "valid", "test"]:
        for source_class, target_class in CARS_BODY_TYPE_MAP.items():
            folder = source_root / split / source_class
            for source in image_files(folder):
                copy_image(source, raw_root, target_class, f"cars_body_type_{split}_{source_class}", counts)


def copy_f1_dataset(data_root: Path, raw_root: Path, counts: Counter[str]) -> None:
    source_root = data_root / "F1_new_upd_car_data"
    if not source_root.exists():
        return

    for team_folder in sorted(path for path in source_root.iterdir() if path.is_dir()):
        safe_team = re.sub(r"[^A-Za-z0-9]+", "_", team_folder.name).strip("_")
        for source in image_files(team_folder):
            copy_image(source, raw_root, "F1", f"f1_{safe_team}", counts)


def copy_stanford_folders(data_root: Path, raw_root: Path, counts: Counter[str]) -> None:
    source_root = data_root / "archive" / "stanford_cars_type"
    for source_class in STANFORD_FOLDER_MAP:
        for source in image_files(source_root / source_class):
            target_class = target_for_stanford_folder(source_class, source)
            copy_image(source, raw_root, target_class, f"stanford_{source_class}", counts)


def copy_stanford_micro_and_pickup_other(data_root: Path, raw_root: Path, counts: Counter[str]) -> None:
    csv_path = data_root / "archive" / "stanford_cars_type.csv"
    source_root = data_root / "archive" / "stanford_cars_type"
    if not csv_path.exists():
        return

    with csv_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            car_name = row.get("car_name", "")
            car_type = row.get("car_type", "")
            filename = row.get("new_filename", "")
            source = source_root / car_type / filename
            if not source.exists():
                continue

            if MICRO_PATTERN.search(car_name):
                copy_image(source, raw_root, "MICRO", f"stanford_micro_{car_type}", counts)
            elif car_type == "Other" and PICKUP_OTHER_PATTERN.search(car_name):
                copy_image(source, raw_root, "PICK_UP", "stanford_other_pickup", counts)


def copy_hf_micro_dataset(data_root: Path, raw_root: Path, counts: Counter[str]) -> None:
    source_root = data_root / "HF_MICRO_car_images"
    for source in image_files(source_root):
        copy_image(source, raw_root, "MICRO", "hf_micro", counts)


def copy_hf_station_wagon_dataset(data_root: Path, raw_root: Path, counts: Counter[str]) -> None:
    source_root = data_root / "HF_STATION_WAGON_car_images"
    for source in image_files(source_root):
        copy_image(source, raw_root, "STATION_WAGON", "hf_station_wagon", counts)


def copy_hf_body_dataset(data_root: Path, raw_root: Path, counts: Counter[str]) -> None:
    source_root = data_root / "HF_BODY_car_images"
    for class_name in PROJECT_CLASSES:
        for source in image_files(source_root / class_name):
            copy_image(source, raw_root, class_name, f"hf_body_{class_name.lower()}", counts)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    raw_root = data_root / "raw"
    counts: Counter[str] = Counter()

    ensure_raw_dirs(raw_root)
    if args.clear:
        guard_clear(raw_root, data_root, args.allow_partial_rebuild)
        clear_raw_dirs(raw_root)

    copy_cars_body_type(data_root, raw_root, counts)
    copy_f1_dataset(data_root, raw_root, counts)
    copy_stanford_folders(data_root, raw_root, counts)
    copy_stanford_micro_and_pickup_other(data_root, raw_root, counts)
    copy_hf_micro_dataset(data_root, raw_root, counts)
    copy_hf_station_wagon_dataset(data_root, raw_root, counts)
    copy_hf_body_dataset(data_root, raw_root, counts)

    print("Copied images into data/raw:")
    for class_name in PROJECT_CLASSES:
        actual_count = len(image_files(raw_root / class_name))
        print(f"  {class_name:14s} {actual_count:5d}")

    if len(image_files(raw_root / "F1")) == 0:
        print("\nWARNING: F1 is still empty. The two downloaded datasets do not contain Formula/F1 images.")
        print("Add a third F1/open-wheel dataset before training the final 8-class model.")

    if len(image_files(raw_root / "MICRO")) < 300:
        print("\nWARNING: MICRO has limited data. It was extracted by model-name keywords from Stanford.")
        print("For best results, add a dedicated micro-car dataset.")

    if len(image_files(raw_root / "STATION_WAGON")) < 500:
        print("\nWARNING: STATION_WAGON has limited data.")
        print("For best results, add more wagon/estate/avant/touring images and review them before training.")


if __name__ == "__main__":
    main()
