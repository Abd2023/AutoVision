"""
Download car body-type source images from DamianBoborzi/car_images.

This is a source-data downloader. Images are saved under:
    data/HF_BODY_car_images/<CLASS>/

Then rebuild/audit/review with:
    python notebooks/prepare_raw_data.py --clear
    python notebooks/run_data_quality_loop.py
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "DamianBoborzi/car_images"
DEFAULT_CLASSES = ["HATCHBACK", "MICRO", "PICK_UP", "SEDAN", "SUV", "VAN"]

PATTERNS = {
    "HATCHBACK": re.compile(r"\b(hatchback|hot hatch|3-door hatch|5-door hatch)\b", re.IGNORECASE),
    "MICRO": re.compile(
        r"(smart fortwo|smart forfour|fiat 500|fiat panda|mini cooper|geo metro|"
        r"toyota iq|toyota aygo|chevrolet spark|hyundai i10|kia picanto|renault twingo|"
        r"citroen c1|peugeot 108|volkswagen up|vw up|seat mii|skoda citigo)",
        re.IGNORECASE,
    ),
    "PICK_UP": re.compile(
        r"\b(pickup|pick-up|pickup truck|crew cab|regular cab|extended cab|double cab|supercab|"
        r"super crew|ute\b|sute\b|truck bed)\b",
        re.IGNORECASE,
    ),
    "SEDAN": re.compile(r"\b(sedan|saloon)\b", re.IGNORECASE),
    "SUV": re.compile(r"\b(suv|sport utility|crossover suv|off-road suv|4x4 suv)\b", re.IGNORECASE),
    "VAN": re.compile(r"\b(van|minivan|mini-van|cargo van|passenger van|mpv|people carrier)\b", re.IGNORECASE),
}

NEGATIVE_PATTERNS = {
    "HATCHBACK": re.compile(r"\b(sedan|saloon|suv|van|minivan|wagon|estate|pickup|pick-up|truck)\b", re.IGNORECASE),
    "MICRO": re.compile(r"\b(sedan|suv|van|minivan|wagon|estate|pickup|pick-up|truck)\b", re.IGNORECASE),
    "PICK_UP": re.compile(r"\b(sedan|suv|van|minivan|wagon|estate|hatchback|semi truck|bus)\b", re.IGNORECASE),
    "SEDAN": re.compile(r"\b(coupe|convertible|roadster|suv|van|minivan|wagon|estate|hatchback|pickup|pick-up|truck)\b", re.IGNORECASE),
    "SUV": re.compile(r"\b(sedan|saloon|van|minivan|wagon|estate|hatchback|pickup|pick-up|truck)\b", re.IGNORECASE),
    "VAN": re.compile(r"\b(sedan|saloon|suv|wagon|estate|hatchback|pickup|pick-up|truck)\b", re.IGNORECASE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream car body-type images from Hugging Face.")
    parser.add_argument("--output-dir", default="data/HF_BODY_car_images")
    parser.add_argument("--target-per-class", type=int, default=1000)
    parser.add_argument("--split", default="train")
    parser.add_argument("--min-aesthetic-score", type=float, default=5.5)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, choices=sorted(PATTERNS))
    parser.add_argument(
        "--include-meshfleet",
        action="store_true",
        help="Include generated meshfleet images. Default keeps real autoevolution images only.",
    )
    return parser.parse_args()


def safe_stem(value: str) -> str:
    stem = Path(value).stem if value else "car"
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    return stem[:90] or "car"


def score_is_good_enough(item: dict, minimum: float) -> bool:
    value = item.get("aesthetic_score")
    if value in {None, ""}:
        return True
    try:
        return float(value) >= minimum
    except (TypeError, ValueError):
        return True


def classify(item: dict, classes: list[str]) -> str | None:
    text = str(item.get("text", ""))
    filename = str(item.get("original_filename", ""))
    combined = f"{filename} {text}"

    for class_name in classes:
        if not PATTERNS[class_name].search(combined):
            continue
        if NEGATIVE_PATTERNS[class_name].search(combined):
            continue
        return class_name
    return None


def metadata_path(output_dir: Path, class_name: str) -> Path:
    return output_dir / class_name / "metadata.csv"


def write_metadata_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["saved_filename", "original_filename", "source", "aesthetic_score", "text"],
        )
        writer.writeheader()


def append_metadata(path: Path, row: dict[str, str]) -> None:
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["saved_filename", "original_filename", "source", "aesthetic_score", "text"],
        )
        writer.writerow(row)


def read_seen_original_filenames(output_dir: Path, classes: list[str]) -> dict[str, set[str]]:
    seen: dict[str, set[str]] = {class_name: set() for class_name in classes}
    for class_name in classes:
        path = metadata_path(output_dir, class_name)
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                original_filename = row.get("original_filename", "").strip()
                if original_filename:
                    seen[class_name].add(original_filename)
    return seen


def existing_counts(output_dir: Path, classes: list[str]) -> dict[str, int]:
    return {
        class_name: len(list((output_dir / class_name).glob(f"hf_body_{class_name.lower()}_*.jpg")))
        for class_name in classes
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    classes = args.classes

    counts = existing_counts(output_dir, classes)
    seen_originals = read_seen_original_filenames(output_dir, classes)
    for class_name in classes:
        write_metadata_header(metadata_path(output_dir, class_name))

    print(f"Streaming {DATASET_NAME}/{args.split}")
    print("Targets:")
    for class_name in classes:
        print(f"  {class_name:14s} {counts[class_name]:5d}/{args.target_per_class}")

    scanned = 0
    dataset = load_dataset(DATASET_NAME, split=args.split, streaming=True)

    for item in dataset:
        scanned += 1

        if not args.include_meshfleet and str(item.get("source", "")).lower() == "meshfleet":
            continue
        if not score_is_good_enough(item, args.min_aesthetic_score):
            continue

        class_name = classify(item, classes)
        if class_name is None or counts[class_name] >= args.target_per_class:
            continue

        original_filename = str(item.get("original_filename", ""))
        if original_filename and original_filename in seen_originals[class_name]:
            continue

        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        output_name = f"hf_body_{class_name.lower()}_{counts[class_name]:05d}_{safe_stem(original_filename)}.jpg"
        output_path = output_dir / class_name / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, "JPEG", quality=95)
        append_metadata(
            metadata_path(output_dir, class_name),
            {
                "saved_filename": output_name,
                "original_filename": original_filename,
                "source": str(item.get("source", "")),
                "aesthetic_score": str(item.get("aesthetic_score", "")),
                "text": str(item.get("text", "")),
            },
        )
        if original_filename:
            seen_originals[class_name].add(original_filename)
        counts[class_name] += 1

        if sum(counts.values()) % 100 == 0:
            print(f"Saved {sum(counts.values())} total images after scanning {scanned} rows.")
            for key in classes:
                print(f"  {key:14s} {counts[key]:5d}/{args.target_per_class}")

        if all(counts[key] >= args.target_per_class for key in classes):
            break

    print("Done.")
    for class_name in classes:
        print(f"  {class_name:14s} {counts[class_name]:5d}/{args.target_per_class}")
    unfinished = [class_name for class_name in classes if counts[class_name] < args.target_per_class]
    if unfinished:
        print("Target not reached for: " + ", ".join(unfinished))
        print("Try lowering --min-aesthetic-score or add another source.")


if __name__ == "__main__":
    main()
