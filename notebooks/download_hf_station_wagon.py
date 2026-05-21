"""
Download extra STATION_WAGON-class images from DamianBoborzi/car_images.

The dataset is streamed, so it does not need to download the full dataset first.
By default this keeps only autoevolution images and skips meshfleet-generated
renders. Use --include-meshfleet if you intentionally want generated images too.

Downloaded files are source data, not automatically final-clean data. After this:
    python notebooks/prepare_raw_data.py --clear
    python notebooks/audit_raw_data.py --samples-per-class 100 --sheet-cols 10
    python notebooks/build_review_candidates.py
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "DamianBoborzi/car_images"

WAGON_PATTERN = re.compile(
    r"("
    r"station wagon|estate car|estate\b|wagon\b|avant\b|touring\b|"
    r"sport[\s_-]?tourer|sports[\s_-]?tourer|variant\b|allroad\b|"
    r"shooting[\s_-]?brake|cross[\s_-]?country|kombi\b|stationcar"
    r")",
    re.IGNORECASE,
)

NON_WAGON_PATTERN = re.compile(
    r"("
    r"van\b|minivan|cargo van|bus\b|suv\b|crossover|hatchback|liftback|"
    r"wagon[\s_-]?r|step[\s_-]?wagon|e[\s_-]?series|mpv\b|pickup|cab\b"
    r")",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream station-wagon images from Hugging Face.")
    parser.add_argument("--output-dir", default="data/HF_STATION_WAGON_car_images")
    parser.add_argument("--target-count", type=int, default=800)
    parser.add_argument("--split", default="train")
    parser.add_argument("--min-aesthetic-score", type=float, default=5.5)
    parser.add_argument(
        "--include-meshfleet",
        action="store_true",
        help="Include generated meshfleet images. Default keeps real autoevolution images only.",
    )
    return parser.parse_args()


def safe_stem(value: str) -> str:
    stem = Path(value).stem if value else "station_wagon"
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    return stem[:90] or "station_wagon"


def is_station_wagon(item: dict) -> bool:
    text = str(item.get("text", ""))
    filename = str(item.get("original_filename", ""))
    combined = f"{filename} {text}"
    if not WAGON_PATTERN.search(combined):
        return False
    return not NON_WAGON_PATTERN.search(combined)


def score_is_good_enough(item: dict, minimum: float) -> bool:
    value = item.get("aesthetic_score")
    if value in {None, ""}:
        return True
    try:
        return float(value) >= minimum
    except (TypeError, ValueError):
        return True


def write_metadata_header(path: Path) -> None:
    if path.exists():
        return
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"
    write_metadata_header(metadata_path)

    existing = len(list(output_dir.glob("hf_station_wagon_*.jpg")))
    saved = existing
    scanned = 0

    print(f"Streaming {DATASET_NAME}/{args.split}")
    dataset = load_dataset(DATASET_NAME, split=args.split, streaming=True)

    for item in dataset:
        scanned += 1

        if not args.include_meshfleet and str(item.get("source", "")).lower() == "meshfleet":
            continue
        if not score_is_good_enough(item, args.min_aesthetic_score):
            continue
        if not is_station_wagon(item):
            continue

        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        original_filename = str(item.get("original_filename", ""))
        output_name = f"hf_station_wagon_{saved:05d}_{safe_stem(original_filename)}.jpg"
        output_path = output_dir / output_name
        image.save(output_path, "JPEG", quality=95)
        append_metadata(
            metadata_path,
            {
                "saved_filename": output_name,
                "original_filename": original_filename,
                "source": str(item.get("source", "")),
                "aesthetic_score": str(item.get("aesthetic_score", "")),
                "text": str(item.get("text", "")),
            },
        )
        saved += 1

        if saved % 50 == 0:
            print(f"Saved {saved}/{args.target_count} STATION_WAGON images after scanning {scanned} rows.")

        if saved >= args.target_count:
            break

    print(f"Done. Saved {saved - existing} new images. Total hf_station_wagon files: {saved}.")
    if saved < args.target_count:
        print("Target was not reached. Try lowering --min-aesthetic-score or add another station-wagon source.")


if __name__ == "__main__":
    main()
