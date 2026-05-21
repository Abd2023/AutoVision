"""
Download extra MICRO-class images from DamianBoborzi/car_images.

The dataset is streamed, so it does not need to download the full dataset first.
By default this keeps only autoevolution images and skips meshfleet-generated
renders. Use --include-meshfleet if you intentionally want generated images too.

Example:
    python notebooks/download_hf_micro.py --target-count 800
    python notebooks/prepare_raw_data.py --clear
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "DamianBoborzi/car_images"

MICRO_PATTERN = re.compile(
    r"(smart fortwo|smart forfour|fiat 500|fiat panda|mini cooper|geo metro|"
    r"toyota iq|toyota aygo|chevrolet spark|hyundai i10|kia picanto|renault twingo|"
    r"citroen c1|peugeot 108|volkswagen up|vw up|seat mii|skoda citigo)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream micro-car images from Hugging Face.")
    parser.add_argument("--output-dir", default="data/HF_MICRO_car_images")
    parser.add_argument("--target-count", type=int, default=800)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--include-meshfleet",
        action="store_true",
        help="Include generated meshfleet images. Default keeps real autoevolution images only.",
    )
    return parser.parse_args()


def is_micro(item: dict) -> bool:
    text = str(item.get("text", ""))
    filename = str(item.get("original_filename", ""))
    return bool(MICRO_PATTERN.search(text) or MICRO_PATTERN.search(filename))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(output_dir.glob("hf_micro_*.jpg")))
    saved = existing
    scanned = 0

    print(f"Streaming {DATASET_NAME}/{args.split}")
    dataset = load_dataset(DATASET_NAME, split=args.split, streaming=True)

    for item in dataset:
        scanned += 1

        if not args.include_meshfleet and str(item.get("source", "")).lower() == "meshfleet":
            continue
        if not is_micro(item):
            continue

        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        output_path = output_dir / f"hf_micro_{saved:05d}.jpg"
        image.save(output_path, "JPEG", quality=95)
        saved += 1

        if saved % 50 == 0:
            print(f"Saved {saved}/{args.target_count} MICRO images after scanning {scanned} rows.")

        if saved >= args.target_count:
            break

    print(f"Done. Saved {saved - existing} new images. Total hf_micro files: {saved}.")
    if saved < args.target_count:
        print("Target was not reached. Try --include-meshfleet or add another MICRO source.")


if __name__ == "__main__":
    main()
