"""
Audit AutoVision raw image data.

Checks:
    - per-class image counts, file sizes, dimensions, extensions
    - corrupt/unreadable images
    - exact duplicate files via SHA-256
    - perceptual duplicate candidates via dHash
    - random contact sheets per class

Outputs are written to data/audit/raw by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


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


@dataclass
class ImageRecord:
    class_name: str
    path: str
    filename: str
    suffix: str
    size_bytes: int
    width: int | None
    height: int | None
    mode: str | None
    sha256: str | None
    dhash: str | None
    corrupt_error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit data/raw image folders.")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--output-dir", default="data/audit/raw")
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--sheet-cols", type=int, default=10)
    parser.add_argument("--thumb-width", type=int, default=160)
    parser.add_argument("--thumb-height", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--near-duplicate-threshold",
        type=int,
        default=4,
        help="Hamming distance threshold for dHash duplicate candidates.",
    )
    parser.add_argument(
        "--max-near-duplicate-pairs-per-class",
        type=int,
        default=1000,
        help="Limit report size for near duplicate candidates.",
    )
    return parser.parse_args()


def image_paths(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [
        path
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash_image(image: Image.Image, hash_size: int = 8) -> int:
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    bit = 0
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[row_start + col]
            right = pixels[row_start + col + 1]
            if left > right:
                value |= 1 << bit
            bit += 1
    return value


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def audit_image(path: Path, class_name: str) -> ImageRecord:
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
            dhash = f"{dhash_image(image):016x}"
        sha256 = sha256_file(path)
        return ImageRecord(
            class_name=class_name,
            path=str(path),
            filename=path.name,
            suffix=path.suffix.lower(),
            size_bytes=path.stat().st_size,
            width=width,
            height=height,
            mode=mode,
            sha256=sha256,
            dhash=dhash,
            corrupt_error=None,
        )
    except (OSError, UnidentifiedImageError, ValueError) as error:
        return ImageRecord(
            class_name=class_name,
            path=str(path),
            filename=path.name,
            suffix=path.suffix.lower(),
            size_bytes=path.stat().st_size if path.exists() else 0,
            width=None,
            height=None,
            mode=None,
            sha256=None,
            dhash=None,
            corrupt_error=str(error),
        )


def summarize_class(records: list[ImageRecord]) -> dict:
    good = [record for record in records if record.corrupt_error is None]
    sizes = [record.size_bytes for record in good]
    widths = [record.width for record in good if record.width is not None]
    heights = [record.height for record in good if record.height is not None]
    suffixes = Counter(record.suffix for record in records)
    modes = Counter(record.mode for record in good)

    def stats(values: list[int]) -> dict[str, float | int | None]:
        if not values:
            return {"min": None, "median": None, "mean": None, "max": None}
        return {
            "min": min(values),
            "median": median(values),
            "mean": round(mean(values), 2),
            "max": max(values),
        }

    return {
        "count": len(records),
        "valid_count": len(good),
        "corrupt_count": len(records) - len(good),
        "total_mb": round(sum(record.size_bytes for record in records) / (1024 * 1024), 2),
        "file_size_kb": {key: (round(value / 1024, 2) if value is not None else None) for key, value in stats(sizes).items()},
        "width": stats(widths),
        "height": stats(heights),
        "suffixes": dict(sorted(suffixes.items())),
        "modes": dict(sorted(modes.items())),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def duplicate_groups(records: list[ImageRecord], key: str) -> list[list[ImageRecord]]:
    buckets: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        value = getattr(record, key)
        if value:
            buckets[value].append(record)
    return [group for group in buckets.values() if len(group) > 1]


def near_duplicate_pairs(
    records: list[ImageRecord],
    threshold: int,
    max_pairs_per_class: int,
) -> list[dict]:
    pairs: list[dict] = []
    by_class: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        if record.dhash and record.corrupt_error is None:
            by_class[record.class_name].append(record)

    for class_name, class_records in by_class.items():
        hashes = [(int(record.dhash, 16), record) for record in class_records if record.dhash]
        class_pairs = 0
        for index, (hash_a, record_a) in enumerate(hashes):
            for hash_b, record_b in hashes[index + 1 :]:
                distance = hamming_distance(hash_a, hash_b)
                if distance <= threshold:
                    pairs.append(
                        {
                            "class_name": class_name,
                            "hamming_distance": distance,
                            "path_a": record_a.path,
                            "path_b": record_b.path,
                        }
                    )
                    class_pairs += 1
                    if class_pairs >= max_pairs_per_class:
                        break
            if class_pairs >= max_pairs_per_class:
                break
    return pairs


def make_contact_sheet(
    records: list[ImageRecord],
    output_path: Path,
    samples_per_class: int,
    sheet_cols: int,
    thumb_width: int,
    thumb_height: int,
    seed: int,
) -> None:
    good = [record for record in records if record.corrupt_error is None]
    rng = random.Random(seed)
    if len(good) > samples_per_class:
        good = rng.sample(good, samples_per_class)

    label_height = 32
    rows = max(1, (len(good) + sheet_cols - 1) // sheet_cols)
    sheet = Image.new("RGB", (sheet_cols * thumb_width, rows * (thumb_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        font = ImageFont.load_default()

    for index, record in enumerate(good):
        col = index % sheet_cols
        row = index // sheet_cols
        x = col * thumb_width
        y = row * (thumb_height + label_height)
        draw.rectangle([x, y, x + thumb_width - 1, y + thumb_height + label_height - 1], outline=(215, 215, 215))
        draw.text((x + 4, y + 4), record.filename[:28], fill=(0, 0, 0), font=font)

        try:
            with Image.open(record.path) as image:
                image = image.convert("RGB")
                image.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
                paste_x = x + (thumb_width - image.width) // 2
                paste_y = y + label_height + (thumb_height - image.height) // 2
                sheet.paste(image, (paste_x, paste_y))
        except (OSError, UnidentifiedImageError, ValueError):
            draw.text((x + 4, y + label_height + 4), "unreadable", fill=(160, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def make_markdown_report(
    output_path: Path,
    summary: dict[str, dict],
    corrupt: list[ImageRecord],
    exact_groups: list[list[ImageRecord]],
    dhash_groups: list[list[ImageRecord]],
    near_pairs: list[dict],
) -> None:
    lines = ["# Raw Data Audit", ""]
    lines.append("## Class Summary")
    lines.append("")
    lines.append("| Class | Count | Corrupt | Total MB | Width Median | Height Median | Extensions |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for class_name in PROJECT_CLASSES:
        data = summary.get(class_name, {})
        suffixes = ", ".join(f"{key}:{value}" for key, value in data.get("suffixes", {}).items())
        lines.append(
            f"| {class_name} | {data.get('count', 0)} | {data.get('corrupt_count', 0)} | "
            f"{data.get('total_mb', 0)} | {data.get('width', {}).get('median')} | "
            f"{data.get('height', {}).get('median')} | {suffixes} |"
        )

    lines.extend(
        [
            "",
            "## Issues",
            "",
            f"- Corrupt/unreadable images: {len(corrupt)}",
            f"- Exact duplicate SHA-256 groups: {len(exact_groups)}",
            f"- Identical dHash perceptual groups: {len(dhash_groups)}",
            f"- Near-duplicate dHash pairs reported: {len(near_pairs)}",
            "",
            "## Contact Sheets",
            "",
        ]
    )
    for class_name in PROJECT_CLASSES:
        lines.append(f"- `{class_name}`: `contact_sheets/{class_name}.jpg`")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[ImageRecord] = []
    for class_name in PROJECT_CLASSES:
        paths = image_paths(raw_root / class_name)
        print(f"Auditing {class_name}: {len(paths)} files")
        for path in paths:
            records.append(audit_image(path, class_name))

    record_rows = [asdict(record) for record in records]
    write_csv(
        output_dir / "images.csv",
        record_rows,
        [
            "class_name",
            "path",
            "filename",
            "suffix",
            "size_bytes",
            "width",
            "height",
            "mode",
            "sha256",
            "dhash",
            "corrupt_error",
        ],
    )

    summary = {
        class_name: summarize_class([record for record in records if record.class_name == class_name])
        for class_name in PROJECT_CLASSES
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    corrupt = [record for record in records if record.corrupt_error is not None]
    write_csv(output_dir / "corrupt_images.csv", [asdict(record) for record in corrupt], list(asdict(records[0]).keys()) if records else [])

    exact_groups = duplicate_groups(records, "sha256")
    exact_rows = []
    for group_id, group in enumerate(exact_groups, start=1):
        for record in group:
            exact_rows.append({"group_id": group_id, "class_name": record.class_name, "path": record.path, "sha256": record.sha256})
    write_csv(output_dir / "exact_duplicates.csv", exact_rows, ["group_id", "class_name", "path", "sha256"])

    dhash_groups = duplicate_groups(records, "dhash")
    dhash_rows = []
    for group_id, group in enumerate(dhash_groups, start=1):
        for record in group:
            dhash_rows.append({"group_id": group_id, "class_name": record.class_name, "path": record.path, "dhash": record.dhash})
    write_csv(output_dir / "perceptual_hash_duplicates.csv", dhash_rows, ["group_id", "class_name", "path", "dhash"])

    near_pairs = near_duplicate_pairs(records, args.near_duplicate_threshold, args.max_near_duplicate_pairs_per_class)
    write_csv(output_dir / "near_duplicate_pairs.csv", near_pairs, ["class_name", "hamming_distance", "path_a", "path_b"])

    for class_name in PROJECT_CLASSES:
        class_records = [record for record in records if record.class_name == class_name]
        make_contact_sheet(
            class_records,
            output_dir / "contact_sheets" / f"{class_name}.jpg",
            args.samples_per_class,
            args.sheet_cols,
            args.thumb_width,
            args.thumb_height,
            args.seed,
        )

    make_markdown_report(output_dir / "report.md", summary, corrupt, exact_groups, dhash_groups, near_pairs)

    print(f"\nAudit written to: {output_dir}")
    print("Counts:")
    for class_name in PROJECT_CLASSES:
        data = summary[class_name]
        print(f"  {class_name:14s} {data['count']:5d} images, {data['corrupt_count']} corrupt")
    print(f"Exact duplicate groups: {len(exact_groups)}")
    print(f"Perceptual duplicate groups: {len(dhash_groups)}")
    print(f"Near-duplicate pairs reported: {len(near_pairs)}")


if __name__ == "__main__":
    main()
