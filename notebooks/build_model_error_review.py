"""
Build a manual review queue from model error-analysis outputs.

This script does not modify data. It creates:
    data/review/model_errors/<run_name>/review_candidates.csv
    data/review/model_errors/<run_name>/contact_sheets/*.jpg
    data/review/model_errors/<run_name>/README.md

Edit the CSV's `decision` column, then apply source-data decisions with:
    python notebooks/apply_review_decisions.py --review-csv <csv> --dataset-root data/ai_raw_generated
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOCUS_PAIRS = [
    "HATCHBACK:SEDAN",
    "SUV:PICK_UP",
    "SUV:STATION_WAGON",
    "PICK_UP:SUV",
    "SEDAN:STATION_WAGON",
    "STATION_WAGON:SEDAN",
]
VALID_DECISIONS = {"REVIEW", "KEEP", "REMOVE", "MOVE", "AMBIGUOUS"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a focused review queue from model errors.")
    parser.add_argument("--errors-dir", default="notebooks/outputs/error_analysis/resnet50_clean_round3")
    parser.add_argument("--manifest", default="data/processed/split_manifest.csv")
    parser.add_argument("--output-dir", default="data/review/model_errors/resnet50_clean_round3")
    parser.add_argument("--split", default="test")
    parser.add_argument("--p0-confidence", type=float, default=0.80)
    parser.add_argument("--focus-pairs", nargs="*", default=DEFAULT_FOCUS_PAIRS, help="Pairs formatted as ACTUAL:PREDICTED.")
    parser.add_argument("--sheet-cols", type=int, default=5)
    parser.add_argument("--thumb-width", type=int, default=180)
    parser.add_argument("--thumb-height", type=int, default=135)
    parser.add_argument("--max-images-per-sheet", type=int, default=160)
    parser.add_argument("--max-per-pair-sheet", type=int, default=80)
    return parser.parse_args()


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path.resolve()
    return project_path(path).resolve()


def parse_pairs(values: list[str]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for value in values:
        actual, predicted = [part.strip() for part in value.split(":", 1)]
        pairs.add((actual, predicted))
    return pairs


def priority_rank(priority: str) -> int:
    return {"P0": 0, "P1": 1, "P2": 2}.get(priority, 9)


def preserve_previous_decisions(candidates: list[dict[str, str]], previous_csv: Path) -> None:
    previous_rows = read_csv(previous_csv)
    previous_by_path = {row.get("path", ""): row for row in previous_rows}
    for candidate in candidates:
        previous = previous_by_path.get(candidate["path"])
        if not previous:
            continue
        previous_decision = previous.get("decision", "").strip().upper()
        if previous_decision in VALID_DECISIONS and previous_decision != "REVIEW":
            candidate["decision"] = previous_decision
            candidate["target_class"] = previous.get("target_class", "").strip()
        if previous.get("notes", "").strip():
            candidate["notes"] = previous["notes"]


def build_candidates(
    mistake_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    p0_confidence: float,
    focus_pairs: set[tuple[str, str]],
) -> list[dict[str, str]]:
    manifest_by_processed = {
        str(normalize_path(row["processed_path"])): row
        for row in manifest_rows
        if row.get("processed_path")
    }
    candidates: list[dict[str, str]] = []

    for index, mistake in enumerate(mistake_rows, start=1):
        processed_path = str(normalize_path(mistake["image_path"]))
        manifest = manifest_by_processed.get(processed_path, {})
        actual = mistake["actual_class"]
        predicted = mistake["predicted_class"]
        confidence = float(mistake["confidence"])
        pair = (actual, predicted)

        reasons: list[str] = []
        priority = "P2"
        if confidence >= p0_confidence:
            reasons.append("high-confidence wrong prediction")
            priority = "P0"
        if pair in focus_pairs:
            reasons.append(f"confusion-cluster pair {actual}->{predicted}")
            if priority != "P0":
                priority = "P1"
        if not reasons:
            reasons.append("remaining model error")

        candidates.append(
            {
                "candidate_id": f"M{index:05d}",
                "priority": priority,
                "split": manifest.get("split", ""),
                "class_name": actual,
                "predicted_class": predicted,
                "confidence": f"{confidence:.6f}",
                "top2_class": mistake.get("top2_class", ""),
                "top2_prob": mistake.get("top2_prob", ""),
                "top3_class": mistake.get("top3_class", ""),
                "top3_prob": mistake.get("top3_prob", ""),
                "path": manifest.get("source_path", processed_path),
                "processed_path": manifest.get("processed_path", processed_path),
                "source_sha256": manifest.get("source_sha256", ""),
                "duplicate_group_size": manifest.get("duplicate_group_size", ""),
                "reason": " | ".join(reasons),
                "decision": "REVIEW",
                "target_class": "",
                "notes": "",
            }
        )

    return sorted(
        candidates,
        key=lambda row: (
            priority_rank(row["priority"]),
            row["class_name"],
            -float(row["confidence"]),
            row["predicted_class"],
        ),
    )


def draw_wrapped_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, width_chars: int, font: ImageFont.ImageFont) -> None:
    x, y = xy
    for raw_line in text.splitlines():
        line = raw_line
        while len(line) > width_chars:
            draw.text((x, y), line[:width_chars], fill=(20, 20, 20), font=font)
            line = line[width_chars:]
            y += 12
        draw.text((x, y), line, fill=(20, 20, 20), font=font)
        y += 12


def make_contact_sheet(
    rows: list[dict[str, str]],
    output_path: Path,
    sheet_cols: int,
    thumb_width: int,
    thumb_height: int,
) -> None:
    if not rows:
        return

    label_height = 82
    rows_count = math.ceil(len(rows) / sheet_cols)
    sheet = Image.new("RGB", (sheet_cols * thumb_width, rows_count * (thumb_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        font = ImageFont.load_default()

    for index, row in enumerate(rows):
        col = index % sheet_cols
        grid_row = index // sheet_cols
        x = col * thumb_width
        y = grid_row * (thumb_height + label_height)
        image_path = Path(row["processed_path"])
        draw.rectangle([x, y, x + thumb_width - 1, y + thumb_height + label_height - 1], outline=(210, 210, 210))

        try:
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
                paste_x = x + (thumb_width - image.width) // 2
                paste_y = y + (thumb_height - image.height) // 2
                sheet.paste(image, (paste_x, paste_y))
        except (OSError, UnidentifiedImageError, ValueError):
            draw.text((x + 4, y + 4), "unreadable", fill=(180, 0, 0), font=font)

        label = (
            f"{row['candidate_id']} {row['priority']} {row['decision']}\n"
            f"{row['class_name']} -> {row['predicted_class']} conf={float(row['confidence']):.3f}\n"
            f"2:{row['top2_class']} {float(row['top2_prob'] or 0.0):.2f}  "
            f"3:{row['top3_class']} {float(row['top3_prob'] or 0.0):.2f}\n"
            f"{Path(row['path']).name}"
        )
        draw_wrapped_text(draw, (x + 4, y + thumb_height + 4), label, max(18, thumb_width // 7), font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def make_paged_contact_sheets(
    rows: list[dict[str, str]],
    output_base: Path,
    sheet_cols: int,
    thumb_width: int,
    thumb_height: int,
    max_images_per_sheet: int,
) -> list[Path]:
    if not rows:
        return []

    output_paths: list[Path] = []
    for page_index, start in enumerate(range(0, len(rows), max_images_per_sheet), start=1):
        page_rows = rows[start : start + max_images_per_sheet]
        output_path = output_base.with_name(f"{output_base.name}_page_{page_index:02d}").with_suffix(".jpg")
        make_contact_sheet(page_rows, output_path, sheet_cols, thumb_width, thumb_height)
        output_paths.append(output_path)
    return output_paths


def write_readme(output_dir: Path, split: str, candidates: list[dict[str, str]], contact_sheets: list[Path]) -> None:
    by_priority = Counter(row["priority"] for row in candidates)
    by_pair = Counter((row["class_name"], row["predicted_class"]) for row in candidates)
    lines = [
        "# Model Error Review Queue",
        "",
        f"Split reviewed: `{split}`",
        "",
        "Edit `review_candidates.csv` and set `decision` to one of:",
        "",
        "- `KEEP`: label is correct and image should remain in the dataset",
        "- `REMOVE`: image is junk, no-car, partial-car, or unusable",
        "- `MOVE`: image belongs to another class; fill `target_class`",
        "- `AMBIGUOUS`: image is too ambiguous to be a good training example",
        "- `REVIEW`: not decided yet",
        "",
        "Apply source-data decisions with:",
        "",
        "```powershell",
        f"python notebooks\\apply_review_decisions.py --review-csv {output_dir / 'review_candidates.csv'} --dataset-root data/ai_raw_generated",
        "```",
        "",
        "## Priority Counts",
        "",
    ]
    for key, value in sorted(by_priority.items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Top Error Pairs", ""])
    for (actual, predicted), count in by_pair.most_common(10):
        lines.append(f"- `{actual} -> {predicted}`: {count}")
    lines.extend(["", "## Contact Sheets", ""])
    for sheet_path in contact_sheets:
        lines.append(f"- `{sheet_path.relative_to(output_dir)}`")
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    errors_dir = project_path(args.errors_dir)
    manifest_path = project_path(args.manifest)
    output_dir = project_path(args.output_dir)
    split = args.split

    mistakes_path = errors_dir / f"mistakes_{split}.csv"
    if not mistakes_path.exists():
        raise FileNotFoundError(f"Missing mistakes file: {mistakes_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing split manifest: {manifest_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(output_dir / "contact_sheets", ignore_errors=True)

    candidates = build_candidates(
        mistake_rows=read_csv(mistakes_path),
        manifest_rows=read_csv(manifest_path),
        p0_confidence=args.p0_confidence,
        focus_pairs=parse_pairs(args.focus_pairs),
    )
    preserve_previous_decisions(candidates, output_dir / "review_candidates.csv")

    fieldnames = [
        "candidate_id",
        "priority",
        "split",
        "class_name",
        "predicted_class",
        "confidence",
        "top2_class",
        "top2_prob",
        "top3_class",
        "top3_prob",
        "path",
        "processed_path",
        "source_sha256",
        "duplicate_group_size",
        "reason",
        "decision",
        "target_class",
        "notes",
    ]
    write_csv(output_dir / "review_candidates.csv", candidates, fieldnames)

    contact_sheets: list[Path] = []
    for priority in ["P0", "P1", "P2"]:
        rows = [row for row in candidates if row["priority"] == priority]
        contact_sheets.extend(
            make_paged_contact_sheets(
                rows,
                output_dir / "contact_sheets" / "by_priority" / priority,
                args.sheet_cols,
                args.thumb_width,
                args.thumb_height,
                args.max_images_per_sheet,
            )
        )

    pair_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        pair_rows[(row["class_name"], row["predicted_class"])].append(row)
    for (actual, predicted), rows in sorted(pair_rows.items(), key=lambda item: (-len(item[1]), item[0][0], item[0][1])):
        contact_sheets.extend(
            make_paged_contact_sheets(
                rows[: args.max_per_pair_sheet],
                output_dir / "contact_sheets" / "by_pair" / f"{actual}_to_{predicted}",
                args.sheet_cols,
                args.thumb_width,
                args.thumb_height,
                args.max_images_per_sheet,
            )
        )

    write_readme(output_dir, split, candidates, contact_sheets)

    print(f"Review candidates written to: {output_dir / 'review_candidates.csv'}")
    print("By priority:")
    for priority, count in sorted(Counter(row["priority"] for row in candidates).items()):
        print(f"  {priority:4s} {count:5d}")
    print("Top pairs:")
    for (actual, predicted), count in Counter((row["class_name"], row["predicted_class"]) for row in candidates).most_common(8):
        print(f"  {actual:14s} -> {predicted:14s} {count:5d}")


if __name__ == "__main__":
    main()
