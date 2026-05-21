"""
Build a manual review queue from raw data audit reports.

This script does not modify data/raw. It creates:
    data/review/raw/review_candidates.csv
    data/review/raw/contact_sheets/*.jpg
    data/review/raw/README.md

Edit the CSV's `decision` column, then apply it with:
    python notebooks/apply_review_decisions.py
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


HIGH_RISK_CLASSES = {"F1", "MICRO", "STATION_WAGON"}
VALID_DECISIONS = {"REVIEW", "KEEP", "REMOVE", "MOVE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create review candidates from audit CSV files.")
    parser.add_argument("--audit-dir", default="data/audit/raw")
    parser.add_argument("--output-dir", default="data/review/raw")
    parser.add_argument("--sheet-cols", type=int, default=8)
    parser.add_argument("--thumb-width", type=int, default=180)
    parser.add_argument("--thumb-height", type=int, default=140)
    parser.add_argument("--max-images-per-sheet", type=int, default=240)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def add_candidate(
    candidates: dict[str, dict[str, str]],
    image: dict[str, str],
    reason: str,
    decision: str = "REVIEW",
    target_class: str = "",
    priority: str = "P2",
) -> None:
    path = image["path"]
    existing = candidates.get(path)
    if existing:
        existing["reason"] = f"{existing['reason']} | {reason}"
        if existing["decision"] == "REVIEW" and decision != "REVIEW":
            existing["decision"] = decision
            existing["target_class"] = target_class
        if priority < existing["priority"]:
            existing["priority"] = priority
        return

    candidates[path] = {
        "candidate_id": f"C{len(candidates) + 1:05d}",
        "priority": priority,
        "class_name": image["class_name"],
        "filename": image["filename"],
        "path": path,
        "width": image["width"],
        "height": image["height"],
        "reason": reason,
        "decision": decision,
        "target_class": target_class,
        "notes": "",
    }


def build_candidates(audit_dir: Path) -> list[dict[str, str]]:
    images = read_csv(audit_dir / "images.csv")
    exact_duplicates = read_csv(audit_dir / "exact_duplicates.csv")
    perceptual_duplicates = read_csv(audit_dir / "perceptual_hash_duplicates.csv")
    near_duplicates = read_csv(audit_dir / "near_duplicate_pairs.csv")

    by_path = {row["path"]: row for row in images}
    candidates: dict[str, dict[str, str]] = {}

    # Safe duplicate removal candidate: keep first path in each exact duplicate group.
    exact_by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in exact_duplicates:
        if row["path"] in by_path:
            exact_by_group[row["group_id"]].append(row)
    for group in exact_by_group.values():
        for duplicate in sorted(group, key=lambda item: item["path"])[1:]:
            image = by_path[duplicate["path"]]
            add_candidate(
                candidates,
                image,
                "exact duplicate file; keep first copy in group",
                decision="REMOVE",
                priority="P0",
            )

    # Perceptual duplicate candidates in high-risk classes only. These are not safe enough to auto-remove.
    perceptual_by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in perceptual_duplicates:
        image = by_path.get(row["path"])
        if image and image["class_name"] in HIGH_RISK_CLASSES:
            perceptual_by_group[row["group_id"]].append(row)
    for group in perceptual_by_group.values():
        for item in group:
            image = by_path[item["path"]]
            add_candidate(candidates, image, "identical perceptual hash group", priority="P2")

    # Near duplicate pairs in high-risk classes.
    for row in near_duplicates:
        if row.get("class_name") not in HIGH_RISK_CLASSES:
            continue
        for key in ["path_a", "path_b"]:
            image = by_path.get(row[key])
            if image:
                add_candidate(
                    candidates,
                    image,
                    f"near duplicate candidate, dHash distance={row['hamming_distance']}",
                    priority="P2",
                )

    for image in images:
        class_name = image["class_name"]
        filename = image["filename"]
        filename_lower = filename.lower()
        width = int(image["width"] or 0)
        height = int(image["height"] or 0)
        ratio = width / height if height else 1.0

        if class_name == "STATION_WAGON":
            if filename_lower.startswith("hf_station_wagon"):
                add_candidate(
                    candidates,
                    image,
                    "new HF station-wagon source; verify before training",
                    priority="P1",
                )
            elif "ford e-series wagon van" in filename_lower:
                add_candidate(
                    candidates,
                    image,
                    "Ford E-Series Wagon Van is visually and semantically a VAN, not station wagon",
                    decision="MOVE",
                    target_class="VAN",
                    priority="P0",
                )
            elif "dodge caliber wagon" in filename_lower:
                add_candidate(
                    candidates,
                    image,
                    "Dodge Caliber Wagon is visually close to hatchback/crossover; manual review",
                    priority="P1",
                )

        elif filename_lower.startswith("hf_body_"):
            priority = "P1" if class_name in HIGH_RISK_CLASSES else "P2"
            add_candidate(
                candidates,
                image,
                "new HF body-class source; verify before training",
                priority=priority,
            )

        if class_name == "MICRO":
            if re.search(r"convertible|roadster|geo metro|mini cooper", filename_lower):
                add_candidate(
                    candidates,
                    image,
                    "MICRO candidate is convertible/roadster/older small car; verify it should remain MICRO",
                    priority="P1",
                )

        if class_name == "F1":
            if width < 260 or height < 160:
                add_candidate(candidates, image, "low-resolution F1 image", priority="P1")
            if ratio > 3.8 or ratio < 0.32:
                add_candidate(candidates, image, "extreme aspect ratio; may be cropped/partial/screenshot", priority="P1")
            if re.search(r"(^|__)images? |(^|__)images?\\b|download|wallpaper|poster|model|toy|diecast|tech_tuesday|tito", filename_lower):
                add_candidate(candidates, image, "generic/suspicious F1 filename; verify full real car is visible", priority="P2")

    return sorted(candidates.values(), key=lambda row: (row["priority"], row["class_name"], row["filename"]))


def make_contact_sheet(
    rows: list[dict[str, str]],
    output_path: Path,
    sheet_cols: int,
    thumb_width: int,
    thumb_height: int,
) -> None:
    if not rows:
        return

    label_height = 58
    rows_count = (len(rows) + sheet_cols - 1) // sheet_cols
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
        draw.rectangle([x, y, x + thumb_width - 1, y + thumb_height + label_height - 1], outline=(210, 210, 210))
        label = f"{row['candidate_id']} {row['decision']} {row['target_class']}".strip()
        draw.text((x + 4, y + 4), label[:32], fill=(0, 0, 0), font=font)
        draw.text((x + 4, y + 20), row["filename"][:34], fill=(0, 0, 0), font=font)
        draw.text((x + 4, y + 36), row["reason"][:34], fill=(120, 0, 0), font=font)

        try:
            with Image.open(row["path"]) as image:
                image = image.convert("RGB")
                image.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
                paste_x = x + (thumb_width - image.width) // 2
                paste_y = y + label_height + (thumb_height - image.height) // 2
                sheet.paste(image, (paste_x, paste_y))
        except (OSError, UnidentifiedImageError, ValueError):
            draw.text((x + 4, y + label_height + 4), "unreadable", fill=(160, 0, 0), font=font)

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

    if len(rows) <= max_images_per_sheet:
        output_path = output_base.with_suffix(".jpg")
        make_contact_sheet(rows, output_path, sheet_cols, thumb_width, thumb_height)
        return [output_path]

    output_paths: list[Path] = []
    for page_index, start in enumerate(range(0, len(rows), max_images_per_sheet), start=1):
        page_rows = rows[start : start + max_images_per_sheet]
        output_path = output_base.with_name(f"{output_base.name}_page_{page_index:02d}").with_suffix(".jpg")
        make_contact_sheet(page_rows, output_path, sheet_cols, thumb_width, thumb_height)
        output_paths.append(output_path)
    return output_paths


def write_readme(output_dir: Path, candidates: list[dict[str, str]], contact_sheets: list[Path]) -> None:
    counts = Counter(row["class_name"] for row in candidates)
    decisions = Counter(row["decision"] for row in candidates)
    lines = [
        "# Raw Data Review Queue",
        "",
        "Edit `review_candidates.csv` and set `decision` to one of:",
        "",
        "- `KEEP`: verified good image",
        "- `REMOVE`: move image to quarantine",
        "- `MOVE`: move image to `target_class`",
        "- `REVIEW`: not decided yet",
        "",
        "Apply decisions with:",
        "",
        "```powershell",
        "python notebooks\\apply_review_decisions.py",
        "```",
        "",
        "## Candidate Counts",
        "",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Decision Defaults", ""])
    for key, value in sorted(decisions.items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Clean Data Rule",
            "",
            "`data/raw` is the training dataset after all `REVIEW` rows are resolved and `REMOVE`/`MOVE` decisions are applied.",
            "Rejected files are moved to `data/quarantine/raw` and are not used for training.",
            "",
            "## Contact Sheets",
            "",
        ]
    )
    for sheet_path in contact_sheets:
        lines.append(f"- `{sheet_path.relative_to(output_dir)}`")
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    audit_dir = Path(args.audit_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(output_dir / "contact_sheets", ignore_errors=True)

    candidates = build_candidates(audit_dir)
    preserve_previous_decisions(candidates, output_dir / "review_candidates.csv")
    fieldnames = [
        "candidate_id",
        "priority",
        "class_name",
        "filename",
        "path",
        "width",
        "height",
        "reason",
        "decision",
        "target_class",
        "notes",
    ]
    write_csv(output_dir / "review_candidates.csv", candidates, fieldnames)

    contact_sheets: list[Path] = []
    for class_name in sorted({row["class_name"] for row in candidates}):
        rows = [row for row in candidates if row["class_name"] == class_name]
        contact_sheets.extend(
            make_paged_contact_sheets(
                rows,
                output_dir / "contact_sheets" / "by_class" / class_name,
                args.sheet_cols,
                args.thumb_width,
                args.thumb_height,
                args.max_images_per_sheet,
            )
        )

    for decision in sorted(VALID_DECISIONS):
        rows = [row for row in candidates if row["decision"] == decision]
        if not rows:
            continue
        contact_sheets.extend(
            make_paged_contact_sheets(
                rows,
                output_dir / "contact_sheets" / "by_decision" / decision,
                args.sheet_cols,
                args.thumb_width,
                args.thumb_height,
                args.max_images_per_sheet,
            )
        )

    write_readme(output_dir, candidates, contact_sheets)

    print(f"Review candidates written to: {output_dir / 'review_candidates.csv'}")
    print("By class:")
    for class_name, count in sorted(Counter(row["class_name"] for row in candidates).items()):
        print(f"  {class_name:14s} {count:5d}")
    print("By default decision:")
    for decision, count in sorted(Counter(row["decision"] for row in candidates).items()):
        print(f"  {decision:14s} {count:5d}")


if __name__ == "__main__":
    main()
