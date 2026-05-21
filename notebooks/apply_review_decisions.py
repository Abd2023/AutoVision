"""
Apply reviewed data-cleaning decisions.

This script reads data/review/raw/review_candidates.csv and moves files instead
of deleting them:
    - REMOVE -> data/quarantine/raw/<class>/
    - MOVE   -> data/raw/<target_class>/

Rows with REVIEW or KEEP are ignored. Use --dry-run first.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


PROJECT_CLASSES = {
    "F1",
    "HATCHBACK",
    "MICRO",
    "PICK_UP",
    "SEDAN",
    "STATION_WAGON",
    "SUV",
    "VAN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply AutoVision raw data review decisions.")
    parser.add_argument("--review-csv", default="data/review/raw/review_candidates.csv")
    parser.add_argument("--quarantine-root", default="data/quarantine/raw")
    parser.add_argument("--dry-run", action="store_true", help="Print operations without moving files.")
    parser.add_argument("--quiet", action="store_true", help="Do not print every file move.")
    parser.add_argument(
        "--apply-review",
        action="store_true",
        help="Allow applying REVIEW rows. By default REVIEW rows are always ignored.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


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


def move_file(source: Path, destination: Path, dry_run: bool, quiet: bool) -> None:
    if not quiet:
        print(f"MOVE {source} -> {destination}")
    if dry_run:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.review_csv))
    quarantine_root = Path(args.quarantine_root)

    applied = 0
    skipped = 0
    missing = 0

    for row in rows:
        decision = row.get("decision", "").strip().upper()
        source = Path(row.get("path", ""))
        class_name = row.get("class_name", "").strip()
        target_class = row.get("target_class", "").strip()

        if decision in {"", "KEEP"}:
            skipped += 1
            continue
        if decision == "REVIEW" and not args.apply_review:
            skipped += 1
            continue
        if not source.exists():
            print(f"SKIP missing: {source}")
            missing += 1
            continue

        if decision == "REMOVE":
            destination = unique_destination(quarantine_root / class_name, source.name)
            move_file(source, destination, args.dry_run, args.quiet)
            applied += 1
        elif decision == "MOVE":
            if target_class not in PROJECT_CLASSES:
                raise ValueError(f"Invalid target_class for {source}: {target_class!r}")
            destination = unique_destination(Path("data/raw") / target_class, source.name)
            move_file(source, destination, args.dry_run, args.quiet)
            applied += 1
        else:
            raise ValueError(f"Invalid decision {decision!r} for {source}")

    print("\nSummary")
    print(f"  applied: {applied}")
    print(f"  skipped: {skipped}")
    print(f"  missing: {missing}")
    print(f"  dry_run: {args.dry_run}")


if __name__ == "__main__":
    main()
