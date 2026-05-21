"""
Run the repeatable raw-data quality loop.

Default behavior is read-only:
    python notebooks/run_data_quality_loop.py

Optional mutations:
    python notebooks/run_data_quality_loop.py --apply-decisions
    python notebooks/run_data_quality_loop.py --prepare-clear
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


REVIEW_CSV = Path("data/review/raw/review_candidates.csv")
SUMMARY_JSON = Path("data/audit/raw/summary.json")
EXACT_DUPLICATES_CSV = Path("data/audit/raw/exact_duplicates.csv")
PERCEPTUAL_DUPLICATES_CSV = Path("data/audit/raw/perceptual_hash_duplicates.csv")
NEAR_DUPLICATES_CSV = Path("data/audit/raw/near_duplicate_pairs.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AutoVision data audit/review loop.")
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--sheet-cols", type=int, default=10)
    parser.add_argument("--prepare-clear", action="store_true", help="Rebuild data/raw from source folders first.")
    parser.add_argument("--apply-decisions", action="store_true", help="Apply KEEP/REMOVE/MOVE decisions before audit.")
    parser.add_argument("--dry-run", action="store_true", help="Use with --apply-decisions to preview moves.")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print(f"\n> {' '.join(command)}")
    subprocess.run(command, check=True)


def read_review_counts() -> Counter[str]:
    if not REVIEW_CSV.exists():
        return Counter()
    with REVIEW_CSV.open(newline="", encoding="utf-8") as file:
        return Counter(row.get("decision", "").strip().upper() or "EMPTY" for row in csv.DictReader(file))


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as file:
        return sum(1 for _ in csv.DictReader(file))


def count_csv_groups(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as file:
        return len({row.get("group_id", "") for row in csv.DictReader(file) if row.get("group_id", "")})


def print_dataset_summary() -> None:
    if not SUMMARY_JSON.exists():
        return
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    print("\nCurrent data/raw counts:")
    for class_name, item in sorted(summary.items()):
        print(f"  {class_name:14s} {item['count']:5d} images, {item['corrupt_count']:d} corrupt")
    print("\nAudit issues:")
    print(f"  exact_duplicate_groups: {count_csv_groups(EXACT_DUPLICATES_CSV)}")
    print(f"  perceptual_duplicate_groups: {count_csv_groups(PERCEPTUAL_DUPLICATES_CSV)}")
    print(f"  near_duplicate_pairs_reported: {count_csv_rows(NEAR_DUPLICATES_CSV)}")


def print_review_summary() -> None:
    decisions = read_review_counts()
    total = sum(decisions.values())
    print("\nReview queue:")
    print(f"  total candidates: {total}")
    for decision, count in sorted(decisions.items()):
        print(f"  {decision:14s} {count:5d}")
    if decisions.get("REVIEW", 0):
        print("\nNext action: edit data/review/raw/review_candidates.csv and change REVIEW rows to KEEP, REMOVE, or MOVE.")
    elif total:
        print("\nNext action: run with --apply-decisions, then run this loop again.")
    else:
        print("\nNext action: data/raw has no generated review candidates from the current audit.")


def main() -> None:
    args = parse_args()

    if args.prepare_clear:
        run([sys.executable, "notebooks/prepare_raw_data.py", "--clear"])

    if args.apply_decisions:
        command = [sys.executable, "notebooks/apply_review_decisions.py"]
        if args.dry_run:
            command.append("--dry-run")
        run(command)

    run(
        [
            sys.executable,
            "notebooks/audit_raw_data.py",
            "--samples-per-class",
            str(args.samples_per_class),
            "--sheet-cols",
            str(args.sheet_cols),
        ]
    )
    run([sys.executable, "notebooks/build_review_candidates.py"])

    print_dataset_summary()
    print_review_summary()


if __name__ == "__main__":
    main()
