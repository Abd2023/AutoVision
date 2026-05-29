from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DIR_NAME_TO_LABEL = {
    "SUV": 1,
    "VAN": 2,
    "STATION WAGON": 3,
    "MICRO": 4,
    "F1 CAR": 5,
    "SEDAN": 6,
    "HATCHBACK": 7,
    "PICKUP": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate AutoVision prediction package against a labeled testdata folder.")
    parser.add_argument("--test-root", required=True, help="Path to labeled testdata folder.")
    parser.add_argument("--preds", required=True, help="Path to generated preds.txt or Preds.txt.")
    parser.add_argument("--true-out", default="True.txt", help="Where to write the generated ground-truth file.")
    return parser.parse_args()


def image_files(folder: Path) -> list[Path]:
    return sorted(
        [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES],
        key=lambda path: path.name.lower(),
    )


def write_true_txt(test_root: Path, output_path: Path) -> None:
    lines: list[str] = []
    for class_dir in sorted([path for path in test_root.iterdir() if path.is_dir()], key=lambda path: path.name):
        if class_dir.name not in DIR_NAME_TO_LABEL:
            continue
        label = DIR_NAME_TO_LABEL[class_dir.name]
        for image_path in image_files(class_dir):
            lines.append(f"{image_path.name} | Pred: {label}")
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_predictions(file_path: Path) -> dict[str, int]:
    data: dict[str, int] = {}
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(" | ")
            filename = parts[0]
            value = int(parts[1].split(":")[1].strip())
            data[filename] = value
    return data


def main() -> None:
    args = parse_args()
    test_root = Path(args.test_root)
    preds_path = Path(args.preds)
    true_path = Path(args.true_out)

    write_true_txt(test_root, true_path)

    pred_dict = read_predictions(preds_path)
    true_dict = read_predictions(true_path)
    class_labels = [1, 2, 3, 4, 5, 6, 7, 8]

    common_files = sorted(set(pred_dict.keys()) & set(true_dict.keys()))
    y_true = [true_dict[file_name] for file_name in common_files]
    y_pred = [pred_dict[file_name] for file_name in common_files]

    print(f"Matched files: {len(common_files)}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=class_labels))
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=class_labels,
            target_names=[str(label) for label in class_labels],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
