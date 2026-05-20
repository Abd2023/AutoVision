"""
Accuracy-first training pipeline for AutoVision.

This script is meant for rebuilding the model when accuracy matters more than
deployment size. It avoids the old random image split by using class-local
group splits, so raw/processed variants or duplicated base IDs do not leak
between train and validation.

Example:
    python notebooks/train_accuracy_first.py --dataset-mode raw_processed
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from timm.data import Mixup, resolve_model_data_config
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLASSES = [
    "F1",
    "HATCHBACK",
    "MICRO",
    "PICK_UP",
    "SEDAN",
    "STATION_WAGON",
    "SUV",
    "VAN",
]


@dataclass
class Sample:
    path: str
    label: int
    source: str
    group: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an accuracy-first car body classifier.")
    parser.add_argument("--data-root", default="data", help="Root folder containing raw/ and processed/.")
    parser.add_argument(
        "--dataset-mode",
        choices=["raw", "processed", "raw_processed"],
        default="raw_processed",
        help="Training data source. raw_processed gives the model both original and background-removed views.",
    )
    parser.add_argument(
        "--model",
        default="convnext_base.fb_in22k_ft_in1k",
        help="Any timm classification model. Strong alternatives: convnext_small.in12k_ft_in1k, efficientnetv2_rw_s.ra2_in1k, swin_base_patch4_window7_224.ms_in22k_ft_in1k.",
    )
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained ImageNet weights.")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument("--cutmix", type=float, default=0.8)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-class", type=int, default=0, help="Debug limit, 0 means no limit.")
    parser.add_argument("--output-dir", default="notebooks/outputs_accuracy")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def base_id(path: Path) -> str:
    name = path.stem
    name = re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", name)
    name = re.sub(r"\.rf\.[0-9a-f]+$", "", name)
    name = re.sub(r"^(train|test|valid)_", "", name)
    return name


def collect_samples(data_root: Path, dataset_mode: str, limit_per_class: int) -> list[Sample]:
    sources = ["raw", "processed"] if dataset_mode == "raw_processed" else [dataset_mode]
    samples: list[Sample] = []
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for class_idx, class_name in enumerate(CLASSES):
        class_samples: list[Sample] = []
        for source in sources:
            folder = data_root / source / class_name
            if not folder.exists():
                continue
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() not in suffixes:
                    continue
                group = f"{class_name}:{base_id(path)}"
                class_samples.append(Sample(str(path), class_idx, source, group))

        if limit_per_class > 0:
            class_samples = class_samples[:limit_per_class]
        samples.extend(class_samples)

    if not samples:
        raise RuntimeError(f"No samples found under {data_root} for dataset mode {dataset_mode}.")
    return samples


def split_by_group(samples: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    rng = random.Random(seed)
    train: list[Sample] = []
    val: list[Sample] = []

    for class_idx in range(len(CLASSES)):
        class_samples = [sample for sample in samples if sample.label == class_idx]
        groups: dict[str, list[Sample]] = {}
        for sample in class_samples:
            groups.setdefault(sample.group, []).append(sample)

        group_keys = list(groups)
        rng.shuffle(group_keys)
        val_group_count = max(1, int(round(len(group_keys) * val_ratio)))
        val_groups = set(group_keys[:val_group_count])

        for group_key, group_samples in groups.items():
            if group_key in val_groups:
                val.extend(group_samples)
            else:
                train.extend(group_samples)

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class CarBodyDataset(Dataset):
    def __init__(self, samples: list[Sample], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        return self.transform(image), sample.label


def build_transforms(model: nn.Module, img_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    data_config = resolve_model_data_config(model)
    mean = data_config.get("mean", (0.485, 0.456, 0.406))
    std = data_config.get("std", (0.229, 0.224, 0.225))
    interpolation = transforms.InterpolationMode.BICUBIC
    resize_size = int(img_size / data_config.get("crop_pct", 0.875))

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.55, 1.0), interpolation=interpolation),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.18), ratio=(0.3, 3.3), value="random"),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, val_transform


def make_loaders(args: argparse.Namespace, model: nn.Module) -> tuple[DataLoader, DataLoader, list[Sample], list[Sample]]:
    samples = collect_samples(Path(args.data_root), args.dataset_mode, args.limit_per_class)
    train_samples, val_samples = split_by_group(samples, args.val_ratio, args.seed)
    train_transform, val_transform = build_transforms(model, args.img_size)

    train_loader = DataLoader(
        CarBodyDataset(train_samples, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        CarBodyDataset(val_samples, val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_samples, val_samples


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    mixup_fn: Mixup | None,
    use_amp: bool,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        hard_labels = labels

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size
        correct += outputs.argmax(dim=1).eq(hard_labels).sum().item()

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return total_loss / max(total, 1), correct / max(total, 1), macro_f1, y_true, y_pred


def save_artifacts(history: dict[str, list[float]], y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.grid(alpha=0.25)
    ax1.legend()
    ax2.plot(history["train_acc"], label="Train Accuracy")
    ax2.plot(history["val_acc"], label="Val Accuracy")
    ax2.plot(history["val_f1"], label="Val Macro F1")
    ax2.set_title("Accuracy and Macro F1")
    ax2.set_xlabel("Epoch")
    ax2.grid(alpha=0.25)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))), normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Dataset mode: {args.dataset_mode}")

    model = timm.create_model(args.model, pretrained=not args.no_pretrained, num_classes=len(CLASSES))
    model.to(device)

    train_loader, val_loader, train_samples, val_samples = make_loaders(args, model)
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print("Train counts:", np.bincount([s.label for s in train_samples], minlength=len(CLASSES)).tolist())
    print("Val counts:", np.bincount([s.label for s in val_samples], minlength=len(CLASSES)).tolist())

    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            prob=1.0,
            switch_prob=0.5,
            label_smoothing=args.label_smoothing,
            num_classes=len(CLASSES),
        )
        train_criterion: nn.Module = SoftTargetCrossEntropy()
    else:
        train_criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    val_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    final_true = np.array([])
    final_pred = np.array([])

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, train_criterion, optimizer, device, scaler, mixup_fn, use_amp
        )
        val_loss, val_acc, val_f1, y_true, y_pred = validate(model, val_loader, val_criterion, device, use_amp)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} time={time.time() - started:.1f}s"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            stale_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
            final_true = y_true
            final_pred = y_pred
            torch.save(best_state, output_dir / "best_model.pt")
            print(f"  saved new best model with macro F1={best_f1:.4f}")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    if best_state is None:
        raise RuntimeError("Training ended without a valid best model.")

    with (output_dir / "class_names.json").open("w", encoding="utf-8") as f:
        json.dump(CLASSES, f, indent=2)
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args) | {"best_epoch": best_epoch, "best_macro_f1": best_f1}, f, indent=2)

    save_artifacts(history, final_true, final_pred, output_dir)
    report = classification_report(final_true, final_pred, target_names=CLASSES, zero_division=0)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Best epoch: {best_epoch}")
    print(f"Best macro F1: {best_f1:.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
