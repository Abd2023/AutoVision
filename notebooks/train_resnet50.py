"""
Train an ImageNet-pretrained body-type classifier on the frozen AutoVision dataset.

Expected dataset layout:
    data/processed/train/<CLASS>/*.jpg
    data/processed/val/<CLASS>/*.jpg
    data/processed/test/<CLASS>/*.jpg

Default training choices are tuned for the current project constraints:
    - ImageNet-pretrained transfer learning
    - frozen-backbone warmup, then full fine-tuning
    - WeightedRandomSampler for imbalanced classes
    - class-weighted CrossEntropyLoss
    - strong augmentation to reduce synthetic-domain overfitting
    - best checkpoint selected by validation balanced accuracy
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    efficientnet_b1,
    resnet50,
)
from torchvision.transforms import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TORCHVISION_MODELS = {"resnet50", "efficientnet_b0", "efficientnet_b1"}


@dataclass
class EpochMetrics:
    epoch: int
    phase: str
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_balanced_acc: float
    learning_rate: float
    seconds: float


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, reduction="none", weight=self.weight)
        pt = torch.softmax(logits, dim=1).gather(1, labels.unsqueeze(1)).squeeze(1).clamp(min=1e-8, max=1.0)
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AutoVision classifier.")
    parser.add_argument("--data-root", default="data/processed", help="Frozen train/val/test dataset root.")
    parser.add_argument("--output-dir", default="notebooks/outputs/resnet50", help="Where checkpoints and metrics are saved.")
    parser.add_argument("--model-source", choices=["torchvision", "timm"], default="torchvision")
    parser.add_argument(
        "--model-name",
        choices=sorted(TORCHVISION_MODELS),
        default="resnet50",
        help="Backbone architecture to fine-tune.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--freeze-epochs", type=int, default=3, help="Train only the classifier head for this many epochs.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows unless multiprocessing is configured.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Classifier-head learning rate.")
    parser.add_argument("--backbone-lr", type=float, default=3e-5, help="Backbone learning rate after unfreezing.")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--loss-type", choices=["cross_entropy", "focal"], default="cross_entropy")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8, help="Early-stop patience on validation balanced accuracy. Use 0 to disable.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use CUDA mixed precision when available.")
    parser.add_argument("--weighted-sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weighted-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-train", type=int, default=0, help="Debug only: balanced subset size for train.")
    parser.add_argument("--limit-val", type=int, default=0, help="Debug only: balanced subset size for val.")
    parser.add_argument("--limit-test", type=int, default=0, help="Debug only: balanced subset size for test.")
    return parser.parse_args()


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(device_arg)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.60, 1.0),
                ratio=(0.75, 1.33),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12, interpolation=InterpolationMode.BILINEAR, fill=255),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.45, contrast=0.45, saturation=0.30, hue=0.08)],
                p=0.90,
            ),
            transforms.RandomGrayscale(p=0.03),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.12),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.20, interpolation=InterpolationMode.BILINEAR, fill=255),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.90, 1.10),
                shear=6,
                interpolation=InterpolationMode.BILINEAR,
                fill=255,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.35, scale=(0.02, 0.18), ratio=(0.3, 3.3), value="random"),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tfms, eval_tfms


def validate_classes(dataset: datasets.ImageFolder, split_name: str) -> None:
    if dataset.classes != PROJECT_CLASSES:
        raise ValueError(
            f"{split_name} classes do not match expected AutoVision order.\n"
            f"Expected: {PROJECT_CLASSES}\n"
            f"Found:    {dataset.classes}"
        )


def balanced_subset(dataset: datasets.ImageFolder, limit: int, seed: int) -> Subset | datasets.ImageFolder:
    if limit <= 0 or limit >= len(dataset):
        return dataset

    rng = random.Random(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(dataset.targets):
        by_class[target].append(index)
    for indices in by_class.values():
        rng.shuffle(indices)

    per_class = max(1, limit // len(dataset.classes))
    selected: list[int] = []
    remaining: list[int] = []
    for target in range(len(dataset.classes)):
        class_indices = by_class[target]
        selected.extend(class_indices[:per_class])
        remaining.extend(class_indices[per_class:])

    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, limit - len(selected))])
    selected = selected[:limit]
    rng.shuffle(selected)
    return Subset(dataset, selected)


def subset_targets(dataset: datasets.ImageFolder | Subset) -> list[int]:
    if isinstance(dataset, Subset):
        base_targets = dataset.dataset.targets
        return [int(base_targets[index]) for index in dataset.indices]
    return [int(target) for target in dataset.targets]


def class_counts(dataset: datasets.ImageFolder | Subset, num_classes: int) -> list[int]:
    counts = Counter(subset_targets(dataset))
    return [counts.get(index, 0) for index in range(num_classes)]


def create_datasets(data_root: Path, args: argparse.Namespace) -> tuple[datasets.ImageFolder | Subset, datasets.ImageFolder | Subset, datasets.ImageFolder | Subset, list[str]]:
    train_tfms, eval_tfms = build_transforms(args.image_size)
    train_root = data_root / "train"
    val_root = data_root / "val"
    test_root = data_root / "test"
    for split_root in [train_root, val_root, test_root]:
        if not split_root.exists():
            raise FileNotFoundError(f"Missing split folder: {split_root}")

    train_ds = datasets.ImageFolder(train_root, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_root, transform=eval_tfms)
    test_ds = datasets.ImageFolder(test_root, transform=eval_tfms)

    validate_classes(train_ds, "train")
    validate_classes(val_ds, "val")
    validate_classes(test_ds, "test")

    train_ds = balanced_subset(train_ds, args.limit_train, args.seed)
    val_ds = balanced_subset(val_ds, args.limit_val, args.seed + 1)
    test_ds = balanced_subset(test_ds, args.limit_test, args.seed + 2)
    return train_ds, val_ds, test_ds, PROJECT_CLASSES


def make_sampler(dataset: datasets.ImageFolder | Subset, num_classes: int, seed: int) -> WeightedRandomSampler:
    targets = subset_targets(dataset)
    counts = Counter(targets)
    sample_weights = [1.0 / max(counts[target], 1) for target in targets]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def make_loaders(
    train_ds: datasets.ImageFolder | Subset,
    val_ds: datasets.ImageFolder | Subset,
    test_ds: datasets.ImageFolder | Subset,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    sampler = make_sampler(train_ds, len(PROJECT_CLASSES), args.seed) if args.weighted_sampler else None
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def build_model(args: argparse.Namespace, num_classes: int) -> nn.Module:
    if args.model_source == "timm":
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("timm is not installed. Install requirements-train.txt or use --model-source torchvision.") from exc
        return timm.create_model(
            args.model_name,
            pretrained=args.pretrained,
            num_classes=num_classes,
            drop_rate=args.dropout,
        )

    if args.model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if args.pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=args.dropout), nn.Linear(in_features, num_classes))
        return model

    if args.model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if args.pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(in_features, num_classes),
        )
        return model

    if args.model_name == "efficientnet_b1":
        weights = EfficientNet_B1_Weights.DEFAULT if args.pretrained else None
        model = efficientnet_b1(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(in_features, num_classes),
        )
        return model

    raise ValueError(f"Unsupported torchvision model_name: {args.model_name}")


def classifier_head(model: nn.Module) -> nn.Module:
    if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        return model.fc
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        return model.classifier
    if hasattr(model, "get_classifier"):
        head = model.get_classifier()
        if isinstance(head, nn.Module):
            return head
    raise RuntimeError("Could not locate classifier head for freezing/optimizer setup.")


def configure_trainable_params(model: nn.Module, freeze_backbone: bool) -> str:
    head = classifier_head(model)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in head.parameters():
            parameter.requires_grad = True
        return "head"

    for parameter in model.parameters():
        parameter.requires_grad = True
    return "finetune"


def make_optimizer(model: nn.Module, args: argparse.Namespace, freeze_backbone: bool) -> AdamW:
    head = classifier_head(model)
    head_param_ids = {id(parameter) for parameter in head.parameters()}
    head_params = [parameter for parameter in head.parameters() if parameter.requires_grad]
    backbone_params = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in head_param_ids and parameter.requires_grad
    ]

    param_groups: list[dict[str, Any]] = []
    if backbone_params and not freeze_backbone:
        param_groups.append({"params": backbone_params, "lr": args.backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr})
    if not param_groups:
        raise RuntimeError("No trainable parameters found.")
    return AdamW(param_groups, weight_decay=args.weight_decay)


def build_class_weights(train_counts: list[int]) -> torch.Tensor:
    total = float(sum(train_counts))
    num_classes = len(train_counts)
    raw_weights = [total / (num_classes * max(count, 1)) for count in train_counts]
    weights = torch.tensor(raw_weights, dtype=torch.float32)
    return weights / weights.mean()


def make_loss(train_counts: list[int], args: argparse.Namespace, device: torch.device) -> nn.Module:
    weights = build_class_weights(train_counts).to(device) if args.class_weighted_loss else None
    if args.loss_type == "focal":
        return FocalLoss(gamma=args.focal_gamma, weight=weights)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)


def make_grad_scaler(use_amp: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def autocast_context(use_amp: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=use_amp)
    return torch.cuda.amp.autocast(enabled=use_amp)


def batch_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> int:
    return int((logits.argmax(dim=1) == labels).sum().item())


def keep_frozen_batchnorm_eval(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: AdamW,
    device: torch.device,
    scaler: Any,
    use_amp: bool,
    grad_clip: float,
    freeze_backbone: bool,
) -> tuple[float, float]:
    model.train()
    if freeze_backbone:
        keep_frozen_batchnorm_eval(model)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += batch_accuracy(logits.detach(), labels)
        total_seen += batch_size

    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    all_labels: list[int] = []
    all_preds: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((preds == labels).sum().item())
        total_seen += batch_size
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    matrix = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    per_class_total = matrix.sum(axis=1)
    per_class_correct = matrix.diagonal()
    per_class_acc = np.divide(
        per_class_correct,
        per_class_total,
        out=np.zeros_like(per_class_correct, dtype=float),
        where=per_class_total != 0,
    )
    balanced_acc = float(per_class_acc[per_class_total != 0].mean()) if np.any(per_class_total != 0) else 0.0
    return {
        "loss": total_loss / max(total_seen, 1),
        "acc": total_correct / max(total_seen, 1),
        "balanced_acc": balanced_acc,
        "confusion_matrix": matrix,
        "per_class_acc": per_class_acc,
        "labels": all_labels,
        "preds": all_preds,
    }


def current_lr(optimizer: AdamW) -> float:
    return float(max(group["lr"] for group in optimizer.param_groups))


def metric_improved(candidate: dict[str, Any], best_balanced_acc: float, best_acc: float) -> bool:
    balanced_acc = float(candidate["balanced_acc"])
    acc = float(candidate["acc"])
    if balanced_acc > best_balanced_acc + 1e-8:
        return True
    return math.isclose(balanced_acc, best_balanced_acc, rel_tol=0.0, abs_tol=1e-8) and acc > best_acc


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    phase: str,
    args: argparse.Namespace,
    class_names: list[str],
    train_counts: list[int],
    val_counts: list[int],
    test_counts: list[int],
    best_val_balanced_acc: float,
    best_val_acc: float,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "phase": phase,
        "model_source": args.model_source,
        "model_name": args.model_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "class_names": class_names,
        "class_to_idx": {name: index for index, name in enumerate(class_names)},
        "image_size": args.image_size,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "test_counts": test_counts,
        "best_val_balanced_acc": best_val_balanced_acc,
        "best_val_acc": best_val_acc,
        "args": vars(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def write_history(path: Path, metrics: list[EpochMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(metric) for metric in metrics]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else list(EpochMetrics.__annotations__.keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize_confusion_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)


def write_confusion_matrix(path: Path, matrix: np.ndarray, class_names: list[str], decimals: int | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *class_names])
        for class_name, row in zip(class_names, matrix):
            values = row.tolist()
            if decimals is not None:
                values = [f"{float(value):.{decimals}f}" for value in values]
            writer.writerow([class_name, *values])


def write_eval_artifacts(output_dir: Path, split_name: str, metrics: dict[str, Any], class_names: list[str]) -> None:
    matrix = metrics["confusion_matrix"]
    normalized_matrix = normalize_confusion_matrix(matrix)
    per_class = {
        class_name: float(metrics["per_class_acc"][index])
        for index, class_name in enumerate(class_names)
    }
    summary = {
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["acc"]),
        "balanced_accuracy": float(metrics["balanced_acc"]),
        "per_class_accuracy": per_class,
    }
    (output_dir / f"{split_name}_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_confusion_matrix(output_dir / f"{split_name}_confusion_matrix.csv", normalized_matrix, class_names, decimals=4)
    write_confusion_matrix(output_dir / f"{split_name}_confusion_matrix_counts.csv", matrix, class_names)
    report = classification_report(
        metrics["labels"],
        metrics["preds"],
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    (output_dir / f"{split_name}_classification_report.txt").write_text(report, encoding="utf-8")
    plot_confusion_matrix(normalized_matrix, class_names, output_dir / f"{split_name}_confusion_matrix.png", normalize=True)
    plot_confusion_matrix(matrix, class_names, output_dir / f"{split_name}_confusion_matrix_counts.png", normalize=False)


def plot_history(path: Path, metrics: list[EpochMetrics]) -> None:
    if not metrics:
        return
    epochs = [metric.epoch for metric in metrics]
    train_loss = [metric.train_loss for metric in metrics]
    val_loss = [metric.val_loss for metric in metrics]
    train_acc = [metric.train_acc for metric in metrics]
    val_acc = [metric.val_acc for metric in metrics]
    val_balanced = [metric.val_balanced_acc for metric in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, train_loss, label="Train loss")
    axes[0].plot(epochs, val_loss, label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, train_acc, label="Train acc")
    axes[1].plot(epochs, val_acc, label="Val acc")
    axes[1].plot(epochs, val_balanced, label="Val balanced acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], path: Path, normalize: bool) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    threshold = matrix.max() / 2.0 if matrix.size and matrix.max() > 0 else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            raw_value = float(matrix[row, col])
            value = f"{raw_value:.2f}" if normalize else str(int(raw_value))
            ax.text(
                col,
                row,
                value,
                ha="center",
                va="center",
                color="white" if raw_value > threshold else "black",
                fontsize=8,
            )

    title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix (Counts)"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    class_names: list[str],
    train_counts: list[int],
    val_counts: list[int],
    test_counts: list[int],
) -> None:
    config = {
        "args": vars(args),
        "class_names": class_names,
        "train_counts": dict(zip(class_names, train_counts)),
        "val_counts": dict(zip(class_names, val_counts)),
        "test_counts": dict(zip(class_names, test_counts)),
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def print_counts(split_name: str, counts: list[int], class_names: list[str]) -> None:
    print(f"{split_name}:")
    for class_name, count in zip(class_names, counts):
        print(f"  {class_name:14s} {count:5d}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root = project_path(args.data_root)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    train_ds, val_ds, test_ds, class_names = create_datasets(data_root, args)
    train_counts = class_counts(train_ds, len(class_names))
    val_counts = class_counts(val_ds, len(class_names))
    test_counts = class_counts(test_ds, len(class_names))
    write_run_config(output_dir, args, class_names, train_counts, val_counts, test_counts)

    print(f"Device: {device}")
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print_counts("train", train_counts, class_names)
    print_counts("val", val_counts, class_names)
    print_counts("test", test_counts, class_names)

    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, args, device)
    model = build_model(args, len(class_names)).to(device)
    loss_fn = make_loss(train_counts, args, device)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = make_grad_scaler(use_amp)
    best_path = output_dir / f"best_{args.model_name}.pt"
    last_path = output_dir / f"last_{args.model_name}.pt"
    history_path = output_dir / "training_history.csv"

    history: list[EpochMetrics] = []
    best_val_balanced_acc = -1.0
    best_val_acc = -1.0
    epochs_without_improvement = 0

    optimizer: AdamW | None = None
    scheduler: CosineAnnealingLR | None = None
    active_phase: str | None = None

    for epoch in range(1, args.epochs + 1):
        freeze_backbone = epoch <= args.freeze_epochs
        phase = configure_trainable_params(model, freeze_backbone)
        if phase != active_phase:
            optimizer = make_optimizer(model, args, freeze_backbone)
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs - epoch + 1),
                eta_min=args.min_lr,
            )
            active_phase = phase
            print(f"\nStarting phase: {phase}")

        assert optimizer is not None
        assert scheduler is not None

        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=args.grad_clip,
            freeze_backbone=freeze_backbone,
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device, len(class_names))
        scheduler.step()
        seconds = time.time() - start

        epoch_metrics = EpochMetrics(
            epoch=epoch,
            phase=phase,
            train_loss=float(train_loss),
            train_acc=float(train_acc),
            val_loss=float(val_metrics["loss"]),
            val_acc=float(val_metrics["acc"]),
            val_balanced_acc=float(val_metrics["balanced_acc"]),
            learning_rate=current_lr(optimizer),
            seconds=seconds,
        )
        history.append(epoch_metrics)
        write_history(history_path, history)
        plot_history(output_dir / "training_curves.png", history)

        improved = metric_improved(val_metrics, best_val_balanced_acc, best_val_acc)
        if improved:
            best_val_balanced_acc = float(val_metrics["balanced_acc"])
            best_val_acc = float(val_metrics["acc"])
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                phase,
                args,
                class_names,
                train_counts,
                val_counts,
                test_counts,
                best_val_balanced_acc,
                best_val_acc,
            )
            write_eval_artifacts(output_dir, "val_best", val_metrics, class_names)
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch,
            phase,
            args,
            class_names,
            train_counts,
            val_counts,
            test_counts,
            best_val_balanced_acc,
            best_val_acc,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | {phase:8s} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"val_bal_acc={val_metrics['balanced_acc']:.4f} | "
            f"best_bal_acc={best_val_balanced_acc:.4f} | {seconds:.1f}s"
        )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} epochs without validation balanced-accuracy improvement.")
            break

    if not best_path.exists():
        raise RuntimeError("Training finished without producing a best checkpoint.")

    checkpoint = load_checkpoint(best_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, loss_fn, device, len(class_names))
    write_eval_artifacts(output_dir, "test", test_metrics, class_names)

    print("\nBest checkpoint:")
    print(f"  {best_path}")
    print(f"  val_balanced_acc={best_val_balanced_acc:.4f}")
    print(f"  val_acc={best_val_acc:.4f}")
    print("\nTest result from best checkpoint:")
    print(f"  test_acc={test_metrics['acc']:.4f}")
    print(f"  test_balanced_acc={test_metrics['balanced_acc']:.4f}")
    print("  per-class accuracy:")
    for class_name, score in zip(class_names, test_metrics["per_class_acc"]):
        print(f"    {class_name:14s} {float(score):.4f}")


if __name__ == "__main__":
    main()
