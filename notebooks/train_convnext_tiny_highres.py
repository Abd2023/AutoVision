"""
Train a high-resolution ConvNeXt-Tiny AutoVision classifier.

This script is intentionally separate from train_resnet50.py so the ResNet and
EfficientNet workflow stays stable while this experiment tests a stronger
fine-grained backbone, moderate augmentation, and optional hflip TTA.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.transforms import InterpolationMode

from train_resnet50 import (
    EpochMetrics,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PROJECT_CLASSES,
    balanced_subset,
    class_counts,
    choose_device,
    configure_trainable_params,
    current_lr,
    load_checkpoint,
    make_grad_scaler,
    make_loaders,
    make_loss,
    make_optimizer,
    metric_improved,
    plot_history,
    print_counts,
    project_path,
    save_checkpoint,
    set_seed,
    train_one_epoch,
    validate_classes,
    write_eval_artifacts,
    write_history,
    write_run_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train high-resolution ConvNeXt-Tiny for AutoVision.")
    parser.add_argument("--data-root", default="data/processed_convnext_tiny_320")
    parser.add_argument("--output-dir", default="notebooks/outputs/convnext_tiny_highres_320_v1")
    parser.add_argument("--model-source", default="torchvision")
    parser.add_argument("--model-name", default="convnext_tiny")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--aug-profile", choices=["moderate", "strong"], default="moderate")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--loss-type", choices=["cross_entropy", "focal"], default="cross_entropy")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-tta", choices=["none", "hflip"], default="hflip")
    parser.add_argument("--weighted-sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weighted-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    return parser.parse_args()


def build_transforms(image_size: int, aug_profile: str) -> tuple[transforms.Compose, transforms.Compose]:
    if aug_profile == "strong":
        train_tfms = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.60, 1.0), ratio=(0.75, 1.33), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=12, interpolation=InterpolationMode.BILINEAR, fill=255),
                transforms.RandomApply([transforms.ColorJitter(0.45, 0.45, 0.30, 0.08)], p=0.90),
                transforms.RandomGrayscale(p=0.03),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.12),
                transforms.RandomPerspective(distortion_scale=0.15, p=0.20, interpolation=InterpolationMode.BILINEAR, fill=255),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.90, 1.10), shear=6, interpolation=InterpolationMode.BILINEAR, fill=255),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.35, scale=(0.02, 0.18), ratio=(0.3, 3.3), value="random"),
            ]
        )
    else:
        train_tfms = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0), ratio=(0.85, 1.20), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8, interpolation=InterpolationMode.BILINEAR, fill=255),
                transforms.RandomApply([transforms.ColorJitter(0.25, 0.25, 0.18, 0.04)], p=0.65),
                transforms.RandomApply(
                    [
                        transforms.RandomAffine(
                            degrees=0,
                            translate=(0.03, 0.03),
                            scale=(0.95, 1.05),
                            shear=3,
                            interpolation=InterpolationMode.BILINEAR,
                            fill=255,
                        )
                    ],
                    p=0.18,
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.12, scale=(0.02, 0.10), ratio=(0.5, 2.0), value="random"),
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


def create_datasets(data_root: Path, args: argparse.Namespace) -> tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder, list[str]]:
    train_tfms, eval_tfms = build_transforms(args.image_size, args.aug_profile)
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


def build_model(args: argparse.Namespace, num_classes: int) -> nn.Module:
    weights = ConvNeXt_Tiny_Weights.DEFAULT if args.pretrained else None
    model = convnext_tiny(weights=weights)
    norm_layer = model.classifier[0]
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        norm_layer,
        nn.Flatten(start_dim=1),
        nn.Dropout(p=args.dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


@torch.no_grad()
def evaluate_with_tta(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
    eval_tta: str,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels: list[int] = []
    all_preds: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        if eval_tta == "hflip":
            logits = (logits + model(torch.flip(images, dims=[3]))) / 2.0
        loss = loss_fn(logits, labels)

        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((preds == labels).sum().item())
        total_samples += batch_size
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    matrix = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    per_class_acc = []
    for class_index in range(num_classes):
        support = int(matrix[class_index].sum())
        per_class_acc.append(float(matrix[class_index, class_index] / support) if support else 0.0)

    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),
        "balanced_acc": float(sum(per_class_acc) / len(per_class_acc)),
        "confusion_matrix": matrix,
        "per_class_acc": per_class_acc,
        "labels": all_labels,
        "preds": all_preds,
    }


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

    best_path = output_dir / "best_convnext_tiny.pt"
    last_path = output_dir / "last_convnext_tiny.pt"
    history_path = output_dir / "training_history.csv"

    history: list[EpochMetrics] = []
    best_val_balanced_acc = -1.0
    best_val_acc = -1.0
    epochs_without_improvement = 0
    optimizer = None
    scheduler = None
    active_phase = None

    for epoch in range(1, args.epochs + 1):
        freeze_backbone = epoch <= args.freeze_epochs
        phase = configure_trainable_params(model, freeze_backbone)
        if phase != active_phase:
            optimizer = make_optimizer(model, args, freeze_backbone)
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - epoch + 1), eta_min=args.min_lr)
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
        val_metrics = evaluate_with_tta(model, val_loader, loss_fn, device, len(class_names), args.eval_tta)
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
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, phase, args, class_names, train_counts, val_counts, test_counts, best_val_balanced_acc, best_val_acc)
            write_eval_artifacts(output_dir, "val_best", val_metrics, class_names)
        else:
            epochs_without_improvement += 1

        save_checkpoint(last_path, model, optimizer, scheduler, epoch, phase, args, class_names, train_counts, val_counts, test_counts, best_val_balanced_acc, best_val_acc)

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
    test_metrics = evaluate_with_tta(model, test_loader, loss_fn, device, len(class_names), args.eval_tta)
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
