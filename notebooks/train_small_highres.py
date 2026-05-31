"""
Train small high-resolution AutoVision classifiers.

This keeps the ConvNeXt recipe separate from the legacy trainer while testing
models that can produce fp32 inference checkpoints under the homework limit.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    efficientnet_b0,
    mobilenet_v3_large,
)

from train_convnext_tiny_highres import build_transforms, evaluate_with_tta
from train_resnet50 import (
    EpochMetrics,
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


SMALL_MODELS = {"efficientnet_b0", "mobilenet_v3_large"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train small high-resolution AutoVision classifiers.")
    parser.add_argument("--data-root", default="data/processed_convnext_tiny_320")
    parser.add_argument("--output-dir", default="notebooks/outputs/efficientnet_b0_highres_320_v1")
    parser.add_argument("--model-source", default="torchvision")
    parser.add_argument("--model-name", choices=sorted(SMALL_MODELS), default="efficientnet_b0")
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
    if args.model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if args.pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(in_features, num_classes),
        )
        return model

    if args.model_name == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.DEFAULT if args.pretrained else None
        model = mobilenet_v3_large(weights=weights)
        if isinstance(model.classifier[2], nn.Dropout):
            model.classifier[2] = nn.Dropout(p=args.dropout, inplace=True)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported small model: {args.model_name}")


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
