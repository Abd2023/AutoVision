"""
Phase 3: Model Architecture & Training Strategy
================================================
Train a FastViT-S12 model for 8-class car body type classification.
Uses Focal Loss + Label Smoothing, heavy augmentation, and Early Stopping.
Generates evaluation artifacts: loss/accuracy curves and confusion matrix.

Usage (from project root):
    python notebooks/train.py
"""

import os
import sys
import copy
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import timm
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    # Paths (relative to project root)
    "data_dir": "data/processed",
    "output_dir": "notebooks/outputs",
    "model_save_path": "notebooks/outputs/best_model.pt",

    # Model
    "model_name": "fastvit_s12",  # Lightweight, ~9M params, fast on CPU
    "num_classes": 8,
    "input_size": 224,
    "pretrained": True,

    # Training
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,

    # Focal Loss
    "focal_gamma": 2.0,       # Focus on hard examples
    "label_smoothing": 0.1,   # Prevent overconfidence

    # Early Stopping
    "patience": 7,            # Stop after 7 epochs with no improvement

    # Data split
    "val_ratio": 0.2,         # 80% train, 20% validation

    # Reproducibility
    "seed": 42,
}

CLASSES = [
    "F1", "HATCHBACK", "MICRO", "PICK_UP",
    "SEDAN", "STATION_WAGON", "SUV", "VAN"
]

# Visual similarity matrix for Knowledge-Distillation Label Smoothing
# Higher value = more visually similar (used for soft targets)
SIMILARITY_MATRIX = {
    "F1":            {"F1": 1.0, "HATCHBACK": 0.01, "MICRO": 0.01, "PICK_UP": 0.01, "SEDAN": 0.01, "STATION_WAGON": 0.01, "SUV": 0.01, "VAN": 0.01},
    "HATCHBACK":     {"F1": 0.01, "HATCHBACK": 1.0, "MICRO": 0.35, "PICK_UP": 0.02, "SEDAN": 0.20, "STATION_WAGON": 0.15, "SUV": 0.10, "VAN": 0.05},
    "MICRO":         {"F1": 0.01, "HATCHBACK": 0.35, "MICRO": 1.0, "PICK_UP": 0.01, "SEDAN": 0.05, "STATION_WAGON": 0.02, "SUV": 0.02, "VAN": 0.05},
    "PICK_UP":       {"F1": 0.01, "HATCHBACK": 0.02, "MICRO": 0.01, "PICK_UP": 1.0, "SEDAN": 0.05, "STATION_WAGON": 0.10, "SUV": 0.25, "VAN": 0.15},
    "SEDAN":         {"F1": 0.01, "HATCHBACK": 0.20, "MICRO": 0.05, "PICK_UP": 0.05, "SEDAN": 1.0, "STATION_WAGON": 0.25, "SUV": 0.10, "VAN": 0.05},
    "STATION_WAGON": {"F1": 0.01, "HATCHBACK": 0.15, "MICRO": 0.02, "PICK_UP": 0.10, "SEDAN": 0.25, "STATION_WAGON": 1.0, "SUV": 0.30, "VAN": 0.10},
    "SUV":           {"F1": 0.01, "HATCHBACK": 0.10, "MICRO": 0.02, "PICK_UP": 0.25, "SEDAN": 0.10, "STATION_WAGON": 0.30, "SUV": 1.0, "VAN": 0.15},
    "VAN":           {"F1": 0.01, "HATCHBACK": 0.05, "MICRO": 0.05, "PICK_UP": 0.15, "SEDAN": 0.05, "STATION_WAGON": 0.10, "SUV": 0.15, "VAN": 1.0},
}


# ==========================================
# SEED FOR REPRODUCIBILITY
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# FOCAL LOSS WITH KNOWLEDGE-DISTILLATION LABEL SMOOTHING
# ==========================================
class FocalLossWithKDSmoothing(nn.Module):
    """
    Combines Focal Loss (focus on hard examples) with
    Knowledge-Distillation-Based Label Smoothing (distribute smoothing
    mass proportionally based on visual similarity, not uniformly).
    """
    def __init__(self, gamma=2.0, smoothing=0.1, similarity_matrix=None, num_classes=8):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.num_classes = num_classes

        # Build the soft-target matrix from the similarity dictionary
        if similarity_matrix is not None:
            self.register_buffer('soft_targets', self._build_soft_targets(similarity_matrix))
        else:
            # Fallback to uniform label smoothing
            self.soft_targets = None

    def _build_soft_targets(self, sim_dict):
        """
        Convert the similarity dictionary into a normalized soft-target matrix.
        For each true class, the smoothing mass is distributed proportionally
        to visual similarity instead of uniformly.
        """
        n = len(CLASSES)
        matrix = torch.zeros(n, n)
        for i, cls_i in enumerate(CLASSES):
            for j, cls_j in enumerate(CLASSES):
                matrix[i, j] = sim_dict[cls_i][cls_j]

        # For each row: normalize the off-diagonal entries to sum to 1
        for i in range(n):
            matrix[i, i] = 0  # Zero out diagonal temporarily
            row_sum = matrix[i].sum()
            if row_sum > 0:
                matrix[i] = matrix[i] / row_sum  # Normalize off-diagonal
            matrix[i] = matrix[i] * self.smoothing  # Scale by smoothing factor
            matrix[i, i] = 1.0 - self.smoothing     # Confidence on true class

        return matrix

    def forward(self, logits, targets):
        # Get soft targets for the batch
        if self.soft_targets is not None:
            soft_labels = self.soft_targets[targets]  # (batch, num_classes)
        else:
            # Uniform label smoothing fallback
            soft_labels = torch.full_like(logits, self.smoothing / (self.num_classes - 1))
            soft_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # Compute log-softmax
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Focal modulation: (1 - p_t)^gamma
        # p_t is the probability assigned to the TRUE class
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (batch,)
        focal_weight = (1 - p_t) ** self.gamma  # (batch,)

        # KL-divergence style loss with soft labels
        loss_per_sample = -(soft_labels * log_probs).sum(dim=1)  # (batch,)

        # Apply focal weighting
        loss = (focal_weight * loss_per_sample).mean()

        return loss


# ==========================================
# DATA LOADING
# ==========================================
class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def get_data_loaders(config):
    """Create train and validation dataloaders with augmentation."""

    # Training augmentations (heavy, as recommended)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config["input_size"], scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Validation: no augmentation, only resize + normalize
    val_transform = transforms.Compose([
        transforms.Resize((config["input_size"], config["input_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load full dataset with train transforms first (we'll override for val)
    full_dataset = datasets.ImageFolder(config["data_dir"])

    # Verify classes match expectations
    print(f"Detected classes: {full_dataset.classes}")
    print(f"Class to index: {full_dataset.class_to_idx}")

    # Stratified split: ensure equal proportions per class
    train_indices = []
    val_indices = []

    targets = np.array(full_dataset.targets)
    for class_idx in range(config["num_classes"]):
        class_indices = np.where(targets == class_idx)[0]
        np.random.shuffle(class_indices)

        split = int(len(class_indices) * (1 - config["val_ratio"]))
        train_indices.extend(class_indices[:split])
        val_indices.extend(class_indices[split:])

    # Create subsets
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    # Override transforms by wrapping
    train_dataset = TransformDataset(train_subset, train_transform)
    val_dataset = TransformDataset(val_subset, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
                            shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    return train_loader, val_loader, full_dataset.classes


# ==========================================
# MODEL
# ==========================================
def create_model(config):
    """Instantiate a pre-trained FastViT-S12 and replace the head."""
    model = timm.create_model(
        config["model_name"],
        pretrained=config["pretrained"],
        num_classes=config["num_classes"]
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config['model_name']}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


# ==========================================
# TRAINING LOOP
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


# ==========================================
# EVALUATION & PLOTTING
# ==========================================
def plot_training_curves(history, output_dir):
    """Plot and save loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(history["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(history["val_loss"], label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training vs Validation Loss", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(history["train_acc"], label="Train Accuracy", linewidth=2)
    ax2.plot(history["val_acc"], label="Val Accuracy", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Training vs Validation Accuracy", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved training curves to {path}")


def plot_confusion_matrix(all_labels, all_preds, class_names, output_dir):
    """Plot and save the normalized 8x8 confusion matrix heatmap."""
    cm = confusion_matrix(all_labels, all_preds, normalize="true")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Normalized Confusion Matrix", fontsize=14)

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {path}")


# ==========================================
# MAIN
# ==========================================
def main():
    config = CONFIG
    set_seed(config["seed"])

    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, class_names = get_data_loaders(config)

    # Model
    model = create_model(config)
    model = model.to(device)

    # Loss function: Focal Loss + KD Label Smoothing
    criterion = FocalLossWithKDSmoothing(
        gamma=config["focal_gamma"],
        smoothing=config["label_smoothing"],
        similarity_matrix=SIMILARITY_MATRIX,
        num_classes=config["num_classes"]
    ).to(device)

    # Optimizer: AdamW with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    # Learning rate scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )

    # Training history
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    # Early Stopping
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Starting Training: {config['epochs']} epochs, patience={config['patience']}")
    print(f"{'='*60}\n")

    for epoch in range(config["epochs"]):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)

        # Compute macro F1
        val_f1 = f1_score(val_labels, val_preds, average="macro")

        # Step scheduler
        scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Print epoch summary
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch+1:02d}/{config['epochs']}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Val F1: {val_f1:.4f} | LR: {lr:.6f}")

        # Early Stopping check (based on val F1 score, which is the project metric)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  -> New best model! Val F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break

    # Save best model
    torch.save(best_model_state, config["model_save_path"])
    print(f"\nBest model saved to {config['model_save_path']}")
    print(f"Best Val Accuracy: {best_val_acc:.4f} | Best Val F1: {best_val_f1:.4f}")

    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    _, _, final_preds, final_labels = validate(model, val_loader, criterion, device)

    # Generate evaluation artifacts
    print("\nGenerating evaluation plots...")
    plot_training_curves(history, config["output_dir"])
    plot_confusion_matrix(final_labels, final_preds, class_names, config["output_dir"])

    # Print detailed classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(final_labels, final_preds, target_names=class_names))

    # Final F1 score
    final_f1 = f1_score(final_labels, final_preds, average="macro")
    print(f"Final Macro F1-Score: {final_f1:.4f}")


if __name__ == "__main__":
    main()
