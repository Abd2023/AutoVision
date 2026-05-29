import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, models

# --- Configuration ---
MODEL_PATH = r"C:\fun_project\yazlab_3\notebooks\outputs\convnext_tiny_highres_320_v1\best_convnext_tiny.pt"
TEST_DIR = "test"  # Assuming you run this from C:\fun_project\yazlab_3\data\testing_on_isa_data\
BATCH_SIZE = 16
FOLDER_TO_MODEL_CLASS = {
    "F1 CAR": "F1",
    "HATCHBACK": "HATCHBACK",
    "MICRO": "MICRO",
    "PICKUP": "PICK_UP",
    "SEDAN": "SEDAN",
    "STATION WAGON": "STATION_WAGON",
    "SUV": "SUV",
    "VAN": "VAN",
}


class Letterbox:
    def __init__(self, size=320, background="white"):
        self.size = size
        self.background = background

    def __call__(self, image):
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((self.size, self.size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (self.size, self.size), self.background)
        x = (self.size - image.width) // 2
        y = (self.size - image.height) // 2
        canvas.paste(image, (x, y))
        return canvas


def build_convnext_tiny(num_classes, dropout_rate):
    model = models.convnext_tiny(weights=None)
    norm_layer = model.classifier[0]
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        norm_layer,
        nn.Flatten(start_dim=1),
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes),
    )
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Load Checkpoint Metadata ---
print(f"Loading model weights from: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
state_dict = checkpoint["model_state_dict"]
checkpoint_args = checkpoint.get("args", {})
checkpoint_classes = checkpoint.get("class_names")
model_name = checkpoint.get("model_name", checkpoint_args.get("model_name", "convnext_tiny"))
image_size = int(checkpoint.get("image_size", checkpoint_args.get("image_size", 320)))
dropout_rate = float(checkpoint_args.get("dropout", 0.30))

if model_name != "convnext_tiny":
    raise ValueError(f"Unsupported ConvNeXt evaluator input: checkpoint model_name={model_name!r}")

# --- 2. Data Preprocessing ---
transform = transforms.Compose(
    [
        Letterbox(image_size, background="white"),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# --- 3. Load the Dataset ---
print(f"Loading test images from: {TEST_DIR}")
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = test_dataset.classes
num_classes = len(class_names)
print(f"Found {num_classes} classes in test folder: {class_names}")

mapped_folder_classes = [FOLDER_TO_MODEL_CLASS.get(name, name) for name in class_names]
if checkpoint_classes is not None and mapped_folder_classes != checkpoint_classes:
    raise ValueError(
        "Test folder class order does not match checkpoint order.\n"
        f"Folder classes mapped to model names: {mapped_folder_classes}\n"
        f"Checkpoint classes: {checkpoint_classes}"
    )

# --- 4. Load the Custom Model ---
print(f"Detected model architecture: {model_name}")
print(f"Using image size: {image_size}")
print(f"Applying classifier Dropout(p={dropout_rate})")
model = build_convnext_tiny(num_classes, dropout_rate)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# --- 5. Run Evaluation ---
all_preds = []
all_labels = []

print("\nRunning inference...")
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --- 6. Print Results ---
correct = np.sum(np.array(all_preds) == np.array(all_labels))
total = len(all_labels)
accuracy = correct / total

print("=" * 50)
print(f"OVERALL ACCURACY: {accuracy * 100:.2f}% ({correct}/{total} correct)")
print("=" * 50)

print("\nDetailed Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
