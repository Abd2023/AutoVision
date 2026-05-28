"""
AutoVision backend API.

The React frontend posts images to /api/predict. This backend loads the latest
ResNet50 PyTorch checkpoint and returns class probabilities.

Usage:
    cd backend
    python main.py
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import gradio as gr
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import resnet50


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "notebooks" / "outputs" / "resnet50_focal_round4" / "best_resnet50.pt"
MODEL_PATH = Path(os.environ.get("AUTOVISION_MODEL_PATH", DEFAULT_MODEL_PATH)).resolve()

CLASSES = ["F1", "HATCHBACK", "MICRO", "PICK_UP", "SEDAN", "STATION_WAGON", "SUV", "VAN"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: torch.nn.Module | None = None
class_names = CLASSES
image_size = 224


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_resnet50(num_classes: int, dropout: float) -> torch.nn.Module:
    network = resnet50(weights=None)
    in_features = network.fc.in_features
    network.fc = torch.nn.Sequential(torch.nn.Dropout(p=dropout), torch.nn.Linear(in_features, num_classes))
    return network


def load_model() -> None:
    global model, class_names, image_size

    if not MODEL_PATH.exists():
        print(f"Warning: model checkpoint not found at {MODEL_PATH}")
        return

    checkpoint = load_checkpoint(MODEL_PATH)
    class_names = checkpoint.get("class_names", CLASSES)
    image_size = int(checkpoint.get("image_size", 224))
    checkpoint_args = checkpoint.get("args", {})
    dropout = float(checkpoint_args.get("dropout", 0.35))

    network = build_resnet50(num_classes=len(class_names), dropout=dropout)
    network.load_state_dict(checkpoint["model_state_dict"])
    network.to(device)
    network.eval()
    model = network
    print(f"Loaded ResNet50 checkpoint: {MODEL_PATH}")
    print(f"Inference device: {device}")


def make_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: letterbox_image(image, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def letterbox_image(image: Image.Image, size: int, background: str = "white") -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), background)
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def run_inference(img: Image.Image) -> dict:
    if model is None:
        return {"error": f"Model not loaded. Expected checkpoint at {MODEL_PATH}"}

    image_tensor = make_preprocess()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image_tensor)[0]
        probabilities = torch.softmax(logits, dim=0).detach().cpu().tolist()

    return {class_names[index]: float(probabilities[index]) for index in range(len(class_names))}


def predict_gradio(image_path):
    try:
        img = Image.open(image_path)
        return run_inference(img)
    except Exception as exc:
        return {"error": str(exc)}


load_model()

with gr.Blocks() as blocks:
    image_input = gr.Image(type="filepath", visible=False)
    label_output = gr.JSON(visible=False)
    btn = gr.Button(visible=False)
    btn.click(fn=predict_gradio, inputs=image_input, outputs=label_output, api_name="predict_body_type")


app = FastAPI(title="AutoVision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        result = run_inference(img)
        if "error" in result:
            return JSONResponse(content=result, status_code=500)
        return JSONResponse(content={"predictions": result})
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=500)


app = gr.mount_gradio_app(app, blocks, path="/gradio")

frontend_dist = PROJECT_ROOT / "frontend" / "dist"
assets_path = frontend_dist / "assets"

if assets_path.exists():
    app.mount("/assets", StaticFiles(directory=assets_path), name="assets")


@app.get("/{full_path:path}")
def serve_index(full_path: str):
    index_path = frontend_dist / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "AutoVision API is running. Frontend not built yet."}


if __name__ == "__main__":
    import uvicorn

    print("Starting AutoVision API on http://localhost:7860")
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
