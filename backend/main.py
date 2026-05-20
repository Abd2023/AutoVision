"""
Phase 5: Backend API (Headless Gradio)
======================================
Serves the ONNX model via FastAPI with a direct upload endpoint.
The Gradio Blocks app is also mounted for demonstration/compatibility,
but the primary frontend uses the /api/predict endpoint via fetch().

Usage:
    cd backend
    python main.py
"""

import os
import io
import numpy as np
from PIL import Image
import onnxruntime as ort
import gradio as gr
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

CLASSES = ["F1", "HATCHBACK", "MICRO", "PICK_UP", "SEDAN", "STATION_WAGON", "SUV", "VAN"]

# Load ONNX model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model_int8.onnx")
if os.path.exists(MODEL_PATH):
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    print(f"ONNX model loaded from {MODEL_PATH}")
else:
    print(f"Warning: ONNX model not found at {MODEL_PATH}")
    session = None
    input_name = None


def run_inference(img: Image.Image) -> dict:
    """Run ONNX inference on a PIL Image. Returns class→probability dict."""
    if not session:
        return {"error": "Model not loaded on server."}

    img = img.convert("RGB").resize((224, 224))

    # Normalize
    img_arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_arr = (img_arr - mean) / std

    # HWC → CHW, add batch dim
    img_arr = np.transpose(img_arr, (2, 0, 1))
    img_arr = np.expand_dims(img_arr, axis=0)

    # Inference
    logits = session.run(None, {input_name: img_arr})[0][0]

    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}


# Gradio wrapper (accepts filepath from Gradio component)
def predict_gradio(image_path):
    try:
        img = Image.open(image_path)
        return run_inference(img)
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------
# Gradio Blocks (headless – no UI, only API)
# ---------------------------------------------------------
with gr.Blocks() as blocks:
    image_input = gr.Image(type="filepath", visible=False)
    label_output = gr.JSON(visible=False)
    btn = gr.Button(visible=False)
    btn.click(fn=predict_gradio, inputs=image_input, outputs=label_output,
              api_name="predict_body_type")


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------
app = FastAPI(title="AutoVision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Direct REST endpoint (used by the React frontend) ────
@app.post("/api/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """Accept an uploaded image and return class probabilities."""
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        result = run_inference(img)
        if "error" in result:
            return JSONResponse(content=result, status_code=500)
        return JSONResponse(content={"predictions": result})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ── Mount Gradio at /gradio (kept for compatibility) ──────
app = gr.mount_gradio_app(app, blocks, path="/gradio")


# ---------------------------------------------------------
# Unified Routing for production frontend
# ---------------------------------------------------------
frontend_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
assets_path = os.path.join(frontend_dist, "assets")

if os.path.exists(assets_path):
    app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

@app.get("/{full_path:path}")
def serve_index(full_path: str):
    index_path = os.path.join(frontend_dist, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "AutoVision API is running. Frontend not built yet."}


if __name__ == "__main__":
    import uvicorn
    print("Starting AutoVision API on http://localhost:7860")
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
