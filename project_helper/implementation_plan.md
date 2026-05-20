# Araba Gövde Tipi Sınıflandırma - Implementation Plan

This document serves as a detailed, step-by-step implementation plan for AI coding agents to execute the Car Body Type Classification project. The goal is to build an 8-class image classifier, optimize it to run lightning-fast on CPUs, and deploy it with a modern React + Headless Gradio architecture, all while strictly keeping the final deployment package **under 95 MB**.

## Phase 1: Project Setup & Dependency Management
**Goal:** Establish a clean separation between the heavy training environment and the lightweight deployment environment.

1.  **Directory Structure:**
    Create the following structured repository:
    ```text
    project_root/
    ├── data/                  # Ignored in final delivery
    │   ├── raw/
    │   └── processed/
    ├── notebooks/             # EDA, Training scripts, Export scripts
    ├── backend/               # FastAPI + Headless Gradio
    │   ├── model/             # Contains the INT8 ONNX file
    │   └── main.py
    ├── frontend/              # Vite + React App
    ├── requirements-train.txt # Heavy (PyTorch, timm, rembg, matplotlib, scikit-learn)
    └── requirements.txt       # Ultra-light (onnxruntime, fastapi, uvicorn, gradio, python-multipart, Pillow, numpy)
    ```
2.  **Initialize Frontend:**
    Navigate to `frontend/` and initialize a Vite React application (`npm create vite@latest . -- --template react`). Install `@gradio/client`, `tailwindcss`, and a charting library (e.g., `recharts`).

## Phase 2: Data Acquisition & Preprocessing (Crucial for Generalization)
**Goal:** Gather a perfectly balanced dataset of 8 classes and eliminate domain bias.

1.  **Sourcing the 8 Classes:**
    *   **Base Classes (SUV, VAN, STATION WAGON, SEDAN, HATCHBACK, PICK UP):** Source from standard Kaggle datasets (e.g., "Stanford Cars" or "Cars Body Type Cropped").
    *   **F1 (Açık Tekerlekli):** Download from Kaggle `f1-image-classification-updated`.
    *   **Micro:** Download from HuggingFace `DamianBoborzi/car_images` (already preprocessed) or filter `Unit293/car_models_3887`.
2.  **Background Subtraction (Offline):**
    Write a Python script in `notebooks/` using `rembg` (which utilizes the 4.7 MB `U2NetP` model) to strip the background from all raw images. Replace the backgrounds with a solid color (e.g., gray) or randomized Perlin noise. Save to `data/processed/`.
3.  **Balancing:**
    Ensure exactly equal numbers of images per class (e.g., 1500 training, 300 validation per class).

## Phase 3: Model Architecture & Training Strategy
**Goal:** Train a highly accurate but ultra-lightweight model using advanced loss formulations to prevent fine-grained confusion.

1.  **Model Selection:**
    In a training script (`notebooks/train.py`), instantiate a pre-trained **MobileNetV4-Conv** or **FastViT-S12** via the `timm` library. Change the final classification head to output 8 classes.
2.  **Advanced Loss Implementation:**
    *   Implement **Focal Loss** to force the model to focus on difficult edge cases (like Hatchback vs. Micro).
    *   *(Optional but Recommended)* Implement **Knowledge-Distillation-Based Label Smoothing** instead of standard cross-entropy to handle semantic similarities (Station Wagon vs. SUV).
3.  **Data Augmentation:**
    Apply heavy augmentation during training: `RandomResizedCrop(224)`, `RandomHorizontalFlip`, `ColorJitter`, and `RandomAffine`.
4.  **Training Loop:**
    Train the model using Early Stopping. Save the best weights (`best_model.pt`).
5.  **Evaluation Metrics:**
    Generate the required evaluation artifacts: Training/Validation Loss Curve, Training/Validation Accuracy Curve, and the 8x8 Normalized Confusion Matrix Heatmap. Save these for the final IEEE report.

## Phase 4: Model Optimization & Export (The 95MB Solution)
**Goal:** Strip away PyTorch and shrink the model for fast CPU inference.

1.  **ONNX Export:**
    Write a script (`notebooks/export.py`) to load `best_model.pt` and export it to `model_fp32.onnx` using `torch.onnx.export` with a dummy input tensor of shape `(1, 3, 224, 224)`.
2.  **Dynamic Quantization:**
    Using `onnxruntime.quantization`, apply **Dynamic Quantization** (INT8) to the exported model.
    *   *Critical:* You MUST set `per_channel=True` when quantizing MobileNet/FastViT models to prevent mathematical underflow/overflow in depthwise convolutions.
    *   Save the resulting `model_int8.onnx` into the `backend/model/` directory. Verify the file size is ~7-10 MB.

## Phase 5: Backend API (Headless Gradio)
**Goal:** Serve the ONNX model efficiently without bundling heavy frontend libraries.

1.  **FastAPI Setup:**
    In `backend/main.py`, import `ort` (onnxruntime) and initialize the Inference Session with `model_int8.onnx`.
2.  **Inference Function:**
    Write a `predict(image_path)` function that uses `PIL` and `numpy` to resize (224x224), normalize (divide by 255.0), and transpose (HWC to CHW) the image, then runs `session.run()` to get softmax probabilities mapped to the 8 string labels.
3.  **Headless Server:**
    Instantiate `app = gradio.Server()`. Register the inference function via the `@app.api(name="predict_body_type")` decorator.
4.  **Unified Routing:**
    Use FastAPI's `StaticFiles` to mount `../frontend/dist/assets`. Create a catch-all route `/{full_path:path}` that serves `../frontend/dist/index.html`. Launch via `app.launch(server_port=7860)`.

## Phase 6: Frontend Development (React SPA)
**Goal:** Build a beautiful, responsive UI that communicates with the headless backend.

1.  **UI Components:**
    Build a Drag-and-Drop image uploader. Once an image is uploaded, display a preview.
2.  **API Connection:**
    On "Predict" button click, use `@gradio/client`:
    ```javascript
    const app = await Client.connect("http://localhost:7860");
    const result = await app.predict("/predict_body_type", { image_path: handle_file(imageFile) });
    ```
3.  **Result Visualization:**
    Display the predicted class text prominently. Feed the probability dictionary into a `recharts` BarChart to visualize the distribution across all 8 classes.

## Phase 7: Final Deployment Packaging
**Goal:** Prepare the final <95MB ZIP file.

1.  **Build Frontend:** Run `npm run build` inside `frontend/`. This creates the optimized `dist` folder.
2.  **Clean Up:** Delete `data/`, `notebooks/`, `frontend/node_modules`, `frontend/src`, `frontend/public`, and any `.pt` files.
3.  **Deliverable:** Zip the remaining `backend/` (with `.onnx` model), `frontend/dist/`, and `requirements.txt`. The entire package will be ~20-30 MB, safely passing the 95 MB limit, and ready for deployment.
