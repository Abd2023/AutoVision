"""
Export an AutoVision PyTorch checkpoint to ONNX.

FP32 export is the default. INT8 quantization is intentionally opt-in because
dynamic quantization damaged the previous FastViT model's accuracy.

Examples:
    python notebooks/export.py --checkpoint notebooks/outputs_accuracy/best_model.pt --model convnext_base.fb_in22k_ft_in1k
    python notebooks/export.py --checkpoint notebooks/outputs/best_model.pt --model fastvit_s12 --output backend/model/model_fp32.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import timm
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained AutoVision model to ONNX.")
    parser.add_argument("--checkpoint", default="notebooks/outputs/best_model.pt")
    parser.add_argument("--model", default="fastvit_s12")
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--output", default="backend/model/model_fp32.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--quantize", action="store_true", help="Also create a dynamic INT8 ONNX copy.")
    parser.add_argument("--quantized-output", default="backend/model/model_int8.onnx")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    model = timm.create_model(args.model, pretrained=False, num_classes=args.num_classes)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    dummy_input = torch.randn(1, 3, args.img_size, args.img_size)
    print(f"Exporting FP32 ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"FP32 ONNX size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")

    if args.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantized_path = Path(args.quantized_output)
        quantized_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Creating dynamic INT8 ONNX: {quantized_path}")
        quantize_dynamic(
            model_input=output_path,
            model_output=quantized_path,
            weight_type=QuantType.QUInt8,
            per_channel=True,
        )
        print(f"INT8 ONNX size: {quantized_path.stat().st_size / (1024 * 1024):.2f} MB")
        print("Validate this INT8 file before using it in the backend.")


if __name__ == "__main__":
    main()
