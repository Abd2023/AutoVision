import os
import torch
import timm
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def export_to_onnx():
    print("Loading PyTorch model...")
    device = torch.device("cpu")
    model = timm.create_model("fastvit_s12", pretrained=False, num_classes=8)
    model.load_state_dict(torch.load("outputs/best_model.pt", map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    onnx_fp32_path = "outputs/model_fp32.onnx"
    
    print(f"Exporting to {onnx_fp32_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_fp32_path, 
        export_params=True, 
        opset_version=14, 
        do_constant_folding=True, 
        input_names=["input"], 
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print("Export complete.")
    
    onnx_int8_path = "../backend/model/model_int8.onnx"
    print(f"Quantizing model to {onnx_int8_path} with per_channel=True...")
    quantize_dynamic(
        model_input=onnx_fp32_path,
        model_output=onnx_int8_path,
        weight_type=QuantType.QUInt8,
        per_channel=True
    )
    
    print("Quantization complete.")
    print(f"ONNX INT8 model size: {os.path.getsize(onnx_int8_path) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    export_to_onnx()
