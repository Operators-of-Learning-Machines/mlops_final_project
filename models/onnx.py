
import torch
import onnxruntime as ort
import numpy as np
from sclera_identity_classification.architectures.squeezenet import SqueezeNet

import onnxruntime as ort

onnx_session = None

def load_onnx_session():
    global onnx_session
    if onnx_session is None:
        onnx_session = ort.InferenceSession(
            "models/model.onnx",
            providers=["CPUExecutionProvider"]
        )

def export_to_onnx(pth_path: str, onnx_path: str):
    device = "cpu"  # ONNX export should be CPU

    net = SqueezeNet(
        transfer_learning_model_path=pth_path,
        out_channels=220
    ).to(device)

    net.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18
    )
