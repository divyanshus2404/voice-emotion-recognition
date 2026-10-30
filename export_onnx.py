"""Export the emotion model to ONNX so it can run in the browser (web/ folder).

    python export_onnx.py

Writes web/model.onnx (8-bit, ~3x smaller) and web/labels.json,
then checks the ONNX output matches PyTorch.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from emotion import EmotionPredictor

WEB = Path(__file__).parent / "web"


def main():
    predictor = EmotionPredictor()
    model = predictor.model


    class LogitsOnly(torch.nn.Module):
        """Same maths as the model's forward(), written out so ONNX export can trace it."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            m = self.m
            hidden = m.wav2vec2(x, output_hidden_states=True, return_dict=True).hidden_states
            if m.config.use_weighted_layer_sum:  # learned mix of all transformer layers
                weights = torch.softmax(m.layer_weights, dim=-1)
                h = (torch.stack(hidden, dim=1) * weights.view(1, -1, 1, 1)).sum(dim=1)
            else:
                h = hidden[-1]
            return m.classifier(m.projector(h).mean(dim=1))

    fp32 = WEB / "model_fp32.onnx"

    dummy = torch.randn(1, 32_000)
    torch.onnx.export(
        LogitsOnly(model), (dummy,), fp32, input_names=["input_values"], output_names=["logits"],
        dynamic_axes={"input_values": {0: "batch", 1: "samples"}}, opset_version=17, dynamo=False,
    )
    # Only quantize the transformer matrix multiplies, per channel: quantizing the conv layers
    # too changed the predictions noticeably.
    quantize_dynamic(fp32, WEB / "model.onnx", op_types_to_quantize=["MatMul"],
                     weight_type=QuantType.QInt8, per_channel=True)
    fp32.unlink()
    (WEB / "labels.json").write_text(json.dumps(predictor.labels))

    # Same audio through PyTorch and ONNX: the predictions should agree.
    audio = np.random.default_rng(0).standard_normal(48_000).astype(np.float32) * 0.1
    inputs = predictor.extractor(audio, sampling_rate=16_000, return_tensors="np")["input_values"]
    with torch.no_grad():
        torch_probs = torch.softmax(model(torch.from_numpy(inputs)).logits, -1).numpy()
    logits = ort.InferenceSession(str(WEB / "model.onnx")).run(None, {"input_values": inputs})[0]
    onnx_probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    print("pytorch:", torch_probs.round(3), "\nonnx:   ", onnx_probs.round(3))
    print(f"size: {(WEB / 'model.onnx').stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
