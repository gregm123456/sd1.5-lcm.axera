import torch
from transformers import CLIPModel, CLIPProcessor
import os

model_id = "openai/clip-vit-base-patch32"
os.makedirs("img2txt/onnx-models", exist_ok=True)

print(f"Loading {model_id}...")
model = CLIPModel.from_pretrained(model_id)

# 1. Export Vision Encoder
dummy_vision_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model.vision_model,
    dummy_vision_input,
    "img2txt/onnx-models/clip_base_vision.onnx",
    input_names=["pixel_values"],
    output_names=["image_embeds"],
    dynamic_axes={"pixel_values": {0: "batch_size"}},
    opset_version=17
)

# 2. Export Text Encoder
dummy_text_input = torch.ones(1, 77, dtype=torch.long)
torch.onnx.export(
    model.text_model,
    dummy_text_input,
    "img2txt/onnx-models/clip_base_text.onnx",
    input_names=["input_ids"],
    output_names=["text_embeds"],
    dynamic_axes={"input_ids": {0: "batch_size"}},
    opset_version=17
)
print("Done! Saved Vision and Text models to img2txt/onnx-models/")
