# CLIP Implementation Plan for Axera AX650N (Optimized Base Version)

This document provides the step-by-step instructions to export, compile, and deploy the **CLIP Base (ViT-B/32)** model. This version is optimized for high-speed inference on the AX650N and fast compilation on resource-constrained host machines.

## Overview
We use `openai/clip-vit-base-patch32`. While it requires a new Text Encoder (512-dim), the total system footprint and latency are significantly lower than the Large version.

For a detailed analysis of how this implementation supports **Constrained Output** (e.g., Demographics, Structured Data) and handles large vocabularies, see [clip_constrained_output.md](clip_constrained_output.md).

---

## Phase 1: Export to ONNX (Raspberry Pi or Local PC) [COMPLETED]

### 1. Create Export Script (Done)
Create `img2txt/export_clip_base.py`:
```python
import torch
from transformers import CLIPModel, CLIPProcessor
import os

model_id = "openai/clip-vit-base-patch32"
os.makedirs("img2txt/onnx-models", exist_ok=True)

print(f"Loading {model_id}...")
# Use eager attention and safetensors to avoid export/security issues
model = CLIPModel.from_pretrained(model_id, use_safetensors=True, attn_implementation="eager")
model.eval()

# 1. Export Vision Encoder
dummy_vision_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    torch.onnx.export(
        model.vision_model,
        dummy_vision_input,
        "img2txt/onnx-models/clip_base_vision.onnx",
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        opset_version=14 # Opset 14 is more stable for Axera toolchain
    )

# 2. Export Text Encoder
dummy_text_input = torch.ones(1, 77, dtype=torch.long)
with torch.no_grad():
    torch.onnx.export(
        model.text_model,
        dummy_text_input,
        "img2txt/onnx-models/clip_base_text.onnx",
        input_names=["input_ids"],
        output_names=["text_embeds"],
        opset_version=14
    )
print("Done! Saved Vision and Text models to img2txt/onnx-models/")
```

### 2. Generate Calibration Data (Done)
Create `img2txt/prepare_clip_calib.py`:
```python
import numpy as np
import os
import tarfile
from PIL import Image
from transformers import CLIPProcessor

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
calib_dir = "img2txt/datasets/clip_calib"
text_calib_dir = "img2txt/datasets/text_calib"
os.makedirs(calib_dir, exist_ok=True)
os.makedirs(text_calib_dir, exist_ok=True)

# Use assets for calibration
image_paths = [os.path.join("assets", f) for f in os.listdir("assets") if f.endswith(('.png', '.jpg'))][:50]

for i, path in enumerate(image_paths):
    img = Image.open(path).convert("RGB")
    inputs = processor(images=img, return_tensors="np")
    pixel_values = inputs['pixel_values'].astype(np.float32)
    np.save(os.path.join(calib_dir, f"input_{i}.npy"), pixel_values)

# Generate dummy text calibration (77 tokens, vocab size 49408)
# Use int64 to match the ONNX model's expected input type
for i in range(50):
    text_input = np.random.randint(0, 49408, (1, 77)).astype(np.int64)
    np.save(os.path.join(text_calib_dir, f"input_{i}.npy"), text_input)

with tarfile.open("img2txt/datasets/clip_calib.tar", "w") as tar:
    tar.add(calib_dir, arcname=".")

with tarfile.open("img2txt/datasets/text_calib.tar", "w") as tar:
    tar.add(text_calib_dir, arcname=".")
```

### 3. Create Pulsar2 Config Files (Done)
Create `img2txt/configs/clip_vision_u16.json`:
```json
{
    "model_type": "ONNX",
    "npu_mode": "NPU3",
    "quant": {
      "input_configs": [
        {
          "tensor_name": "pixel_values",
          "calibration_dataset": "./datasets/clip_calib.tar",
          "calibration_size": 50,
          "calibration_format": "Numpy"
        }
      ],
      "calibration_method": "MinMax",
      "precision_analysis": false,
      "conv_bias_data_type": "FP32",
      "layer_configs": [
        {
          "start_tensor_names": ["DEFAULT"],
          "end_tensor_names": ["DEFAULT"],
          "data_type": "U16"
        }
      ]
    },
    "compiler": {
      "npu_perf": false
    }
}
```

Create `img2txt/configs/text_encoder_u16.json`:
```json
{
    "model_type": "ONNX",
    "npu_mode": "NPU3",
    "quant": {
      "input_configs": [
        {
          "tensor_name": "input_ids",
          "calibration_dataset": "./datasets/text_calib.tar",
          "calibration_size": 50,
          "calibration_format": "Numpy"
        }
      ],
      "calibration_method": "MinMax",
      "precision_analysis": false,
      "conv_bias_data_type": "FP32",
      "layer_configs": [
        {
          "start_tensor_names": ["DEFAULT"],
          "end_tensor_names": ["DEFAULT"],
          "data_type": "U16"
        }
      ]
    },
    "input_processors": [
      {
        "tensor_name": "input_ids",
        "src_dtype": "S32"
      }
    ],
    "compiler": {
      "npu_perf": false
    }
}
```

---

## Phase 2: Compilation (Intel Mac) [COMPLETED]

### 1. Transfer Files (Done)
```bash
# Create directory structure on Mac
ssh gregm2@192.168.4.130 "mkdir -p ~/pulsar_build/img2txt/{onnx-models,datasets,configs,axmodels}"

# Transfer ONNX models
scp img2txt/onnx-models/clip_base_*.onnx* gregm2@192.168.4.130:~/pulsar_build/img2txt/onnx-models/

# Transfer calibration data
scp img2txt/datasets/*.tar gregm2@192.168.4.130:~/pulsar_build/img2txt/datasets/

# Transfer config files
scp img2txt/configs/*.json gregm2@192.168.4.130:~/pulsar_build/img2txt/configs/
```

### 2. Run Build (Fast) (Done)
On the Mac, navigate to the img2txt directory and run:
```bash
cd ~/pulsar_build/img2txt

# Vision Encoder (U16 quantization)
docker run --rm -v "$(pwd)":/data -w /data pulsar2:5.1 \
    pulsar2 build --input onnx-models/clip_base_vision.onnx \
    --output_dir axmodels --output_name clip_base_vision.axmodel \
    --config configs/clip_vision_u16.json

# Text Encoder (U16 quantization)
docker run --rm -v "$(pwd)":/data -w /data pulsar2:5.1 \
    pulsar2 build --input onnx-models/clip_base_text.onnx \
    --output_dir axmodels --output_name clip_base_text.axmodel \
    --config configs/text_encoder_u16.json
```

### 3. Transfer Models Back to Raspberry Pi (Done)
Run this from your **Mac**:
```bash
scp axmodels/clip_base_*.axmodel gregm@192.168.4.121:/home/gregm/sd1.5-lcm.axera_greg/models/
```

---

## Phase 3: Inference Wrapper (Raspberry Pi) [COMPLETED]

Update `img2txt/clip_interrogate.py` (Done):
```python
import axengine
import numpy as np
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VISION_MODEL = os.path.join(SCRIPT_DIR, "../models/clip_base_vision.axmodel")
TEXT_MODEL = os.path.join(SCRIPT_DIR, "../models/clip_base_text.axmodel")
TOKENIZER_DIR = "openai/clip-vit-base-patch32" # Use standard HF tokenizer

class CLIPInterrogator:
    def __init__(self):
        self.vision_session = axengine.InferenceSession(VISION_MODEL)
        self.text_session = axengine.InferenceSession(TEXT_MODEL)
        
        # Load CLIP Model for projections and tokenizer
        self.model = CLIPModel.from_pretrained(TOKENIZER_DIR)
        self.processor = CLIPProcessor.from_pretrained(TOKENIZER_DIR)
        self.tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_DIR)
        
        # Extract projection weights for CPU application (since they weren't in the ONNX export)
        self.visual_projection = self.model.visual_projection.weight.detach().numpy().T
        self.text_projection = self.model.text_projection.weight.detach().numpy().T

    def interrogate(self, image, labels):
        # 1. Image Pass
        inputs = self.processor(images=image, return_tensors="np")
        # The model returns [last_hidden_state, pooler_output]
        v_out = self.vision_session.run(None, {"pixel_values": inputs['pixel_values'].astype(np.float32)})
        img_emb_raw = v_out[1] # Shape (1, 768)
        
        # Apply visual projection
        img_emb = img_emb_raw @ self.visual_projection # Shape (1, 512)
        
        # 2. Text Pass (Loop because model is static batch size 1)
        text_embs = []
        for label in labels:
            text_inputs = self.tokenizer(label, padding="max_length", max_length=77, truncation=True, return_tensors="np")
            t_out = self.text_session.run(None, {"input_ids": text_inputs['input_ids'].astype(np.int32)})
            t_emb_raw = t_out[1] # Shape (1, 512)
            
            # Apply text projection
            t_emb = t_emb_raw @ self.text_projection # Shape (1, 512)
            text_embs.append(t_emb)
        
        text_embs = np.concatenate(text_embs, axis=0)
        
        # 3. Similarity (CPU)
        img_emb /= np.linalg.norm(img_emb, axis=-1, keepdims=True)
        text_embs /= np.linalg.norm(text_embs, axis=-1, keepdims=True)
        return img_emb @ text_embs.T
```

### 2. Standalone Interrogation Script (Done)
Create `img2txt/run_clip_interrogate.py` to provide a command-line interface for tagging images using a curated list of tags.

---

## Phase 4: Service Integration (A1111 Compatibility)

The primary goal is to provide a drop-in replacement for the Automatic1111 interrogation API. This allows existing tools to use the Axera NPU without modification.

### 1. A1111 Standard Endpoint
Add the following to `pi_axera_sd_generator.py`. This matches the [A1111 API Spec](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API).

**Endpoint**: `POST /sdapi/v1/interrogate`

**Request Body**:
```json
{
  "image": "base64...",
  "model": "clip"
}
```

**Response Body**:
```json
{
  "caption": "a red fox, snow, highly detailed, outdoor"
}
```

### 2. Implementation Code
```python
@app.route("/sdapi/v1/interrogate", methods=["POST"])
def sdapi_interrogate():
    data = request.get_json(force=True)
    image_b64 = data.get("image")
    
    if not image_b64:
        return jsonify({"error": "No image provided"}), 400
    
    # Decode image
    image_data = base64.b64decode(image_b64.split(",")[-1])
    image = Image.open(BytesIO(image_data)).convert("RGB")
    
    # Run interrogation
    # Note: In a production A1111 setup, this would use a massive tag list.
    # For the Pi, we use a curated list of high-value tags.
    tags = ["man", "woman", "outdoor", "indoor", "night", "day", "highly detailed"]
    scores = clip_interrogator.interrogate(image, tags)
    
    # Filter by threshold (0.25 is a good starting point for CLIP Base)
    result_tags = [tags[i] for i, score in enumerate(scores[0]) if score > 0.25]
    
    return jsonify({"caption": ", ".join(result_tags)})
```

### 2. Memory Management
Since the NPU memory is shared, ensure `clip_interrogator` uses the same `axengine` context or is loaded/unloaded if RAM is tight. However, for the AX650N, keeping the Vision Encoder resident (~300MB) alongside SD1.5 is feasible.

---

## Lessons Learned from CLIP Conversion
1. **Static Batch Size**: The Axera compiler (Pulsar2) does not support dynamic axes (e.g., `s99`). Models must be exported with a fixed batch size (usually 1).
2. **Opset Version**: Use `opset_version=14`. Higher versions (like 17) can introduce operators (like `ScaledDotProductAttention`) that the compiler or ONNX export process may struggle with in certain PyTorch versions.
3. **Attention Implementation**: Use `attn_implementation="eager"` when loading models from Transformers to avoid issues with the newer `sdpa` (Scaled Dot Product Attention) during ONNX export.
4. **Data Types**: 
    - Ensure calibration `.npy` files for images are explicitly `float32`.
    - For Text Encoders, calibration data should be `int64` to match the model's input, but the Pulsar2 config `src_dtype` should be `S32` (signed 32-bit) for the NPU processor.
5. **Security/Safetensors**: Use `use_safetensors=True` to bypass security restrictions in newer PyTorch/Transformers versions when loading weights.
6. **NPU Locking**: Use the existing `unet_lock` or a new `vision_lock` to prevent concurrent NPU access if multiple requests arrive.
