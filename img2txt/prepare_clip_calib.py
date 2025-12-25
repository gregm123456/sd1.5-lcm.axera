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

# Generate dummy text calibration (77 tokens)
for i in range(50):
    text_input = np.random.randint(0, 49408, (1, 77)).astype(np.int64)
    np.save(os.path.join(text_calib_dir, f"input_{i}.npy"), text_input)

with tarfile.open("img2txt/datasets/clip_calib.tar", "w") as tar:
    tar.add(calib_dir, arcname=".")

with tarfile.open("img2txt/datasets/text_calib.tar", "w") as tar:
    tar.add(text_calib_dir, arcname=".")
