from typing import List, Union
import numpy as np
import axengine
import torch
from PIL import Image
from transformers import CLIPTokenizer, PreTrainedTokenizer, CLIPProcessor, CLIPModel
import os
import time
import threading
import base64
import random
import sys
import hashlib
import json
from io import BytesIO
from flask import Flask, request, jsonify


def maybe_convert_prompt(prompt: Union[str, List[str]], tokenizer: "PreTrainedTokenizer"):  # noqa: F821
    if not isinstance(prompt, List):
        prompts = [prompt]
    else:
        prompts = prompt

    prompts = [_maybe_convert_prompt(p, tokenizer) for p in prompts]

    if not isinstance(prompt, List):
        return prompts[0]

    return prompts


def _maybe_convert_prompt(prompt: str, tokenizer: "PreTrainedTokenizer"):  # noqa: F821
    tokens = tokenizer.tokenize(prompt)
    unique_tokens = set(tokens)
    for token in unique_tokens:
        if token in tokenizer.added_tokens_encoder:
            replacement = token
            i = 1
            while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                replacement += f" {token}_{i}"
                i += 1

            prompt = prompt.replace(token, replacement)

    return prompt


def get_embeds(prompt, tokenizer, text_encoder):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    # ensure numpy int32
    text_encoder_out = text_encoder.run(None, {"input_ids": text_input_ids.to("cpu").numpy().astype(np.int32)})[0]
    return text_encoder_out


def get_alphas_cumprod():
    betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float32) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).detach().numpy()
    final_alphas_cumprod = alphas_cumprod[0]
    self_timesteps = np.arange(0, 1000)[::-1].copy().astype(np.int64)
    return alphas_cumprod, final_alphas_cumprod, self_timesteps


# Curated list of tags for general image interrogation
DEFAULT_TAGS = [
    "man", "woman", "boy", "girl", "person", "group of people",
    "outdoor", "indoor", "nature", "city", "landscape", "interior",
    "day", "night", "sunset", "sunrise",
    "highly detailed", "masterpiece", "sharp focus", "blurry",
    "digital art", "photograph", "painting", "sketch", "anime",
    "forest", "mountain", "beach", "ocean", "street", "building",
    "cat", "dog", "horse", "bird", "animal",
    "car", "bicycle", "plane", "boat",
    "food", "drink", "fruit", "vegetable",
    "blue sky", "cloudy", "rainy", "snowy", "sunny",
    "red", "blue", "green", "yellow", "black", "white", "colorful"
]


class CLIPInterrogator:
    def __init__(self, vision_model_path, text_model_path, cache_dir=None):
        print(f"Loading CLIP Vision Model: {vision_model_path}")
        self.vision_session = axengine.InferenceSession(vision_model_path)
        print(f"Loading CLIP Text Model: {text_model_path}")
        self.text_session = axengine.InferenceSession(text_model_path)
        
        tokenizer_dir = "openai/clip-vit-base-patch32"
        print(f"Loading CLIP Model for projections and tokenizer: {tokenizer_dir}")
        self.model = CLIPModel.from_pretrained(tokenizer_dir)
        self.processor = CLIPProcessor.from_pretrained(tokenizer_dir)
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_dir)
        
        # Extract projection weights for CPU application
        self.visual_projection = self.model.visual_projection.weight.detach().numpy().T
        self.text_projection = self.model.text_projection.weight.detach().numpy().T
        
        # We can delete the model to save RAM if needed
        del self.model
        
        self.cached_embeddings = None
        self.cached_labels = None
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def precompute_text_embeddings(self, labels):
        # Generate a hash of the labels and the template to use as a cache key
        template = "a photo of a {}"
        labels_str = ",".join(labels) + "|" + template
        labels_hash = hashlib.sha256(labels_str.encode()).hexdigest()
        cache_file = None
        
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"clip_cache_{labels_hash}.npy")
            if os.path.exists(cache_file):
                print(f"Loading cached embeddings from {cache_file}...")
                self.cached_embeddings = np.load(cache_file)
                self.cached_labels = labels
                return

        print(f"Pre-computing embeddings for {len(labels)} labels...")
        text_embs = []
        start_time = time.time()
        
        for i, label in enumerate(labels):
            if i % 100 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = (len(labels) - i) / rate
                print(f"  Processed {i}/{len(labels)} labels ({rate:.1f} labels/sec, ~{remaining:.0f}s remaining)")
            
            # Use a template to provide context (e.g., "a photo of a dog")
            templated_label = template.format(label)
            text_inputs = self.tokenizer(templated_label, padding="max_length", max_length=77, truncation=True, return_tensors="np")
            t_out = self.text_session.run(None, {"input_ids": text_inputs['input_ids'].astype(np.int32)})
            t_emb_raw = t_out[1] # Shape (1, 512)
            
            # Apply text projection
            t_emb = t_emb_raw @ self.text_projection # Shape (1, 512)
            text_embs.append(t_emb)
        
        self.cached_embeddings = np.concatenate(text_embs, axis=0)
        # Normalize
        self.cached_embeddings /= np.linalg.norm(self.cached_embeddings, axis=-1, keepdims=True)
        self.cached_labels = labels
        
        if cache_file:
            print(f"Saving embeddings to cache: {cache_file}")
            np.save(cache_file, self.cached_embeddings)
            
        print(f"Embeddings pre-computed for {len(labels)} labels in {time.time() - start_time:.2f}s.")

    def interrogate(self, image, labels=None):
        # 1. Image Pass
        inputs = self.processor(images=image, return_tensors="np")
        # The model returns [last_hidden_state, pooler_output]
        # We want pooler_output (index 1)
        v_out = self.vision_session.run(None, {"pixel_values": inputs['pixel_values'].astype(np.float32)})
        img_emb_raw = v_out[1] # Shape (1, 768)
        
        # Apply visual projection
        img_emb = img_emb_raw @ self.visual_projection # Shape (1, 512)
        img_emb /= np.linalg.norm(img_emb, axis=-1, keepdims=True)
        
        # 2. Text Pass
        if labels is None:
            if self.cached_embeddings is None:
                raise ValueError("No labels provided and no cached embeddings found.")
            text_embs_norm = self.cached_embeddings
            current_labels = self.cached_labels
        elif labels == self.cached_labels:
            text_embs_norm = self.cached_embeddings
            current_labels = self.cached_labels
        else:
            # Fallback to on-the-fly computation for custom labels
            text_embs = []
            for label in labels:
                templated_label = f"a photo of a {label}"
                text_inputs = self.tokenizer(templated_label, padding="max_length", max_length=77, truncation=True, return_tensors="np")
                t_out = self.text_session.run(None, {"input_ids": text_inputs['input_ids'].astype(np.int32)})
                t_emb_raw = t_out[1]
                t_emb = t_emb_raw @ self.text_projection
                text_embs.append(t_emb)
            
            text_embs = np.concatenate(text_embs, axis=0)
            text_embs_norm = text_embs / np.linalg.norm(text_embs, axis=-1, keepdims=True)
            current_labels = labels
        
        # 3. Similarity (CPU)
        scores = img_emb @ text_embs_norm.T
        return scores, current_labels

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def classify_categories(self, image, schema):
        # 1. Get image embedding
        inputs = self.processor(images=image, return_tensors="np")
        v_out = self.vision_session.run(None, {"pixel_values": inputs['pixel_values'].astype(np.float32)})
        img_emb_raw = v_out[1]
        img_emb = img_emb_raw @ self.visual_projection
        img_emb /= np.linalg.norm(img_emb, axis=-1, keepdims=True)

        results = {}
        
        # Create a mapping for fast lookup if we have cached labels
        label_to_idx = {}
        if self.cached_labels:
            label_to_idx = {label: i for i, label in enumerate(self.cached_labels)}

        for category, labels in schema.items():
            cat_embs = []
            valid_labels = []
            
            for label in labels:
                if label in label_to_idx:
                    cat_embs.append(self.cached_embeddings[label_to_idx[label]])
                    valid_labels.append(label)
                else:
                    # On-the-fly encoding for missing labels
                    templated_label = f"a photo of a {label}"
                    text_inputs = self.tokenizer(templated_label, padding="max_length", max_length=77, truncation=True, return_tensors="np")
                    t_out = self.text_session.run(None, {"input_ids": text_inputs['input_ids'].astype(np.int32)})
                    t_emb_raw = t_out[1]
                    t_emb = t_emb_raw @ self.text_projection
                    t_emb_norm = t_emb / np.linalg.norm(t_emb, axis=-1, keepdims=True)
                    cat_embs.append(t_emb_norm.flatten())
                    valid_labels.append(label)
            
            if not cat_embs:
                continue
                
            cat_embs = np.stack(cat_embs)
            # Similarity
            scores = (img_emb @ cat_embs.T).flatten()
            
            # Softmax with temperature scaling (CLIP standard is 100.0)
            probs = self._softmax(scores * 100.0)
            
            winner_idx = np.argmax(probs)
            results[category] = {
                "winner": valid_labels[winner_idx],
                "confidence": float(probs[winner_idx])
            }
            
        return results, img_emb


app = Flask(__name__)


def log_request(endpoint, data):
    """Log request parameters, redacting large base64 strings."""
    safe_data = data.copy()
    for key in ["init_image", "init_images", "image"]:
        if key in safe_data:
            if isinstance(safe_data[key], list):
                safe_data[key] = [f"<base64 data, {len(str(item))} chars>" for item in safe_data[key]]
            else:
                safe_data[key] = f"<base64 data, {len(str(safe_data[key]))} chars>"
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Request to {endpoint}:", flush=True)
    print(json.dumps(safe_data, indent=2), flush=True)


# Configuration (adjust paths as needed)
TEXT_MODEL_DIR = os.environ.get("TEXT_MODEL_DIR", "./models/")
TEXT_ENCODER_MODEL = os.environ.get("TEXT_ENCODER_MODEL", os.path.join(TEXT_MODEL_DIR, "text_encoder", "sd15_text_encoder_sim.axmodel"))
UNET_MODEL = os.environ.get("UNET_MODEL", "./models/unet.axmodel")
VAE_DECODER_MODEL = os.environ.get("VAE_DECODER_MODEL", "./models/vae_decoder.axmodel")
VAE_ENCODER_MODEL = os.environ.get("VAE_ENCODER_MODEL", "./models/vae_encoder.axmodel")
CLIP_VISION_MODEL = os.environ.get("CLIP_VISION_MODEL", os.path.join(TEXT_MODEL_DIR, "clip_base_vision.axmodel"))
CLIP_TEXT_MODEL = os.environ.get("CLIP_TEXT_MODEL", os.path.join(TEXT_MODEL_DIR, "clip_base_text.axmodel"))
TIME_INPUT_TXT2IMG = os.environ.get("TIME_INPUT_TXT2IMG", "./models/time_input_txt2img.npy")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./txt2img_server_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load tokenizer and models once
print("Loading tokenizer and models (this may take a moment)...")
tokenizer = CLIPTokenizer.from_pretrained(os.path.join(TEXT_MODEL_DIR, "tokenizer"))
text_encoder = axengine.InferenceSession(TEXT_ENCODER_MODEL)
unet_session_main = axengine.InferenceSession(UNET_MODEL)
vae_decoder = axengine.InferenceSession(VAE_DECODER_MODEL)
vae_encoder = axengine.InferenceSession(VAE_ENCODER_MODEL)

# Initialize CLIP Interrogator
clip_interrogator = None
interrogate_tags = DEFAULT_TAGS

# Try to load custom tags
TAGS_FILE = os.environ.get("INTERROGATE_TAGS_FILE", "pi_axera_sd_service/imagenet_21k_labels_clean.txt")
if os.path.exists(TAGS_FILE):
    print(f"Loading interrogation tags from {TAGS_FILE}...")
    try:
        with open(TAGS_FILE, "r") as f:
            loaded_tags = [line.strip() for line in f if line.strip()]
        if loaded_tags:
            interrogate_tags = loaded_tags
            print(f"Loaded {len(interrogate_tags)} tags.")
    except Exception as e:
        print(f"Error loading tags file: {e}")

if os.path.exists(CLIP_VISION_MODEL) and os.path.exists(CLIP_TEXT_MODEL):
    try:
        cache_dir = os.path.join(TEXT_MODEL_DIR, "clip_cache")
        clip_interrogator = CLIPInterrogator(CLIP_VISION_MODEL, CLIP_TEXT_MODEL, cache_dir=cache_dir)
        # Pre-compute embeddings for the tags
        clip_interrogator.precompute_text_embeddings(interrogate_tags)
    except Exception as e:
        print(f"Failed to load CLIP Interrogator: {e}")
else:
    print("CLIP Interrogator models not found, /sdapi/v1/interrogate will be disabled.")

time_input_txt2img = np.load(TIME_INPUT_TXT2IMG)
alphas_cumprod, final_alphas_cumprod, self_timesteps = get_alphas_cumprod()

# Locks to avoid concurrent runs overlapping (safer for runtimes that are not thread-safe)
text_lock = threading.Lock()
unet_lock = threading.Lock()
vae_lock = threading.Lock()
clip_lock = threading.Lock()

# fixed timesteps used in the original example
DEFAULT_TIMESTEPS = np.array([999, 759, 499, 259]).astype(np.int64)


def encode_image_to_latent(image: Image.Image):
    """Encode PIL image to latent space for img2img"""
    # Convert to tensor and normalize (same as preprocess in reference)
    image = np.array(image).astype(np.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    # VAE encode - returns (1, 8, 64, 64) with mean and log_var
    with vae_lock:
        # some axengine VAE encoder models expect the input name 'image_sample'
        # instead of 'x' — provide both keys to be robust if model accepts either.
        feed = {"image_sample": image_tensor.numpy()}
        try:
            vae_encoder_out = vae_encoder.run(None, feed)[0]
        except Exception:
            # fallback to 'x' if encoder expects that
            vae_encoder_out = vae_encoder.run(None, {"x": image_tensor.numpy()})[0]
    
    # VAE encoder outputs (1, 8, 64, 64) - split into mean and logvar, then sample
    # This matches the DiagonalGaussianDistribution sampling in the reference implementation
    vae_encoder_out = torch.from_numpy(vae_encoder_out).to(torch.float32)
    mean, logvar = torch.chunk(vae_encoder_out, 2, dim=1)  # Each is (1, 4, 64, 64)
    
    # Sample from the distribution: latent = mean + std * noise
    std = torch.exp(0.5 * logvar)
    # Use a fixed seed for reproducibility (matching reference which uses torch.manual_seed(0))
    generator = torch.Generator().manual_seed(0)
    noise = torch.randn(mean.shape, generator=generator, device="cpu", dtype=torch.float32)
    latent = mean + std * noise
    
    latent = latent * 0.18215  # Scale latent
    return latent.numpy()


def preprocess_image(image: Image.Image, resize_mode: int) -> Image.Image:
    """
    Resize image to 512x512 based on resize_mode.
    0: Just resize (stretch)
    1: Crop and resize (cover)
    2: Resize and fill (contain)
    """
    target_w, target_h = 512, 512
    
    if resize_mode == 0:  # Just resize
        return image.resize((target_w, target_h), Image.LANCZOS)
    
    elif resize_mode == 1:  # Crop and resize
        w, h = image.size
        ratio = max(target_w / w, target_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        
        # Center crop
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        return image.crop((left, top, left + target_w, top + target_h))
        
    elif resize_mode == 2:  # Resize and fill
        w, h = image.size
        ratio = min(target_w / w, target_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        
        # Create black background and paste
        new_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        left = (target_w - new_w) // 2
        top = (target_h - new_h) // 2
        new_img.paste(image, (left, top))
        return new_img
        
    return image.resize((target_w, target_h), Image.LANCZOS)  # Fallback


def generate_txt2img(prompt: str, timesteps: np.ndarray = DEFAULT_TIMESTEPS, seed: int = None):
    prompt = maybe_convert_prompt(prompt, tokenizer)

    # Setup generator if seed provided
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    start_total = time.time()

    # text encoder
    start = time.time()
    with text_lock:
        prompt_embeds_npy = get_embeds(prompt, tokenizer, text_encoder)
    text_time = (time.time() - start) * 1000

    # initial latent
    latents_shape = [1, 4, 64, 64]
    latent = torch.randn(latents_shape, generator=generator, device="cpu", dtype=torch.float32,
                         layout=torch.strided).detach().numpy()

    # unet loop
    for i, timestep in enumerate(timesteps):
        with unet_lock:
            noise_pred = unet_session_main.run(None, {"sample": latent.astype(np.float32),
                                                     "/down_blocks.0/resnets.0/act_1/Mul_output_0": np.expand_dims(time_input_txt2img[i], axis=0),
                                                     "encoder_hidden_states": prompt_embeds_npy})[0]

        sample = latent
        model_output = noise_pred
        if i < len(timesteps) - 1:
            prev_timestep = timesteps[i + 1]
        else:
            prev_timestep = timestep

        alpha_prod_t = alphas_cumprod[timestep]
        alpha_prod_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0 else final_alphas_cumprod

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        scaled_timestep = timestep * 10
        c_skip = 0.5 ** 2 / (scaled_timestep ** 2 + 0.5 ** 2)
        c_out = scaled_timestep / (scaled_timestep ** 2 + 0.5 ** 2) ** 0.5
        predicted_original_sample = (sample - (beta_prod_t ** 0.5) * model_output) / (alpha_prod_t ** 0.5)

        denoised = c_out * predicted_original_sample + c_skip * sample

        if i != len(timesteps) - 1:
            noise = torch.randn(model_output.shape, generator=generator, device="cpu", dtype=torch.float32,
                                layout=torch.strided).to("cpu").detach().numpy()
            prev_sample = (alpha_prod_t_prev ** 0.5) * denoised + (beta_prod_t_prev ** 0.5) * noise
        else:
            prev_sample = denoised

        latent = prev_sample

    # vae decoder
    with vae_lock:
        latent = latent / 0.18215
        image = vae_decoder.run(None, {"x": latent.astype(np.float32)})[0]

    # Convert to PIL Image
    image = np.transpose(image, (0, 2, 3, 1)).squeeze(axis=0)
    image_denorm = np.clip(image / 2 + 0.5, 0, 1)
    image = (image_denorm * 255).round().astype("uint8")
    pil_image = Image.fromarray(image[:, :, :3])

    total_time = (time.time() - start_total) * 1000
    return pil_image, text_time, total_time


def generate_img2img(prompt: str, init_image: Image.Image, strength: float = 0.5, seed: int = None):
    # Mapping logic (Pinning)
    if strength <= 0.35:
        start_index = 3  # 1 step (259)
    elif strength <= 0.60:
        start_index = 2  # 2 steps (499, 259)
    elif strength <= 0.85:
        start_index = 1  # 3 steps (759, 499, 259)
    else:
        start_index = 0  # 4 steps (999, 759, 499, 259)
    
    timesteps = DEFAULT_TIMESTEPS[start_index:]
    # Use time_input_txt2img as the master source
    time_input = time_input_txt2img[start_index:]
    
    prompt = maybe_convert_prompt(prompt, tokenizer)

    # Setup generator if seed provided
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    start_total = time.time()

    # text encoder
    start = time.time()
    with text_lock:
        prompt_embeds_npy = get_embeds(prompt, tokenizer, text_encoder)
    text_time = (time.time() - start) * 1000

    # encode init image to latent
    latent = encode_image_to_latent(init_image)

    # Add noise for img2img (using the first timestep of the slice)
    timestep = timesteps[0]
    alpha_prod_t = alphas_cumprod[timestep]
    beta_prod_t = 1 - alpha_prod_t
    noise = torch.randn(latent.shape, generator=generator, device="cpu", dtype=torch.float32,
                        layout=torch.strided).detach().numpy()
    latent = (alpha_prod_t ** 0.5) * latent + (beta_prod_t ** 0.5) * noise

    # unet loop
    for i, timestep in enumerate(timesteps):
        with unet_lock:
            noise_pred = unet_session_main.run(None, {"sample": latent.astype(np.float32),
                                                     "/down_blocks.0/resnets.0/act_1/Mul_output_0": np.expand_dims(time_input[i], axis=0),
                                                     "encoder_hidden_states": prompt_embeds_npy})[0]

        sample = latent
        model_output = noise_pred
        if i < len(timesteps) - 1:
            prev_timestep = timesteps[i + 1]
        else:
            prev_timestep = timestep

        alpha_prod_t = alphas_cumprod[timestep]
        alpha_prod_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0 else final_alphas_cumprod

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        scaled_timestep = timestep * 10
        c_skip = 0.5 ** 2 / (scaled_timestep ** 2 + 0.5 ** 2)
        c_out = scaled_timestep / (scaled_timestep ** 2 + 0.5 ** 2) ** 0.5
        predicted_original_sample = (sample - (beta_prod_t ** 0.5) * model_output) / (alpha_prod_t ** 0.5)

        denoised = c_out * predicted_original_sample + c_skip * sample

        if i != len(timesteps) - 1:
            noise = torch.randn(model_output.shape, generator=generator, device="cpu", dtype=torch.float32,
                                layout=torch.strided).to("cpu").detach().numpy()
            prev_sample = (alpha_prod_t_prev ** 0.5) * denoised + (beta_prod_t_prev ** 0.5) * noise
        else:
            prev_sample = denoised

        latent = prev_sample

    # vae decoder
    with vae_lock:
        latent = latent / 0.18215
        image = vae_decoder.run(None, {"x": latent.astype(np.float32)})[0]

    # Convert to PIL Image
    image = np.transpose(image, (0, 2, 3, 1)).squeeze(axis=0)
    image_denorm = np.clip(image / 2 + 0.5, 0, 1)
    image = (image_denorm * 255).round().astype("uint8")
    pil_image = Image.fromarray(image[:, :, :3])

    total_time = (time.time() - start_total) * 1000
    return pil_image, text_time, total_time


@app.route("/generate", methods=["POST"])
def generate_route():
    try:
        data = request.get_json(force=True)
        log_request("/generate", data)
        mode = data.get("mode")
        if not mode or mode not in ["txt2img", "img2img"]:
            return jsonify({"error": "missing or invalid 'mode' field (must be 'txt2img' or 'img2img')"}), 400

        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"error": "missing 'prompt' field"}), 400

        # Handle seed
        seed = data.get("seed")
        if seed is None or seed == -1:
            seed = random.randint(0, 2**32 - 1)
        else:
            try:
                seed = int(seed)
            except ValueError:
                return jsonify({"error": "invalid 'seed' field (must be an integer)"}), 400

        init_image = None
        strength = 0.5
        if mode == "img2img":
            init_image_b64 = data.get("init_image")
            if not init_image_b64:
                return jsonify({"error": "missing 'init_image' field for img2img mode"}), 400
            
            # Handle denoising_strength
            strength = data.get("denoising_strength", data.get("strength", 0.5))
            try:
                strength = float(strength)
            except ValueError:
                strength = 0.5

            # Handle resize_mode
            resize_mode = data.get("resize_mode", 0)
            try:
                resize_mode = int(resize_mode)
                if resize_mode not in [0, 1, 2]:
                    resize_mode = 0
            except ValueError:
                resize_mode = 0

            try:
                init_image_data = base64.b64decode(init_image_b64)
                init_image = Image.open(BytesIO(init_image_data)).convert("RGB")
                # Preprocess image
                init_image = preprocess_image(init_image, resize_mode)
            except Exception as e:
                return jsonify({"error": f"invalid base64 image: {str(e)}"}), 400

        # Generate image
        if mode == "txt2img":
            pil_image, text_time, total_time = generate_txt2img(prompt, seed=seed)
        else:
            pil_image, text_time, total_time = generate_img2img(prompt, init_image, strength=strength, seed=seed)

        # Save image to OUTPUT_DIR for diagnostics
        filename = f"gen_{int(time.time() * 1000)}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        try:
            pil_image.save(save_path)
        except Exception:
            # if saving fails, continue and still return base64
            save_path = None

        # Encode to base64 for client
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        resp = {
            "status": "ok",
            "base64": img_base64,
            "text_time_ms": text_time,
            "total_time_ms": total_time,
            "seed": seed,
        }
        if save_path:
            resp["path"] = save_path

        return jsonify(resp)

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/sdapi/v1/txt2img", methods=["POST"])
def sdapi_txt2img():
    try:
        data = request.get_json(force=True)
        log_request("/sdapi/v1/txt2img", data)
        prompt = data.get("prompt", "")
        if not prompt:
            return jsonify({"error": "missing 'prompt' field"}), 400

        # Handle seed
        seed = data.get("seed")
        if seed is None or seed == -1:
            seed = random.randint(0, 2**32 - 1)
        else:
            try:
                seed = int(seed)
            except ValueError:
                return jsonify({"error": "invalid 'seed' field (must be an integer)"}), 400

        # Generate image using existing function
        pil_image, text_time, total_time = generate_txt2img(prompt, seed=seed)

        # Save image to OUTPUT_DIR for diagnostics
        filename = f"gen_{int(time.time() * 1000)}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        try:
            pil_image.save(save_path)
        except Exception:
            # if saving fails, continue and still return base64
            save_path = None

        # Encode to base64
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Format response like SD WebUI API
        response = {
            "images": [img_base64],
            "parameters": {
                "prompt": prompt,
                "steps": 4,  # Fixed for LCM
                "width": 512,
                "height": 512,
                "seed": seed,
            },
            "info": f"Generated in {total_time:.2f}ms"
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/sdapi/v1/img2img", methods=["POST"])
def sdapi_img2img():
    try:
        data = request.get_json(force=True)
        log_request("/sdapi/v1/img2img", data)
        prompt = data.get("prompt", "")
        if not prompt:
            return jsonify({"error": "missing 'prompt' field"}), 400

        init_images = data.get("init_images", [])
        if not init_images:
            return jsonify({"error": "missing 'init_images' field"}), 400

        # Handle seed
        seed = data.get("seed")
        if seed is None or seed == -1:
            seed = random.randint(0, 2**32 - 1)
        else:
            try:
                seed = int(seed)
            except ValueError:
                return jsonify({"error": "invalid 'seed' field (must be an integer)"}), 400

        # Handle resize_mode
        resize_mode = data.get("resize_mode", 0)
        try:
            resize_mode = int(resize_mode)
            if resize_mode not in [0, 1, 2]:
                resize_mode = 0
        except ValueError:
            resize_mode = 0

        # Handle denoising_strength
        strength = data.get("denoising_strength", 0.5)
        try:
            strength = float(strength)
        except ValueError:
            strength = 0.5

        # Decode base64 image
        init_image_b64 = init_images[0]
        try:
            init_image_data = base64.b64decode(init_image_b64)
            init_image = Image.open(BytesIO(init_image_data)).convert("RGB")
            # Preprocess image
            init_image = preprocess_image(init_image, resize_mode)
        except Exception as e:
            return jsonify({"error": f"invalid base64 image: {str(e)}"}), 400

        # Generate image using existing function
        pil_image, text_time, total_time = generate_img2img(prompt, init_image, strength=strength, seed=seed)

        # Save image to OUTPUT_DIR for diagnostics
        filename = f"gen_{int(time.time() * 1000)}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        try:
            pil_image.save(save_path)
        except Exception:
            # if saving fails, continue and still return base64
            save_path = None

        # Encode to base64
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Format response like SD WebUI API
        response = {
            "images": [img_base64],
            "parameters": {
                "prompt": prompt,
                "init_images": init_images,
                "denoising_strength": strength,
                "steps": 4,  # Fixed for LCM
                "width": 512,
                "height": 512,
                "seed": seed,
            },
            "info": f"Generated in {total_time:.2f}ms"
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/sdapi/v1/interrogate", methods=["POST"])
def sdapi_interrogate():
    if clip_interrogator is None:
        return jsonify({"error": "CLIP Interrogator is not initialized (models missing?)"}), 501

    try:
        data = request.get_json(force=True)
        log_request("/sdapi/v1/interrogate", data)
        image_b64 = data.get("image")
        if not image_b64:
            return jsonify({"error": "missing 'image' field"}), 400

        # Decode base64 image
        try:
            image_data = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_data)).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"invalid base64 image: {str(e)}"}), 400

        # Interrogate
        with clip_lock:
            # Use cached tags (None passed)
            scores, used_tags = clip_interrogator.interrogate(image)
            scores = scores[0]

        # Filter by threshold and sort by score
        threshold = 0.22
        top_k = 10
        results = []
        for i, score in enumerate(scores):
            if score >= threshold:
                results.append((used_tags[i], score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        caption = ", ".join([r[0] for r in results])
        if not caption:
            # Fallback to top 1 if nothing above threshold
            top_idx = np.argmax(scores)
            top_score = float(scores[top_idx])
            caption = used_tags[top_idx]
            results = [(caption, top_score)]

        # Return both the caption string and the detailed scores
        return jsonify({
            "caption": caption,
            "interrogations": [{"tag": r[0], "score": float(r[1])} for r in results]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/sdapi/v1/interrogate/structured", methods=["POST"])
def sdapi_interrogate_structured():
    if clip_interrogator is None:
        return jsonify({"error": "CLIP Interrogator is not initialized"}), 501

    try:
        data = request.get_json(force=True)
        log_request("/sdapi/v1/interrogate/structured", data)
        image_b64 = data.get("image")
        categories = data.get("categories")
        
        if not image_b64:
            return jsonify({"error": "missing 'image' field"}), 400
        if not categories or not isinstance(categories, dict):
            return jsonify({"error": "missing or invalid 'categories' field (must be a dictionary)"}), 400

        # Decode base64 image
        try:
            image_data = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_data)).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"invalid base64 image: {str(e)}"}), 400

        # Interrogate
        with clip_lock:
            results, img_emb = clip_interrogator.classify_categories(image, categories)
            
            # Also get general tags if we have cached embeddings
            general_tags = []
            if clip_interrogator.cached_embeddings is not None:
                scores = (img_emb @ clip_interrogator.cached_embeddings.T).flatten()
                # Get top 10 general tags
                top_indices = np.argsort(scores)[-10:][::-1]
                for idx in top_indices:
                    if scores[idx] > 0.2: # Minimum threshold for general tags
                        general_tags.append(clip_interrogator.cached_labels[idx])

        return jsonify({
            "results": results,
            "general_tags": general_tags
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # default host/port
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on http://{host}:{port}  (output -> {OUTPUT_DIR})")
    app.run(host=host, port=port, threaded=True)