from typing import List, Union
import numpy as np
import axengine
import torch
from PIL import Image
from transformers import CLIPTokenizer, PreTrainedTokenizer
import os
import time
import threading
import base64
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


app = Flask(__name__)

# Configuration (adjust paths as needed)
TEXT_MODEL_DIR = os.environ.get("TEXT_MODEL_DIR", "./models/")
UNET_MODEL = os.environ.get("UNET_MODEL", "./models/unet.axmodel")
VAE_DECODER_MODEL = os.environ.get("VAE_DECODER_MODEL", "./models/vae_decoder.axmodel")
VAE_ENCODER_MODEL = os.environ.get("VAE_ENCODER_MODEL", "./models/vae_encoder.axmodel")
TIME_INPUT_TXT2IMG = os.environ.get("TIME_INPUT_TXT2IMG", "./models/time_input_txt2img.npy")
TIME_INPUT_IMG2IMG = os.environ.get("TIME_INPUT_IMG2IMG", "./models/time_input_img2img.npy")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./txt2img_server_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load tokenizer and models once
print("Loading tokenizer and models (this may take a moment)...")
tokenizer = CLIPTokenizer.from_pretrained(os.path.join(TEXT_MODEL_DIR, "tokenizer"))
text_encoder = axengine.InferenceSession(os.path.join(TEXT_MODEL_DIR, "text_encoder", "sd15_text_encoder_sim.axmodel"))
unet_session_main = axengine.InferenceSession(UNET_MODEL)
vae_decoder = axengine.InferenceSession(VAE_DECODER_MODEL)
vae_encoder = axengine.InferenceSession(VAE_ENCODER_MODEL)
time_input_txt2img = np.load(TIME_INPUT_TXT2IMG)
time_input_img2img = np.load(TIME_INPUT_IMG2IMG)
alphas_cumprod, final_alphas_cumprod, self_timesteps = get_alphas_cumprod()

# Locks to avoid concurrent runs overlapping (safer for runtimes that are not thread-safe)
text_lock = threading.Lock()
unet_lock = threading.Lock()
vae_lock = threading.Lock()

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


def generate_txt2img(prompt: str, timesteps: np.ndarray = DEFAULT_TIMESTEPS):
    prompt = maybe_convert_prompt(prompt, tokenizer)

    start_total = time.time()

    # text encoder
    start = time.time()
    with text_lock:
        prompt_embeds_npy = get_embeds(prompt, tokenizer, text_encoder)
    text_time = (time.time() - start) * 1000

    # initial latent
    latents_shape = [1, 4, 64, 64]
    latent = torch.randn(latents_shape, generator=None, device="cpu", dtype=torch.float32,
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
            noise = torch.randn(model_output.shape, generator=None, device="cpu", dtype=torch.float32,
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


def generate_img2img(prompt: str, init_image: Image.Image, timesteps: np.ndarray = DEFAULT_TIMESTEPS):
    prompt = maybe_convert_prompt(prompt, tokenizer)

    start_total = time.time()

    # text encoder
    start = time.time()
    with text_lock:
        prompt_embeds_npy = get_embeds(prompt, tokenizer, text_encoder)
    text_time = (time.time() - start) * 1000

    # encode init image to latent
    latent = encode_image_to_latent(init_image)

    # Add noise for img2img (using the first timestep)
    timestep = timesteps[0]
    alpha_prod_t = alphas_cumprod[timestep]
    beta_prod_t = 1 - alpha_prod_t
    noise = torch.randn(latent.shape, generator=None, device="cpu", dtype=torch.float32,
                        layout=torch.strided).detach().numpy()
    latent = (alpha_prod_t ** 0.5) * latent + (beta_prod_t ** 0.5) * noise

    # unet loop
    for i, timestep in enumerate(timesteps):
        with unet_lock:
            noise_pred = unet_session_main.run(None, {"sample": latent.astype(np.float32),
                                                     "/down_blocks.0/resnets.0/act_1/Mul_output_0": np.expand_dims(time_input_img2img[i], axis=0),
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
            noise = torch.randn(model_output.shape, generator=None, device="cpu", dtype=torch.float32,
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
        mode = data.get("mode")
        if not mode or mode not in ["txt2img", "img2img"]:
            return jsonify({"error": "missing or invalid 'mode' field (must be 'txt2img' or 'img2img')"}), 400

        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"error": "missing 'prompt' field"}), 400

        init_image = None
        if mode == "img2img":
            init_image_b64 = data.get("init_image")
            if not init_image_b64:
                return jsonify({"error": "missing 'init_image' field for img2img mode"}), 400
            try:
                init_image_data = base64.b64decode(init_image_b64)
                init_image = Image.open(BytesIO(init_image_data)).convert("RGB")
                # Resize to 512x512 if needed
                if init_image.size != (512, 512):
                    init_image = init_image.resize((512, 512), Image.LANCZOS)
            except Exception as e:
                return jsonify({"error": f"invalid base64 image: {str(e)}"}), 400

        # Generate image
        if mode == "txt2img":
            pil_image, text_time, total_time = generate_txt2img(prompt)
        else:
            pil_image, text_time, total_time = generate_img2img(prompt, init_image)

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
        prompt = data.get("prompt", "")
        if not prompt:
            return jsonify({"error": "missing 'prompt' field"}), 400

        # Generate image using existing function
        pil_image, text_time, total_time = generate_txt2img(prompt)

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
        prompt = data.get("prompt", "")
        if not prompt:
            return jsonify({"error": "missing 'prompt' field"}), 400

        init_images = data.get("init_images", [])
        if not init_images:
            return jsonify({"error": "missing 'init_images' field"}), 400

        # Decode base64 image
        init_image_b64 = init_images[0]
        try:
            init_image_data = base64.b64decode(init_image_b64)
            init_image = Image.open(BytesIO(init_image_data)).convert("RGB")
            # Resize to 512x512 if needed
            if init_image.size != (512, 512):
                init_image = init_image.resize((512, 512), Image.LANCZOS)
        except Exception as e:
            return jsonify({"error": f"invalid base64 image: {str(e)}"}), 400

        # Generate image using existing function
        pil_image, text_time, total_time = generate_img2img(prompt, init_image)

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
                "steps": 4,  # Fixed for LCM
                "width": 512,
                "height": 512,
            },
            "info": f"Generated in {total_time:.2f}ms"
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # default host/port
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on http://{host}:{port}  (output -> {OUTPUT_DIR})")
    app.run(host=host, port=port, threaded=True)