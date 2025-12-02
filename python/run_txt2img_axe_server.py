from typing import List, Union
import numpy as np
import axengine
import torch
from PIL import Image
from transformers import CLIPTokenizer, PreTrainedTokenizer
import os
import time
import threading
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
TIME_INPUT = os.environ.get("TIME_INPUT", "./models/time_input_txt2img.npy")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./txt2img_server_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load tokenizer and models once
print("Loading tokenizer and models (this may take a moment)...")
tokenizer = CLIPTokenizer.from_pretrained(os.path.join(TEXT_MODEL_DIR, "tokenizer"))
text_encoder = axengine.InferenceSession(os.path.join(TEXT_MODEL_DIR, "text_encoder", "sd15_text_encoder_sim.axmodel"))
unet_session_main = axengine.InferenceSession(UNET_MODEL)
vae_decoder = axengine.InferenceSession(VAE_DECODER_MODEL)
time_input = np.load(TIME_INPUT)
alphas_cumprod, final_alphas_cumprod, self_timesteps = get_alphas_cumprod()

# Locks to avoid concurrent runs overlapping (safer for runtimes that are not thread-safe)
text_lock = threading.Lock()
unet_lock = threading.Lock()
vae_lock = threading.Lock()

# fixed timesteps used in the original example
DEFAULT_TIMESTEPS = np.array([999, 759, 499, 259]).astype(np.int64)


def generate_image(prompt: str, save_path: str, timesteps: np.ndarray = DEFAULT_TIMESTEPS):
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

    # save
    image = np.transpose(image, (0, 2, 3, 1)).squeeze(axis=0)
    image_denorm = np.clip(image / 2 + 0.5, 0, 1)
    image = (image_denorm * 255).round().astype("uint8")
    pil_image = Image.fromarray(image[:, :, :3])
    pil_image.save(save_path)

    total_time = (time.time() - start_total) * 1000
    return {
        "path": save_path,
        "text_time_ms": text_time,
        "total_time_ms": total_time,
    }


@app.route("/generate", methods=["POST"])
def generate_route():
    try:
        data = request.get_json(force=True)
        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"error": "missing 'prompt' field"}), 400

        filename = f"gen_{int(time.time() * 1000)}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)

        result = generate_image(prompt, save_path)
        return jsonify({"status": "ok", "path": result["path"], "text_time_ms": result["text_time_ms"], "total_time_ms": result["total_time_ms"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # default host/port
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on http://{host}:{port}  (output -> {OUTPUT_DIR})")
    app.run(host=host, port=port, threaded=True)
