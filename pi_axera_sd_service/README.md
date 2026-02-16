# Pi Axera SD Image Generator Service

This service provides a unified REST API endpoint for generating images using Stable Diffusion on Raspberry Pi with Axera AX650N hardware.

## Features

- **Unified Endpoint**: Single `/generate` endpoint supporting both txt2img and img2img modes
- **Structured Interrogation**: New `/sdapi/v1/interrogate/structured` endpoint for categorized classification (e.g., Demographics)
- **Base64 Response**: Images returned as base64-encoded PNG for direct client consumption
- **Hardware Optimized**: Tailored for Axera AX650N with LCM-LoRA acceleration
- **Large Taxonomy**: Built-in support for 20,101 ImageNet-21K labels with RAM caching
- **Systemd Integration**: Runs as a system service with automatic restart
- **Structured Logging**: Real-time JSON logging of request parameters with base64 redaction for easy debugging

## API

### POST /generate

Generate an image using Stable Diffusion.

#### Request Body (JSON)

```json
{
  "mode": "txt2img",
  "prompt": "A photograph of a red fox in the snow, highly detailed",
  "seed": 1234567890
}
```

For img2img:

```json
{
  "mode": "img2img",
  "prompt": "Make this image look like a pencil sketch",
  "init_image": "<base64 PNG>",
  "denoising_strength": 0.75,
  "resize_mode": 1
}
```

#### Response

Success:
```json
{
  "status": "ok",
  "base64": "<base64 PNG>",
  "text_time_ms": 123.45,
  "total_time_ms": 456.78,
  "seed": 1234567890
}
```

Error:
```json
{
  "status": "error",
  "error": "description of error"
}
```

### POST /sdapi/v1/interrogate

Standard flat interrogation (Top-K tags).

#### Request Body (JSON)
```json
{
  "image": "<base64 PNG>"
}
```

#### Response
```json
{
  "caption": "woman, portrait, long hair",
  "interrogations": [
    {"tag": "woman", "score": 0.28},
    {"tag": "portrait", "score": 0.25}
  ]
}
```

### POST /sdapi/v1/interrogate/structured

Categorized classification using Softmax normalization. Ideal for "Form Filling" (Demographics, Settings, etc.).

#### Request Body (JSON)
```json
{
  "image": "<base64 PNG>",
  "categories": {
    "gender": ["man", "woman", "boy", "girl"],
    "hair_color": ["pink", "blonde", "brown", "black"]
  }
}
```

#### Response
```json
{
  "results": {
    "gender": {"winner": "woman", "confidence": 0.94},
    "hair_color": {"winner": "pink", "confidence": 0.99}
  },
  "general_tags": ["portrait", "face", "long hair"]
}
```

## Installation

See [RASPBERRY_PI_SETUP.md](../RASPBERRY_PI_SETUP.md) in the repository root for complete, step-by-step installation instructions.

Quick summary:

1. **Create Python venv:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install flask transformers torch pillow numpy axengine cffi ml-dtypes
   ```

3. **Prepare model files** in `pi_axera_sd_service/models/`:
   - `.axmodel` files (text_encoder, unet, vae_decoder, vae_encoder) from `realistic-vision-v6-axera-hw/`
   - `tokenizer/` directory with vocab.json, merges.txt, tokenizer_config.json, special_tokens_map.json
   - `time_input_txt2img.npy` and `time_input_img2img.npy`
   - `text_encoder/config.json`
   
   ```bash
   # Example: copy from realistic-vision directory
   mkdir -p pi_axera_sd_service/models/text_encoder
   cp realistic-vision-v6-axera-hw/sd15_text_encoder_sim.axmodel pi_axera_sd_service/models/text_encoder/
   cp realistic-vision-v6-axera-hw/{unet,vae_decoder,vae_encoder}.axmodel pi_axera_sd_service/models/
   cp python/models/time_input_*.npy pi_axera_sd_service/models/
   cp -r realistic-vision-v6-axera-hw/tokenizer pi_axera_sd_service/models/
   cp python/models/text_encoder/config.json pi_axera_sd_service/models/text_encoder/
   ```

4. **Install systemd service:**
   ```bash
   sudo cp pi_axera_sd_generator.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable pi_axera_sd_generator
   sudo systemctl start pi_axera_sd_generator
   ```

## Configuration

Default environment variables (set in systemd service file):

| Variable | Default | Purpose |
|----------|---------|----------|
| `TEXT_MODEL_DIR` | `./pi_axera_sd_service/models/` | Directory containing tokenizer and text_encoder |
| `TEXT_ENCODER_MODEL` | `./pi_axera_sd_service/models/text_encoder/sd15_text_encoder_sim.axmodel` | Text encoder model path |
| `UNET_MODEL` | `./pi_axera_sd_service/models/unet.axmodel` | UNet model path |
| `VAE_DECODER_MODEL` | `./pi_axera_sd_service/models/vae_decoder.axmodel` | VAE decoder model path |
| `VAE_ENCODER_MODEL` | `./pi_axera_sd_service/models/vae_encoder.axmodel` | VAE encoder model path |
| `TIME_INPUT_TXT2IMG` | `./pi_axera_sd_service/models/time_input_txt2img.npy` | Time embeddings for txt2img |
| `TIME_INPUT_IMG2IMG` | `./pi_axera_sd_service/models/time_input_img2img.npy` | Time embeddings for img2img |
| `OUTPUT_DIR` | `./pi_axera_sd_service/output` | Directory to save generated images |
| `HOST` | `0.0.0.0` | Server bind address (0.0.0.0 = accessible from network) |
| `PORT` | `5000` | Server port |

To override at runtime (manual testing), export environment variables before running the script:

```bash
export HOST=127.0.0.1  # localhost only
export PORT=5001       # custom port
python pi_axera_sd_service/pi_axera_sd_generator.py
```

To override in systemd, edit `/etc/systemd/system/pi_axera_sd_generator.service` and update the `Environment=` lines, then reload:

```bash
sudo systemctl daemon-reload
sudo systemctl restart pi_axera_sd_generator
```

## Testing & Usage

**Wait for service to start** (~10-30 seconds for model loading):

```bash
sudo systemctl status pi_axera_sd_generator
# Should show: Active: active (running)
```

**Test txt2img:**

```bash
curl -X POST http://127.0.0.1:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"mode": "txt2img", "prompt": "a beautiful sunset over mountains"}'
```

**Test img2img** (requires base64-encoded PNG):

```bash
curl -X POST http://127.0.0.1:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "img2img",
    "prompt": "convert to sketch style",
    "init_image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
    "denoising_strength": 0.75
  }'
```

**Save response image:**

```bash
curl -s -X POST http://127.0.0.1:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"mode": "txt2img", "prompt": "test"}' | \
  python -c "import sys, json, base64; data = json.load(sys.stdin); open('output.png', 'wb').write(base64.b64decode(data['base64']))" && echo "Saved to output.png"
```

## Monitoring & Logs

The service logs all incoming request parameters in a structured JSON format. To prevent log pollution, large base64-encoded image data is automatically redacted and replaced with a character count.

To follow the logs in real-time:
```bash
sudo journalctl -u pi_axera_sd_generator.service -f
```

Example log output:
```json
[2025-12-27 09:31:08] Request to /generate:
{
  "mode": "img2img",
  "prompt": "a cute cat",
  "init_image": "<base64 data, 96 chars>",
  "steps": 4,
  "seed": 42
}
```

## Compatibility

- Compatible with Stable Diffusion WebUI API clients (subset of parameters)
- Works with picker client (`sd_client.py`)
- Optimized for Raspberry Pi with Axera AX650N

## Performance & Scalability

We conducted extensive load testing to determine the optimal configuration for the AX650N hardware. The results and scripts are available in the [parallel_experiment/](parallel_experiment/) directory.

**Key Takeaways:**
1.  **Single Instance is Optimal:** Running a single service instance provides the best balance of latency (~3s/image) and throughput.
2.  **Hardware Limits:** The hardware supports a maximum of **3 concurrent instances**. Attempting to run 4 causes driver failures.
3.  **Diminishing Returns:** Running multiple instances in parallel **does not improve total throughput** (capped at ~0.38 images/sec) but **significantly degrades latency** (increasing from 3s to ~8s per image).
4.  **Recommendation:** Stick to running one instance of the service on port 5000.

## Limitations & Robustness

- Output is always **512x512** pixels (hardware/model limitation).
- **Inference Steps:** `txt2img` is fixed at 4 steps. `img2img` supports 1-4 steps via the `denoising_strength` parameter.
- **No Negative Prompt / CFG Support:** The service runs at effective CFG=1.0 (no guidance) for maximum speed with LCM-LoRA. Negative prompts are ignored because the unconditional pass required for Classifier-Free Guidance is skipped to double inference throughput.
- No support for cfg_scale or other advanced parameters (not supported by hardware acceleration)
- **Robust parameter handling:** Any unsupported, extra, or out-of-range parameters (such as sampler, cfg_scale, n_iter, batch_size, or non-512 dimensions) are silently ignored. The service always attempts to yield a valid image, erring on the side of successful generation rather than erroring out.