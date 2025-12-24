# Pi Axera SD Image Generator Service

This service provides a unified REST API endpoint for generating images using Stable Diffusion on Raspberry Pi with Axera AX650N hardware.

## Features

- **Unified Endpoint**: Single `/generate` endpoint supporting both txt2img and img2img modes
- **Base64 Response**: Images returned as base64-encoded PNG for direct client consumption
- **Hardware Optimized**: Tailored for Axera AX650N with LCM-LoRA acceleration
- **Thread-Safe**: Concurrent request handling with proper locking
- **Systemd Integration**: Runs as a system service with automatic restart

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
  "init_image": "<base64 PNG>"
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

## Installation

1. Ensure all model files are in the `models/` directory
2. Install dependencies: `pip install flask transformers torch pillow numpy axengine`
3. Copy the service file to systemd: `sudo cp pi_axera_sd_generator.service /etc/systemd/system/`
4. Reload systemd: `sudo systemctl daemon-reload`
5. Enable and start: `sudo systemctl enable pi_axera_sd_generator && sudo systemctl start pi_axera_sd_generator`

## Configuration

Environment variables:

- `TEXT_MODEL_DIR`: Path to text encoder and tokenizer (default: `./models/`)
- `UNET_MODEL`: Path to UNet model (default: `./models/unet.axmodel`)
- `VAE_DECODER_MODEL`: Path to VAE decoder (default: `./models/vae_decoder.axmodel`)
- `VAE_ENCODER_MODEL`: Path to VAE encoder (default: `./models/vae_encoder.axmodel`)
- `TIME_INPUT_TXT2IMG`: Path to txt2img time input (default: `./models/time_input_txt2img.npy`)
- `TIME_INPUT_IMG2IMG`: Path to img2img time input (default: `./models/time_input_img2img.npy`)
- `OUTPUT_DIR`: Directory for temporary outputs (default: `./txt2img_server_output`)
- `HOST`: Server host (default: `127.0.0.1`)
- `PORT`: Server port (default: `5000`)

## Testing

Test txt2img:
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"mode": "txt2img", "prompt": "A beautiful landscape"}'
```

Test img2img:
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"mode": "img2img", "prompt": "Convert to sketch", "init_image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="}'
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

- Output is always **512x512** pixels and uses **4 inference steps** (hardware/model limitation), regardless of input parameters.
- **No Negative Prompt / CFG Support:** The service runs at effective CFG=1.0 (no guidance) for maximum speed with LCM-LoRA. Negative prompts are ignored because the unconditional pass required for Classifier-Free Guidance is skipped to double inference throughput.
- No support for cfg_scale or other advanced parameters (not supported by hardware acceleration)
- **Robust parameter handling:** Any unsupported, extra, or out-of-range parameters (such as sampler, cfg_scale, n_iter, batch_size, or non-512 dimensions/steps) are silently ignored. The service always attempts to yield a valid image, erring on the side of successful generation rather than erroring out.