# Installing the Pi Axera SD Service on Raspberry Pi with AX650N

This document provides a complete, working setup guide for running the `pi_axera_sd_service` (unified image generator) on a Raspberry Pi or similar ARM Linux system with Axera AX650N hardware.

## Overview

The service is a Flask-based REST API that:
- Loads Stable Diffusion models (tokenizer, text encoder, UNet, VAE) once at startup
- Accepts requests via `/generate` endpoint supporting both txt2img and img2img modes
- Returns images as base64-encoded PNG
- Saves generated images locally for diagnostics
- Runs as a systemd service with automatic restart on failure

## Environment

- **Device**: Raspberry Pi 5 or similar ARM64 system with Debian-based OS
- **Hardware**: AX650N M.2 card with Axera runtime support (`libax_engine.so`)
- **Python**: 3.8+ (tested with 3.13)
- **Storage**: ~1.5 GB for models (SSD recommended)
- **RAM**: 8 GB minimum (models require significant memory)

## Prerequisites

1. Verify Axera card is present and functional:
   ```bash
   axcl-smi
   ```
   Should show AX650N device information.

2. Ensure the Axera runtime is installed and `libax_engine.so` is discoverable:
   ```bash
   python -c "import axengine; print('axengine ok')"
   ```
   If this fails, consult your Axera SDK documentation.

## Step 1: System Setup

Install system packages:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-venv python3-dev build-essential git
```

## Step 2: Create Python Virtual Environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## Step 3: Install Python Dependencies

```bash
pip install flask pillow numpy transformers torch axengine cffi ml-dtypes
```

**Note**: `axengine` is the Python wrapper for the Axera runtime. It is available as a wheel (`axengine-0.1.3-py3-none-any.whl` in the repo root) or from PyPI. If installation fails, verify that the Axera SDK and `libax_engine.so` are installed on your system.

## Step 4: Prepare Model Files

The service expects models in a specific directory structure. The `.axmodel` files are not included in `python/models/` and must be obtained from the `realistic-vision-v6-axera-hw/` directory (or another compatible model package).

**Required model directory structure:**

```
pi_axera_sd_service/models/
├── tokenizer/
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── text_encoder/
│   ├── config.json
│   └── sd15_text_encoder_sim.axmodel
├── unet.axmodel
├── vae_decoder.axmodel
├── vae_encoder.axmodel
├── time_input_txt2img.npy
└── time_input_img2img.npy
```

**Copy the model files:**

```bash
# From repository root
mkdir -p pi_axera_sd_service/models/text_encoder

# Copy .axmodel files from realistic-vision directory
cp realistic-vision-v6-axera-hw/sd15_text_encoder_sim.axmodel pi_axera_sd_service/models/text_encoder/
cp realistic-vision-v6-axera-hw/unet.axmodel pi_axera_sd_service/models/
cp realistic-vision-v6-axera-hw/vae_decoder.axmodel pi_axera_sd_service/models/
cp realistic-vision-v6-axera-hw/vae_encoder.axmodel pi_axera_sd_service/models/

# Copy time input files
cp python/models/time_input_*.npy pi_axera_sd_service/models/

# Copy tokenizer and text_encoder config
cp -r realistic-vision-v6-axera-hw/tokenizer pi_axera_sd_service/models/
cp python/models/text_encoder/config.json pi_axera_sd_service/models/text_encoder/

# Verify all files are in place
ls -la pi_axera_sd_service/models/
```

All `.axmodel` files should be present (total ~1.3 GB).

### Optional: Interrogation Model Files

The `/sdapi/v1/interrogate` and `/sdapi/v1/interrogate/structured` endpoints require two additional files placed directly in `pi_axera_sd_service/models/` (not inside `text_encoder/`):

```
pi_axera_sd_service/models/clip_base_vision.axmodel
pi_axera_sd_service/models/clip_base_text.axmodel
```

These are **not** produced by the standard model conversion pipeline and are **not tracked in git** (`.gitignore` excludes `*.axmodel`). They come from a separate CLIP export/compile pipeline documented in [img2txt/clip_implementation.md](img2txt/clip_implementation.md), so you'll need to either rebuild them yourself (ONNX export → Pulsar2 compile) or copy them from a previous working deployment. Without them, the service starts and works fine for txt2img/img2img — only the interrogation endpoints return `501`.

When these files are present, first startup (and any restart without a warm Hugging Face cache) also downloads `openai/clip-vit-base-patch32` (~1.2GB) from Hugging Face Hub to get the tokenizer/processor and projection weights. This requires outbound internet access; without an `HF_TOKEN` set, anonymous rate limits can occasionally cause a transient failure on a small file even after the large weights download successfully — a simple restart resolves it. See the troubleshooting section below.

## Step 5: Run the Service Manually (Testing)

Before installing as a systemd service, test the service manually:

```bash
source .venv/bin/activate
cd /path/to/sd1.5-lcm.axera  # repository root

# Run with default paths (models in pi_axera_sd_service/models/)
python pi_axera_sd_service/pi_axera_sd_generator.py
```

Expect ~10-30 seconds for model loading. Once started, you should see:
```
[INFO] Available providers:  ['AXCLRTExecutionProvider']
Loading tokenizer and models (this may take a moment)...
Starting server on http://0.0.0.0:5000  (output -> ./pi_axera_sd_service/output)
```

Test with curl (in another terminal):

```bash
curl -X POST http://127.0.0.1:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"mode": "txt2img", "prompt": "a beautiful sunset"}'
```

Response includes `"status":"ok"` and `"base64"` with the generated image. Press `Ctrl+C` to stop the server.

## Step 6: Install as Systemd Service

The service file is configured with default environment variables. It must be copied to `/etc/systemd/system/` and may need path adjustments.

```bash
sudo cp pi_axera_sd_service/pi_axera_sd_generator.service /etc/systemd/system/
```

**Update the service file paths** if your repository is not at `/home/gregm/sd1.5-lcm.axera`:

```bash
sudo nano /etc/systemd/system/pi_axera_sd_generator.service
```

Make sure these lines match your setup:
```ini
User=<your_username>  # usually 'gregm' or 'pi'
WorkingDirectory=/path/to/sd1.5-lcm.axera  # full path to repo root
ExecStart=/path/to/sd1.5-lcm.axera/.venv/bin/python pi_axera_sd_service/pi_axera_sd_generator.py
```

Then reload, enable, and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable pi_axera_sd_generator
sudo systemctl start pi_axera_sd_generator

# Check status
sudo systemctl status pi_axera_sd_generator

# Follow logs (Ctrl+C to stop)
sudo journalctl -u pi_axera_sd_generator -f
```

Once running, the service listens on `0.0.0.0:5000` and is accessible from any networked machine.

## Step 7: Test the Service

Once the systemd service is running, test with curl:

```bash
# txt2img
curl -s -X POST http://127.0.0.1:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"mode": "txt2img", "prompt": "a beautiful sunset"}' | grep -o '"status":"[^"]*"'

# Should output: "status":"ok"
```

Check service logs:

```bash
sudo journalctl -u pi_axera_sd_generator -n 20
```

## Troubleshooting

**Service fails to start - check logs:**
```bash
sudo journalctl -u pi_axera_sd_generator -f
```

Common errors:

- **RuntimeError: axclrtEngineLoadFromFile failed**
  - Verify all `.axmodel` files exist in `pi_axera_sd_service/models/`
  - Check file sizes: text_encoder ~248MB, unet ~940MB, vae_decoder ~70MB, vae_encoder ~47MB
  - Verify Axera runtime and card are functioning: `axcl-smi`

- **FileNotFoundError for tokenizer**
  - Verify `pi_axera_sd_service/models/tokenizer/` contains: vocab.json, merges.txt, tokenizer_config.json, special_tokens_map.json
  - Verify `pi_axera_sd_service/models/text_encoder/config.json` exists

- **Permission denied on /tmp/axcl/**
  - This is a warning from the Axera runtime and does not affect functionality

- **Service restarts repeatedly**
  - Increase `RestartSec` in the systemd service file (default 100ms may be too fast)
  - Check available memory: `free -h`

- **`501` from `/sdapi/v1/interrogate` or `/sdapi/v1/interrogate/structured`**
  - The startup log will show `CLIP Interrogator models not found, /sdapi/v1/interrogate will be disabled.`
  - Verify `pi_axera_sd_service/models/clip_base_vision.axmodel` and `clip_base_text.axmodel` exist (directly in `models/`, not `models/text_encoder/`). See "Optional: Interrogation Model Files" above.

- **`Failed to load CLIP Interrogator: Can't load image processor for 'openai/clip-vit-base-patch32'...`**
  - This happens when the Hugging Face Hub download of the (small) processor config fails after the large model weights already downloaded — typically an anonymous rate-limit or transient network hiccup, often alongside a `HF_TOKEN` warning in the logs just before it.
  - Fix: restart the service (`sudo systemctl restart pi_axera_sd_generator`) — most of the ~1.2GB is already cached locally, so only the small file needs to succeed. Setting an `HF_TOKEN` environment variable reduces the chance of this recurring.

**Network connectivity:**

If the service is bound to `0.0.0.0`, access from other machines on the network:

```bash
curl http://<pi-ip>:5000/generate -H "Content-Type: application/json" -d '{"mode": "txt2img", "prompt": "test"}'
```

## Performance Notes

- Model loading: ~10-30 seconds at startup
- Inference time: ~3 seconds per image on AX650N (hardware-accelerated)
- Output directory: generated PNGs saved to `pi_axera_sd_service/output/` for diagnostics
- The service is single-threaded per request but can handle concurrent requests via threading

## Next Steps

Refer to [pi_axera_sd_service/README.md](pi_axera_sd_service/README.md) for:
- Full API documentation
- Response formats
- Request examples (txt2img, img2img, interrogation)
- Configuration options
