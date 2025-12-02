# Raspberry Pi setup for axera/stable-diffusion server

This document explains the steps we followed to get the `python/run_txt2img_axe_server.py` server running on a Raspberry Pi (or similar single-board Linux). It is intended to be a practical, repeatable checklist including common pitfalls and a sample `systemd` service.

## Summary

- Server purpose: load models once, accept HTTP POST `/generate` with JSON `{ "prompt": "..." }`, and save generated images to an output directory.
- Key idea: keep tokenizer, text encoder, UNet and VAE loaded in memory so repeated requests do not re-load models.

## Tested environment notes

- Device: Raspberry Pi (Raspbian / Raspberry Pi OS) or other Debian-based ARM Linux.
- Python: 3.8+ (use the system `python3` or a dedicated venv).
- Disk: models are large — ensure you have enough storage (SSD recommended).
- Memory: Stable Diffusion models are memory hungry. On Pi-class devices you will likely need either a tiny/quantized model or offload heavy work to a server with a GPU. This doc documents what we did and tradeoffs.

## Prerequisites

- Copy or place the `models/` directory from this repo to the Pi (or point `TEXT_MODEL_DIR` to wherever your models live).
- Make sure `time_input_txt2img.npy` and the `.axmodel` files exist where the server expects them.

## Create a Python virtual environment (recommended)

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-venv python3-dev build-essential git
python3 -m venv ~/sd_env
source ~/sd_env/bin/activate
pip install --upgrade pip
```

## Install Python dependencies

- Minimal dependencies required for the server script:

```bash
pip install flask pillow numpy transformers torch
```

- Note about `axengine` / `pyaxengine`:
  - The repo uses `axengine.InferenceSession` to run `.axmodel` artifacts. Installing `pyaxengine` or the vendor-provided runtime may be platform-specific and not available from PyPI for ARM. On the Pi we either:
    - Use a prebuilt runtime/SDK for the platform (follow the vendor docs), or
    - Run the server on a more powerful host (x86_64) and have the Pi act as a thin client.
  - If `pip install pyaxengine` fails, consult your device's vendor runtime package or use a different device with supported runtime.

## Model files and layout

Keep the same relative layout as the repository or set environment variables to point at your model location.

- Default expected layout relative to server working directory:

```
models/
  tokenizer/
  text_encoder/sd15_text_encoder_sim.axmodel
  unet.axmodel
  vae_decoder.axmodel
  time_input_txt2img.npy
```

## Run the server manually

From the repository root (or wherever you placed the models):

```bash
source ~/sd_env/bin/activate
# optionally set paths if not default
export TEXT_MODEL_DIR=./models/
export UNET_MODEL=./models/unet.axmodel
export VAE_DECODER_MODEL=./models/vae_decoder.axmodel
export TIME_INPUT=./models/time_input_txt2img.npy
export OUTPUT_DIR=./txt2img_server_output

python python/run_txt2img_axe_server.py
```

- The server prints a message like `Starting server on http://127.0.0.1:5000  (output -> ./txt2img_server_output)` and will load the tokenizer and models once at startup. Startup may take many seconds or minutes depending on device.

## Example request

```bash
curl -X POST http://127.0.0.1:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A photograph of a red fox in the snow, highly detailed"}'
```

Response example (JSON):

```json
{"status":"ok","path":"./txt2img_server_output/gen_1700000000000.png","text_time_ms":XXX,"total_time_ms":YYY}
```

## Run as a background service (systemd)

Create a systemd unit to keep the server running and restart on failure. Example `/etc/systemd/system/sd_txt2img.service`:

```ini
[Unit]
Description=StableDiffusion axengine server
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/sd1.5-lcm.axera_greg
Environment=TEXT_MODEL_DIR=/home/pi/sd1.5-lcm.axera_greg/models/
Environment=UNET_MODEL=/home/pi/sd1.5-lcm.axera_greg/models/unet.axmodel
Environment=VAE_DECODER_MODEL=/home/pi/sd1.5-lcm.axera_greg/models/vae_decoder.axmodel
Environment=TIME_INPUT=/home/pi/sd1.5-lcm.axera_greg/models/time_input_txt2img.npy
Environment=OUTPUT_DIR=/home/pi/sd1.5-lcm.axera_greg/txt2img_server_output
ExecStart=/home/pi/sd_env/bin/python /home/pi/sd1.5-lcm.axera_greg/python/run_txt2img_axe_server.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable sd_txt2img.service
sudo systemctl start sd_txt2img.service
sudo journalctl -u sd_txt2img.service -f
```

## Performance tips and memory constraints

- If you run out of RAM during model loading or inference:
  - Use a machine with more RAM or GPU acceleration.
  - Consider using quantized / small models tailored for edge devices.
  - Add swap (slow) or zram (recommended) but be aware of wear on SD cards.
- Avoid running other memory-heavy processes while the server is loaded.

## Troubleshooting

- `ImportError` for `axengine` / `pyaxengine`: install the vendor runtime that supports your platform. PyPI may not contain ARM-compatible wheels.
- `FileNotFoundError` for model files: verify `TEXT_MODEL_DIR`, `UNET_MODEL`, `VAE_DECODER_MODEL`, and `TIME_INPUT` point to the correct files.
- Permission errors when running systemd: check `User=` and file permissions.

## Next steps and optional improvements

- Add more POST params: `seed`, `timesteps`, `width`, `height`, `num_inference_steps`.
- Add authentication to the HTTP endpoint for security.
- Add a request queue or a process pool for better throughput and isolation.
- Add health-check endpoint and metrics.

## A note on where things may differ from Pi to desktop

- The repository was developed and tested on a machine capable of running the `axengine` runtime. If the Pi cannot run the runtime you can still use the server approach, but run the server on a capable machine and have the Pi send requests to it.

---

If you'd like, I can also add a short `README` snippet to the repo root or create the systemd unit file for you. Tell me whether you want the server to bind to `0.0.0.0` for LAN access or keep it `127.0.0.1`.
# Raspberry Pi / AX650N Setup for sd1.5-lcm.axera

This document captures the exact steps we used to get the project running with Axera AXE (Axera NPU) support on a Raspberry Pi / host with an AX650N M.2 card. It covers Python env setup, PyAXEngine installation, ensuring the native runtime `libax_engine.so` is available, copying models/tokenizer into `./models`, and commands to run the provided inference scripts.

**Important**: adjust paths and versions to match your environment. These commands assume you're running from the repository root (`/home/gregm/sd1.5-lcm.axera_greg`).

**Prerequisites**
- Python 3.8+ (use the same `python` to run the repo scripts and to install packages).
- Access to the Axera native runtime library (`libax_engine.so`) from an AX650 SDK or system package.
- M.2 AX650N card inserted and driver installed if using the card (check `axcl-smi`).
- Sufficient disk space (models are large).

**Quick checks**
- Confirm `python` executable used by your shells:

```bash
which python
python -c "import sys; print('python executable:', sys.executable)"
```

- Confirm Axera card is visible (example output shows AX650N):

```bash
axcl-smi
# output: shows AX650N present and memory usage
```


**1) Create and activate a Python virtual environment (recommended)**

```bash
# from repo root
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**2) Install required Python packages**

Install the Python dependencies needed by the repo and for PyAXEngine examples:

```bash
pip install cffi ml-dtypes numpy pillow opencv-python-headless
# Also install repo-specific deps if needed (optional)
pip install -r lcm-lora-sdv1-5/requirements.txt 2>/dev/null || true
```

**3) Install PyAXEngine (the Python wrapper that provides `import axengine`)**

You have two main options:

A) Install the official wheel (recommended when available)

1. Download the latest `axengine-<version>-py3-none-any.whl` from the PyAXEngine GitHub releases page:
   https://github.com/AXERA-TECH/pyaxengine/releases/latest

2. Then install it with pip (use the same Python/venv as above):

```bash
pip install /path/to/axengine-<version>-py3-none-any.whl
```

B) Install directly from GitHub (pip will build/install):

```bash
pip install git+https://github.com/AXERA-TECH/pyaxengine.git
```

C) Or clone and install locally:

```bash
git clone https://github.com/AXERA-TECH/pyaxengine.git
cd pyaxengine
pip install .
```

**4) Ensure the native runtime library `libax_engine.so` is present and discoverable**

PyAXEngine is a wrapper for the native Axera runtime. You need `libax_engine.so` from the Axera SDK. Make it discoverable in one of these ways:

A) Install it system-wide and run `ldconfig`:

```bash
sudo cp /path/to/ax_sdk/lib/libax_engine.so /usr/local/lib/
sudo ldconfig
```

B) Or set `LD_LIBRARY_PATH` in your shell before running the scripts:

```bash
export LD_LIBRARY_PATH=/path/to/ax_sdk/lib:$LD_LIBRARY_PATH
# add to ~/.bashrc to persist if desired
```

If you don't have the SDK: the PyAXEngine README mentions you can request the SDK or at least `libax_engine.so` from Axera.

**5) Verify `axengine` import**

After installation and library setup, test Python import:

```bash
python -c "import axengine; print('axengine ok, version:', getattr(axengine,'__version__','unknown'))"
```

If this errors with `ModuleNotFoundError: No module named 'axengine'`, re-check you installed the wheel into the same Python environment.

If `import axengine` succeeds but runtime fails to find `libax_engine.so`, you'll see linker errors or runtime provider discovery errors—revisit `LD_LIBRARY_PATH` or `ldconfig` steps.

**6) Prepare models/tokenizer for the repo**

The project's inference scripts expect model files under `./models` by default. If you already have `lcm-lora-sdv1-5` as a subdirectory (it contains `models/`), you can either copy or symlink the relevant files into `./models`.

Safe copy commands (non-destructive; backs up existing files to `models.bak`):

```bash
# from repo root
mkdir -p models models.bak
for f in unet.onnx unet.axmodel vae_decoder.onnx vae_decoder.axmodel vae_encoder.onnx vae_encoder.axmodel time_input_txt2img.npy time_input_img2img.npy; do
  if [ -e "models/$f" ]; then
    mv "models/$f" "models.bak/"
  fi
done
if [ -d "models/tokenizer" ] || [ -d "models/text_encoder" ]; then
  mv models/tokenizer models.bak/ 2>/dev/null || true
  mv models/text_encoder models.bak/ 2>/dev/null || true
fi

# copy models and related folders from subproject (won't overwrite due to -n)
cp -n lcm-lora-sdv1-5/models/unet.onnx models/ 2>/dev/null || true
cp -n lcm-lora-sdv1-5/models/unet.axmodel models/ 2>/dev/null || true
cp -n lcm-lora-sdv1-5/models/vae_decoder.onnx models/ 2>/dev/null || true
cp -n lcm-lora-sdv1-5/models/vae_decoder.axmodel models/ 2>/dev/null || true
cp -n lcm-lora-sdv1-5/models/vae_encoder.onnx models/ 2>/dev/null || true
cp -n lcm-lora-sdv1-5/models/vae_encoder.axmodel models/ 2>/dev/null || true
cp -n lcm-lora-sdv1-5/models/time_input_txt2img.npy models/ 2>/dev/null || true
cp -n lcm-lora-sdv1-5/models/time_input_img2img.npy models/ 2>/dev/null || true

# copy tokenizer and text_encoder directories
cp -rn lcm-lora-sdv1-5/models/tokenizer models/tokenizer
cp -rn lcm-lora-sdv1-5/models/text_encoder models/text_encoder
```

Alternative: create symlinks (lightweight, no extra disk use):

```bash
ln -sfn "$(pwd)/lcm-lora-sdv1-5/models/unet.onnx" models/unet.onnx
ln -sfn "$(pwd)/lcm-lora-sdv1-5/models/unet.axmodel" models/unet.axmodel
ln -sfn "$(pwd)/lcm-lora-sdv1-5/models/vae_decoder.onnx" models/vae_decoder.onnx
ln -sfn "$(pwd)/lcm-lora-sdv1-5/models/vae_decoder.axmodel" models/vae_decoder.axmodel
ln -sfn "$(pwd)/lcm-lora-sdv1-5/models/tokenizer" models/tokenizer
ln -sfn "$(pwd)/lcm-lora-sdv1-5/models/text_encoder" models/text_encoder
```

Notes:
- Copy is safest if you may remove or move the subproject later; symlink is convenient for development.
- The tokenizer must live at `./models/tokenizer` (the Transformers tokenizer code will look for that path).

**7) Run an ONNX test (CPU or ONNXRuntime GPU if available)**

If you want to run ONNX-based inference (example below):

```bash
# default script uses ./models/unet.onnx and ./models/vae_decoder.onnx
python python/run_txt2img_onnx_infer.py

# or explicitly point to models
python python/run_txt2img_onnx_infer.py --unet_model ./models/unet.onnx --vae_decoder_model ./models/vae_decoder.onnx
```

If ONNXRuntime tries to access a GPU it can't find, you'll see warnings about device discovery failing; it's usually safe (falls back to CPU) but for GPU use ensure the runtime and drivers match your hardware.

**8) Run AXE (axmodel) inference using PyAXEngine**

Once `axengine` works and `libax_engine.so` is discoverable, run the AXE example (this will use the Axera providers):

```bash
# default script expects ./models/unet.axmodel and ./models/vae_decoder.axmodel
python python/run_txt2img_axe_infer.py

# or explicitly
python python/run_txt2img_axe_infer.py --unet_model ./models/unet.axmodel --vae_decoder_model ./models/vae_decoder.axmodel
```

When running on an M.2 AX650N card, set provider flags if the script supports them (PyAXEngine examples accept `-p AXCLRTExecutionProvider` for card usage or `-p AxEngineExecutionProvider` for board): see the PyAXEngine README/examples.

**9) Troubleshooting**

- `ModuleNotFoundError: No module named 'axengine'`
  - Ensure you installed PyAXEngine into the same Python environment. Re-run `pip show axengine` or `pip list` to confirm.

- `ImportError` or runtime error about `libax_engine.so`
  - Ensure the native library is on `LD_LIBRARY_PATH` or installed in `/usr/local/lib` and `ldconfig` run.

- Tokenizer errors like `Can't load tokenizer for './models/tokenizer'` or `Repo id must be in the form`
  - The Transformers code sometimes tries to treat a local path as an HF repo id if you call `from_pretrained` incorrectly. Use `CLIPTokenizer.from_pretrained('./models/tokenizer')` (local dir) — having the files (`vocab.json`, `merges.txt`, `tokenizer_config.json`) present in `./models/tokenizer` fixes this.

- ONNXRuntime GPU discovery warnings
  - If you see `GPU device discovery failed` it usually indicates ONNXRuntime wasn't built with your target GPU provider. Verify ONNXRuntime installation and drivers.

**10) Production note**
- The PyAXEngine project is great for prototyping. For production deployments on M.2 cards, consider `pyaxcl` (the Axera pyAXCL package) as it provides fuller card features.

**11) Licensing & SDK**
- Check `lcm-lora-sdv1-5/LICENSE` and the Axera SDK license to confirm redistribution/usage rules for models and runtime.

**12) Useful commands recap**

```bash
# verify python
which python
python -c "import sys; print(sys.executable)"

# install deps
pip install cffi ml-dtypes numpy pillow opencv-python-headless

# install pyaxengine (wheel)
pip install /path/to/axengine-<version>-py3-none-any.whl

# OR install from GitHub
pip install git+https://github.com/AXERA-TECH/pyaxengine.git

# make native lib discoverable
export LD_LIBRARY_PATH=/path/to/ax_sdk/lib:$LD_LIBRARY_PATH
sudo ldconfig

# copy models/tokenizer from subproject
# (see full copy snippet above)

# run onnx
python python/run_txt2img_onnx_infer.py

# run axe
python python/run_txt2img_axe_infer.py
```

If you want, I can also add a small `scripts/` helper to automate copying/symlinking models into `./models` and to run a smoke test. Tell me whether you prefer copying or symlinking, and I will add that helper script and run it.

---
File: `RASPBERRY_PI_SETUP.md`
Created in repository root. Follow up if you want an automated script added.
