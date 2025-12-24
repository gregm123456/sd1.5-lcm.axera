# Pi Axera SD Generator — Generation Tips & API Reference

This document details all supported parameters for the `/generate` endpoint, with usage notes and tested curl examples for both txt2img and img2img modes.

---

## Supported Parameters

### Common (Both Modes)
| Parameter      | Type    | Required | Description                                      |
|---------------|---------|----------|--------------------------------------------------|
| mode          | string  | yes      | "txt2img" or "img2img"                           |
| prompt        | string  | yes      | Text prompt for image generation                 |
| seed          | int     | no       | Seed for deterministic generation (random if omitted) |

### img2img Only
| Parameter          | Type    | Required | Description                                          |
|-------------------|---------|----------|------------------------------------------------------|
| init_image        | base64  | yes      | Base64-encoded PNG/JPEG (512x512 recommended)        |
| denoising_strength| float   | no       | 0.0-1.0. Controls modification level (def: 0.5)      |
| resize_mode       | int     | no       | 0=Stretch (def), 1=Crop, 2=Pad                       |

---

## Fixed Parameters (Hardware Locked)

The following parameters are accepted for API compatibility (e.g., with SD WebUI clients) but are **fixed by the compiled hardware models**. Any values passed for these will be ignored.

| Parameter      | Type    | Value | Reason                                           |
|---------------|---------|-------|--------------------------------------------------|
| width         | int     | 512   | UNet/VAE models are compiled for 512x512         |
| height        | int     | 512   | UNet/VAE models are compiled for 512x512         |
| steps (txt2img)| int     | 4     | LCM-LoRA is optimized for 4-step inference       |

> **Note on Steps:**
> While `txt2img` is fixed at 4 steps, `img2img` uses a subset of these steps (1-4) based on the `denoising_strength` provided. Passing a custom `steps` parameter is still ignored; the step count is derived solely from the strength.

---

## Not Supported (Robustly Ignored)
- negative_prompt
- sampler_name
- cfg_scale
- n_iter
- batch_size

> **Note on Negative Prompts:**
> Negative prompts are not supported because the service uses LCM-LoRA without Classifier-Free Guidance (CFG). Enabling CFG would require running the model twice per step (conditional + unconditional), effectively halving the performance. To maintain real-time speeds on the embedded hardware, the unconditional pass is skipped.

> **Note:**
> This implementation is designed for specialized hardware and robustly ignores any unsupported, extra, or out-of-range parameters. **The output will always be 512x512 pixels and use 4 inference steps, regardless of input.** The service always attempts to yield an image if possible, rather than returning an error for unknown or unhandled fields. This ensures maximum compatibility with clients that may send a superset of parameters.

---

## Example: txt2img

**Minimal:**
```bash
curl -sS -X POST http://127.0.0.1:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"mode":"txt2img","prompt":"A red fox in the snow, highly detailed"}'
```

**With optional fields (including fixed hardware parameters):**
```bash
curl -sS -X POST http://127.0.0.1:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "txt2img",
    "prompt": "A futuristic cityscape, night, neon lights",
    "width": 512,
    "height": 512,
    "steps": 4
  }'
```

---

## Example: img2img

**Minimal:**
```bash
B64="<base64-encoded-512x512-png>"
curl -sS -X POST http://127.0.0.1:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"mode":"img2img","prompt":"Make this a pencil sketch","init_image":"'${B64}'"}'
```

**With optional fields (including fixed hardware parameters):**
```bash
B64="<base64-encoded-512x512-png>"
curl -sS -X POST http://127.0.0.1:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "img2img",
    "prompt": "Make this a pencil sketch, high contrast",
    "init_image": "'${B64}'",
    "denoising_strength": 0.75,
    "width": 512,
    "height": 512
  }'
```

---

## Response Format

Success:
```json
{
  "status": "ok",
  "base64": "<base64 PNG>",
  "text_time_ms": 123.45,
  "total_time_ms": 456.78,
  "seed": 1234567890,
  "path": "pi_axera_sd_service/output/gen_1700000000000.png"
}
```

Error:
```json
{
  "status": "error",
  "error": "description of error"
}
```

---

## Tips
- Output is always **512x512** resolution (hardware limitation).
- **Steps:** `txt2img` always uses 4 steps. `img2img` uses 1-4 steps depending on `denoising_strength`.
- Extra, unknown, or out-of-range fields are robustly ignored.
- For img2img, always provide a base64-encoded PNG or JPEG (512x512 recommended).
- The `path` field in the response shows where the PNG was saved on the server (for diagnostics).
- All images are also returned as base64 PNG in the response.

---

## Denoising Strength Mapping (img2img)
The service maps the float `denoising_strength` to discrete hardware-supported steps:

| Range | Steps | Description |
|-------|-------|-------------|
| 0.00 - 0.35 | 1 | Minimal changes, very fast |
| 0.36 - 0.60 | 2 | Balanced (Default) |
| 0.61 - 0.85 | 3 | Strong modification |
| 0.86 - 1.00 | 4 | Complete reimagining |

---

## Troubleshooting
- If you get a shape or dtype error, check that your `init_image` is a 512x512 PNG/JPEG.
- If you get a missing field error, check your JSON keys and required fields.
- For best results, use prompts similar to those used in Stable Diffusion WebUI.

---

## See Also
- `README.md` for deployment and systemd setup
- `RASPBERRY_PI_SETUP.md` for full Pi/AX650N environment setup
