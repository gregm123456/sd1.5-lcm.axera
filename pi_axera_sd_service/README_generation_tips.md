# Pi Axera SD Generator — Generation Tips & API Reference

This document details all supported parameters for the `/generate` endpoint, with usage notes and tested curl examples for both txt2img and img2img modes.

---

## Supported Parameters

### Common (Both Modes)
| Parameter      | Type    | Required | Description                                      |
|---------------|---------|----------|--------------------------------------------------|
| mode          | string  | yes      | "txt2img" or "img2img"                           |
| prompt        | string  | yes      | Text prompt for image generation                 |

### txt2img Only
| Parameter      | Type    | Required | Description                                      |
|---------------|---------|----------|--------------------------------------------------|
| width         | int     | no       | Image width (default: 512, fixed)                |
| height        | int     | no       | Image height (default: 512, fixed)               |
| steps         | int     | no       | Inference steps (default: 4, fixed by hardware)  |

### img2img Only
| Parameter      | Type    | Required | Description                                      |
|---------------|---------|----------|--------------------------------------------------|
| init_image    | base64  | yes      | Base64-encoded PNG/JPEG (512x512 recommended)    |


### Not Supported (Robustly Ignored or Fixed by Hardware)
- seed
- sampler_name
- cfg_scale
- n_iter
- batch_size

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

**With all supported fields:**
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

**With all supported fields:**
```bash
B64="<base64-encoded-512x512-png>"
curl -sS -X POST http://127.0.0.1:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "img2img",
    "prompt": "Make this a pencil sketch, high contrast",
    "init_image": "'${B64}'",
    "width": 512,
    "height": 512,
    "steps": 4
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
- Output is always **512x512** resolution and **4 steps** (hardware limitation), regardless of input.
- Extra, unknown, or out-of-range fields are robustly ignored.
- For img2img, always provide a base64-encoded PNG or JPEG (512x512 recommended).
- The `path` field in the response shows where the PNG was saved on the server (for diagnostics).
- All images are also returned as base64 PNG in the response.

---

## Troubleshooting
- If you get a shape or dtype error, check that your `init_image` is a 512x512 PNG/JPEG.
- If you get a missing field error, check your JSON keys and required fields.
- For best results, use prompts similar to those used in Stable Diffusion WebUI.

---

## See Also
- `README.md` for deployment and systemd setup
- `RASPBERRY_PI_SETUP.md` for full Pi/AX650N environment setup
