# Unified Endpoint for SD Image Generation — Implementation Plan

## Overview
This implementation plan is specifically targeted for Raspberry Pi devices equipped with Axera AX650N (and optionally LLM8850) hardware. All API, deployment, and setup instructions are tailored for this platform and its constraints. The unified `/generate` endpoint supports both txt2img and img2img, inspired by Stable Diffusion WebUI standards and designed for seamless integration with the picker client and Axera hardware.

**Service Name:** `pi_axera_sd_generator`
**Code & Documentation Subdirectory:** `pi_axera_sd_service/`

All new code, documentation, deployment scripts, and configuration files for the image generation service should be placed in the `pi_axera_sd_service` subdirectory. References to service setup, systemd unit files, and usage should use the name `pi_axera_sd_generator`.

---

## 0. Feature & Parameter Discovery (Pre-Implementation Step)

- Before implementation begins, scan all `run_*.py` scripts in the `python/` directory to ascertain the available and supported image generation features and parameters.
- For each discovered feature/parameter:
  - Document its name, type, and usage in the script.
  - Indicate how it maps to Stable Diffusion API parameters (if applicable).
- Stop and prompt the user for review and selection of which features/parameters should be included in the unified endpoint.
- Only proceed with endpoint design and implementation after user confirmation of the selected features and their mappings.

---

## 1. API Design
### Endpoint
- **POST** `/generate`

### Payload (JSON)
- `mode`: "txt2img" or "img2img" (required)
- `prompt`: string (required)
- `init_image`: base64 string (optional, required for img2img)
- `width`, `height`: int (optional, defaults to hardware limits)
- `steps`: int (optional, defaults to hardware limits)
- `seed`: int (optional)
- `sampler_name`: string (optional, limited to supported samplers)
- `cfg_scale`: float (optional)
- `n_iter`, `batch_size`: int (optional, defaults to 1)
- Additional fields as supported by hardware (see picker/sd_config.py)

### Example Payloads
#### txt2img
```json
{
  "mode": "txt2img",
  "prompt": "A photograph of a red fox in the snow, highly detailed",
  "width": 512,
  "height": 512,
  "steps": 7
}
```
#### img2img
```json
{
  "mode": "img2img",
  "prompt": "Make this image look like a pencil sketch",
  "init_image": "<base64 PNG>",
  "width": 512,
  "height": 512,
  "steps": 7
}
```

---

## 2. Request Parsing
- Accepts JSON payload.
- For `img2img`, parses `init_image` as base64 PNG/JPEG, converts to PIL.Image.
- Validates required fields and supported options.
- Applies defaults from `sd_config.py` if fields are missing.

---

## 3. Image Generation Logic
- For `txt2img`, calls the existing txt2img pipeline (see `run_txt2img_axe_infer.py` or server).
- For `img2img`, decodes base64 image, calls img2img pipeline (see `run_img2img_axe_infer.py`).
- All generation functions should:
  - Encode the generated image as base64 for the network response.
  - (Saving to a server-side path is optional and only for server diagnostics or local persistence; clients do not receive file writes.)
  - Record timing (text encoder, total).

---

## 4. Response Format
- JSON object with:
  - `status`: "ok" or "error"
  - `base64`: base64-encoded PNG (for direct client consumption)
  - `text_time_ms`: time spent in text encoder
  - `total_time_ms`: total generation time
  - `error`: error message (if any)
  - (`path` may be included for server-side diagnostics, but is not intended for client use.)

#### Example Response
```json
{
  "status": "ok",
  "base64": "<base64 PNG>",
  "text_time_ms": 123,
  "total_time_ms": 456
}
```

---

## 5. Compatibility & Integration
- **Picker Client**: Ensure response and payload match picker expectations (see `sd_client.py`).
- **SD WebUI**: Protocol is subset-compatible; clients using SD WebUI API should work with minimal changes.
- **Hardware Limits**: Only expose options supported by Axera hardware (see `sd_config.py`).

---

## 6. Documentation
- Document payload and response in `/docs` and README.
- Provide example curl requests and responses.
- List supported options and defaults.

---

## 7. Implementation Steps
1. Refactor server code (e.g., `run_txt2img_axe_server.py`) to support both modes in `/generate`.
2. Add base64 image parsing for `img2img`.
3. Update response to include base64 result.
4. Validate and sanitize all inputs.
5. Test with picker client and SD WebUI-compatible requests.
6. Document endpoint and options.

---

## 8. Running as a System Service

To run the image generator server as a system service (using systemd):

1. **Provide a CLI Entrypoint**
   - Ensure the server can be started via a command such as:
     ```bash
     python pi_axera_sd_service/pi_axera_sd_generator.py
     ```
   - Accept environment variables or CLI arguments for configuration (host, port, model paths).

2. **Create a systemd Service Unit File**
   - Example: `/etc/systemd/system/pi_axera_sd_generator.service`
     ```ini
     [Unit]
     Description=Pi Axera SD Image Generator Service
     After=network.target

     [Service]
     Type=simple
     User=youruser
     WorkingDirectory=/home/youruser/sd1.5-lcm.axera_greg
     ExecStart=/usr/bin/python3 pi_axera_sd_service/pi_axera_sd_generator.py
     Restart=on-failure
     Environment=TEXT_MODEL_DIR=/home/youruser/sd1.5-lcm.axera_greg/models/
     Environment=UNET_MODEL=/home/youruser/sd1.5-lcm.axera_greg/models/unet.axmodel
     # Add other environment variables as needed

     [Install]
     WantedBy=multi-user.target
     ```

3. **Enable and Manage the Service**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable pi_axera_sd_generator
   sudo systemctl start pi_axera_sd_generator
   sudo systemctl status pi_axera_sd_generator
   ```

4. **Document Service Management**
   - Include instructions in the README or deployment docs for starting, stopping, and troubleshooting the service.

---

## 9. Raspberry Pi Setup & Deployment Documentation

- The implementation must include comprehensive Raspberry Pi setup and deployment instructions.
- Document:
  - Python environment creation and activation
  - Dependency installation (including platform-specific notes for axengine)
  - Model file layout and preparation
  - Manual server run instructions
  - Example API requests and expected responses
  - Systemd service setup and management
  - Performance tips, troubleshooting, and platform-specific caveats
- Reference and maintain the `RASPBERRY_PI_SETUP.md` in the repository root, ensuring it is up-to-date and covers all steps for Pi and similar ARM devices.

---

## References
- [plan-unifiedEndpointSdImageGeneration.prompt.md](.github/prompts/plan-unifiedEndpointSdImageGeneration.prompt.md)
- [picker/sd_client.py](hardware_exercises/picker/sd_client.py)
- [picker/sd_config.py](hardware_exercises/picker/sd_config.py)
- [run_txt2img_axe_server.py](python/run_txt2img_axe_server.py)
- [run_img2img_axe_infer.py](python/run_img2img_axe_infer.py)

---

## Notes
- The endpoint should be robust to missing/extra fields, returning clear errors.
- All timings should be included for diagnostics.
- Response should be as close as possible to picker client expectations.
- Only expose options that are actually supported by the hardware and server code.
- The server does not deliver file writes to clients; all image data is returned via network (base64 in JSON response).
