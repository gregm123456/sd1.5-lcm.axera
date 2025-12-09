Unified Endpoint for SD Image Generation

Design a single `/generate` endpoint supporting both txt2img and img2img, inspired by Stable Diffusion WebUI standards and `/docs` conventions. Accepts image uploads as base64, matching typical SD API practices.

Additional Notes:
- The `run_*.py` sample code for Ax650N hardware (Raspberry Pi) demonstrates the full, though limited, range of image generation capabilities available for exposure. The endpoint protocol should ideally be a subset of Stable Diffusion, but is not strictly bound to it.
- The initial deployment will primarily serve the picker application. The endpoint and protocol should be designed to minimize required changes to the picker client, ensuring easy integration.

Steps:
1. Review Stable Diffusion WebUI API docs (`/docs`) for standard payloads, especially for txt2img and img2img.
2. Define a unified `/generate` endpoint accepting a JSON payload with:
   - `mode`: "txt2img" or "img2img"
   - `prompt`: string
   - `init_image`: base64 string (optional, for img2img)
   - Other settings: width, height, steps, seed, sampler, etc. (limited to what hardware supports)
3. Ensure the endpoint can parse base64 images for img2img, matching SD API conventions.
4. Specify response format: image path, timing, and optionally base64-encoded result.
5. Document the payload and response for picker and other clients.

Further Considerations:
1. The response should provide the generated image as base64, this is a network service. (Or, more specifically, it should work exactly as the picker expects.)
3. Consider compatibility with existing picker and SD WebUI clients.
4. Protocol should reflect actual hardware capabilities, not just SD standards. SD standards are to be an inspiration, not a strict requirement.
