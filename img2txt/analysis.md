# Image-to-Text (img2txt) Capability Analysis for Axera AX650N

This document analyzes the feasibility of implementing image-to-text (captioning/interrogation) features using the existing Axera-optimized toolchain and hardware.

## 1. Current Status of the Workspace

The existing models and code are strictly optimized for **Stable Diffusion 1.5 Image Generation**.

*   **Available Components**: CLIP Text Encoder (`sd15_text_encoder_sim.axmodel`), UNet, and VAE.
*   **Missing Components**: The **CLIP Vision Encoder** (the "eyes" of the model) and a **Language Model (LLM/Decoder)** head are not present in the current `.axmodel` set.
*   **Data Flow**: The current pipeline is one-way (Text $\rightarrow$ Image). Image-to-text requires a reverse or multimodal pipeline.

## 2. Compact Model Alternatives

To achieve "CLIP Interrogator" or "BLIP" style functionality on the Raspberry Pi 5 / AX650N, the following compact models are recommended:

| Model | Description | Suitability | Performance (Est.) |
|-------|-------------|-------------|--------------------|
| **CLIP (ViT-L/14)** | Keyword/Tag extraction. | High (A1111 Style) | **Very Fast** (1 pass) |
| **BLIP-Base** | Natural language captioning. | High (A1111 Style) | **Slower** (1 + N passes) |
| **InternVL2-1B** | Verified in [pulsar2-docs](../pulsar2-docs/source/appendix/build_llm.rst). | Very High (Native) | Moderate |
| **MobileVLM V2 1.7B** | Faster/Stronger baseline for mobile. | High (Low RAM) | Moderate |

### MobileVLM V2: Deep Dive & AX650N Suitability
MobileVLM V2 is an improved family of Vision-Language Models (VLMs) that builds on the original MobileVLM architecture. It is specifically designed to be a "Faster and Stronger Baseline" for mobile devices.

*   **Architecture**:
    *   **Vision Tower**: CLIP ViT-L/14-336 (Vision Transformer).
    *   **LDP V2 Projector**: An improved "Lightweight Downsample Projector" that further optimizes visual token reduction.
    *   **LLM Backbone**: Uses "MobileLLaMA" (1.4B parameters for the 1.7B model), which is highly efficient for edge inference.
*   **Key Improvements in V2**:
    *   **Higher Accuracy**: The 1.7B V2 model outperforms the original 1.7B model across all major benchmarks (e.g., MME score of 1302.8 vs 1196.2).
    *   **Deployment Friendly**: Officially supported by `llama.cpp`, indicating a high degree of community optimization for ARM/Mobile architectures.
*   **Suitability for AX650N**:
    *   **The 1.7B "Sweet Spot"**: The 1.7B variant (which uses the 1.4B MobileLLaMA) is the ideal candidate for the Raspberry Pi 5. It provides a sophisticated "assistant" experience (VQA) within a memory footprint that won't starve your Stable Diffusion services.
    *   **NPU Optimization**: The `pulsar2 llm_build` tool is perfectly suited to handle the LDP V2 projector and the transformer layers of MobileLLaMA, allowing for full INT8 acceleration on the AX650N.

### CLIP vs. BLIP: Performance Trade-offs
*   **CLIP (Keyword Style)**: Uses a "one-shot" approach. The image passes through the Vision Encoder once to produce an embedding, which is then compared against a pre-computed keyword database. This is near-instant on the Axera NPU.
*   **BLIP (Caption Style)**: Uses an "autoregressive" approach. After the initial vision pass, the Text Decoder must run sequentially for every word generated. A 15-word caption requires 15+ sequential NPU passes, making it significantly slower than CLIP.

## 3. Axera Conversion Pathway

The conversion process for these models differs from the CNN-based Realistic Vision pipeline. Instead of the standard `pulsar2 build`, the toolchain provides a dedicated LLM/VLM pathway.

### Tooling: `pulsar2 llm_build`
As documented in [pulsar2-docs/source/appendix/build_llm.rst](../pulsar2-docs/source/appendix/build_llm.rst), the `llm_build` command is specifically designed for Transformer-based architectures.

**Key Parameters for VLM:**
*   `--image_size`: Configures the `vision_part` of the VLM (default 224 for BLIP/InternVL).
*   `--hidden_state_type`: Supports `bf16` or `fp16` for transformer layers.
*   `--weight_type`: Supports `int8` (s8) or `int4` (s4) quantization to fit in NPU memory.

## 4. Proposed Implementation Strategy

To implement img2txt without introducing significantly new tooling:

1.  **Export**: Create an export script (similar to [sd15_export_onnx.py](../model_convert/sd15_export_onnx.py)) that uses `transformers.BlipForConditionalGeneration` to export the Vision Tower and Text Decoder to ONNX.
2.  **Compile**: Use the `pulsar2 llm_build` command within the existing Pulsar2 Docker environment to generate `.axmodel` files.
3.  **Inference**:
    *   Use `axengine` to run the Vision Encoder on the NPU.
    *   The resulting image embeddings are then fed into the Text Decoder (also on NPU) to generate the caption tokens.
4.  **Integration**: Add an interrogation capability to the [pi_axera_sd_generator.py](../pi_axera_sd_service/pi_axera_sd_generator.py) service.

### API Standardization Options
To maintain compatibility with existing tools (like SillyTavern or SD mobile apps), two integration paths are available:

*   **A1111 Standard**: Implement `POST /sdapi/v1/interrogate`. This expects a JSON body with `{"image": "base64...", "model": "clip"}` and returns `{"caption": "..."}`.
*   **Native Unified Style**: Extend the existing `/generate` endpoint with a new mode: `{"mode": "interrogate", "image": "base64..."}`.

## 5. Conclusion
While the feature is not "plug-and-play" with the current model set, the **Pulsar2 toolchain already in the workspace** contains the necessary logic (`llm_build`) to support compact vision-language models like BLIP-Base or InternVL.
