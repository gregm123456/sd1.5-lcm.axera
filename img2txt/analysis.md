# Image-to-Text (img2txt) Capability Analysis for Axera AX650N

This document analyzes the feasibility of implementing image-to-text (captioning/interrogation) features using the existing Axera-optimized toolchain and hardware.

## 1. Current Status of the Workspace

**STATUS: IMPLEMENTED**

The image-to-text capability has been fully integrated into the `pi_axera_sd_service`.

*   **Available Components**: 
    *   CLIP Text Encoder (`sd15_text_encoder_sim.axmodel`)
    *   CLIP Vision Encoder (`clip_base_vision.axmodel`) - **ADDED**
    *   ImageNet-21K Embedding Cache (20,101 terms)
*   **Data Flow**: Bidirectional. Supports both Image Generation (Text $\rightarrow$ Image) and Interrogation (Image $\rightarrow$ Text/Categories).

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
*   **CLIP (Keyword Style)**: **IMPLEMENTED**. Uses a "one-shot" approach with a pre-computed embedding cache. This is near-instant on the Axera NPU and supports dynamic categorization via Softmax.
*   **BLIP (Caption Style)**: Future consideration for natural language descriptions.

## 3. Axera Conversion Pathway

The conversion process for these models differs from the CNN-based Realistic Vision pipeline. Instead of the standard `pulsar2 build`, the toolchain provides a dedicated LLM/VLM pathway.

### Tooling: `pulsar2 llm_build`
As documented in [pulsar2-docs/source/appendix/build_llm.rst](../pulsar2-docs/source/appendix/build_llm.rst), the `llm_build` command is specifically designed for Transformer-based architectures.

**Key Parameters for VLM:**
*   `--image_size`: Configures the `vision_part` of the VLM (default 224 for BLIP/InternVL).
*   `--hidden_state_type`: Supports `bf16` or `fp16` for transformer layers.
*   `--weight_type`: Supports `int8` (s8) or `int4` (s4) quantization to fit in NPU memory.

## 4. Implementation Strategy (COMPLETED)

The implementation follows the "CLIP (Keyword Style)" approach:

1.  **Vision Pass**: The image is encoded once using `clip_base_vision.axmodel` on the NPU.
2.  **Structured Slicing**: The resulting embedding is compared against specific "buckets" of pre-computed text embeddings in RAM.
3.  **Categorized Softmax**: Probabilities are calculated within each category to provide high-confidence winners.
4.  **Integration**: Fully integrated into [pi_axera_sd_generator.py](../pi_axera_sd_service/pi_axera_sd_generator.py).

## 5. Conclusion
The AX650N now supports high-speed, structured image interrogation, enabling "Form Filler" applications for demographics, scene analysis, and more.
