---
license: mit
tags:
- clip
- zero-shot-image-classification
- axera
- ax650n
- npu
- raspberry-pi
---

# CLIP Zero-Shot Image Interrogator (Axera Optimized)

This repository contains a compiled derivative of **CLIP ViT-B/32** (`openai/clip-vit-base-patch32`), optimized for high-speed zero-shot image classification / "interrogation" on Axera hardware (AX650N / LLM8850).

**Note**: This repository is a compilation of existing open-source work. The maintainer of this repository is not the original creator of the CLIP model, but has performed the export, quantization, and compilation for the Axera NPU platform.

## Model Description

- **Base Model**: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- **Task**: Zero-shot image classification ("forced-choice" comparison between an image embedding and a set of text label embeddings — not caption generation)
- **Target Hardware**: Axera AX650N / LLM8850 (NPU3)
- **Format**: Axera `.axmodel` (compiled via Pulsar2 v5.1, U16 quantization)
- **Components**: Vision encoder (`clip_base_vision.axmodel`) and text encoder (`clip_base_text.axmodel`), compiled separately. Similarity is computed on CPU as a simple dot product after each encoder pass.

## Included Assets

| File | Size | Purpose |
|------|------|---------|
| `clip_base_vision.axmodel` | ~94 MB | Compiled vision encoder |
| `clip_base_text.axmodel` | ~136 MB | Compiled text encoder |
| `tags/imagenet_21k_labels_clean.txt` | ~230 KB | Curated ImageNet-21K vocabulary (20,101 terms) |
| `tags/clip_cache_<hash>.npy` | ~40 MB | Precomputed, L2-normalized text embeddings for the vocabulary above |

The tag list and embedding cache are included so you can run zero-shot classification immediately, without waiting ~30-40s to precompute embeddings for 20k terms on first use. This is optional: `clip_interrogate.py` will happily precompute embeddings for **any** custom vocabulary you pass in — the cache file name is a SHA-256 hash of the label list + prompt template, so changing the vocabulary automatically invalidates the old cache and recomputes as needed. Feel free to delete the `tags/` folder if you plan to always supply your own label sets.

## Performance

Measured on Raspberry Pi 5 + Axera AX650N M.2 Accelerator:

| Operation | Hardware | Latency |
|-----------|----------|---------|
| Vision Encoding | Axera NPU | ~30-50 ms (once per image) |
| Text Encoding | Axera NPU | <5 ms (once per new label, cached thereafter) |
| Similarity Scoring | CPU | <1 ms (per image/label pair) |

Because the image only needs a single NPU pass regardless of how many labels or categories you compare it against, this scales well to large vocabularies or many simultaneous "form fields" (e.g. classifying gender, age bracket, and setting all from one image pass). See [clip_constrained_output.md](https://github.com/AXERA-TECH/sd1.5-lcm.axera/blob/main/img2txt/clip_constrained_output.md) in the parent project for a deeper analysis of this "constrained output" approach and known failure modes at large vocabulary sizes.

## Runtime Dependencies

```bash
pip install axengine transformers torch pillow numpy
```

**Important**: on first run (and on every restart unless cached), this loads `openai/clip-vit-base-patch32` from the Hugging Face Hub to obtain the tokenizer, image processor, and the visual/text projection matrices (these projections aren't baked into the exported `.axmodel` files, since they're applied on CPU after each NPU pass). This requires outbound internet access and downloads ~1.2GB into `~/.cache/huggingface/hub` the first time. Anonymous (no `HF_TOKEN`) requests are subject to lower Hugging Face rate limits, which can occasionally cause a transient failure on a small file (e.g. `preprocessor_config.json`) even after the large weight files succeed — if this happens, simply retry; most of the cache will already be in place. Setting an `HF_TOKEN` environment variable reduces the odds of this.

## Usage

```bash
python run_clip_interrogate.py path/to/image.png
```

Or programmatically:

```python
from PIL import Image
from clip_interrogate import CLIPInterrogator

ci = CLIPInterrogator()  # loads clip_base_vision.axmodel / clip_base_text.axmodel from this directory
ci.precompute_text_embeddings(open("tags/imagenet_21k_labels_clean.txt").read().splitlines())

image = Image.open("photo.jpg").convert("RGB")
scores, labels = ci.interrogate(image)
```

For "forced-choice" structured classification (e.g. a demographics form), use `classify_categories`:

```python
results, _ = ci.classify_categories(image, {
    "gender": ["man", "woman", "boy", "girl"],
    "hair_color": ["pink", "blonde", "brown", "black"],
})
# {"gender": {"winner": "woman", "confidence": 0.94}, "hair_color": {"winner": "pink", "confidence": 0.99}}
```

## Conversion Workflow Summary

1. **Export**: `CLIPModel.vision_model` / `text_model` exported separately to ONNX (`opset_version=14`, `attn_implementation="eager"`, static batch size of 1).
2. **Calibration**: ~50 sample images (vision) and 50 random token sequences (text) used for MinMax calibration.
3. **Quantization**: Compiled using Pulsar2 with U16 precision across all layers.

Key lessons learned during conversion (useful if you want to recompile with a different CLIP variant or vocabulary):
- The Axera compiler does not support dynamic batch axes (e.g. ONNX exports with a symbolic batch dim like `s99` will fail to build) — always export with a fixed batch size.
- Use `opset_version=14`; higher opsets can introduce operators (e.g. `ScaledDotProductAttention`) that the compiler struggles with.
- Use `attn_implementation="eager"` when loading the model from `transformers` to avoid SDPA-related ONNX export issues.
- Text encoder calibration data should be `int64` to match the model's input, but the Pulsar2 `src_dtype` config should be `S32`.

## Credits and Citations

- **CLIP**: Developed by [OpenAI](https://huggingface.co/openai/clip-vit-base-patch32) ([paper](https://arxiv.org/abs/2103.00020)).
- **Conversion Tools**: The [AXERA-TECH/sd1.5-lcm.axera](https://github.com/AXERA-TECH/sd1.5-lcm.axera) project (forked from [BUG1989](https://github.com/BUG1989/sd1.5-lcm.axera)).

**Optimization & Compilation**: The artifacts in this repository were exported and compiled for the Axera NPU using the Pulsar2 toolchain.

## Additional Resources

- **Project Repository**: [AXERA-TECH/sd1.5-lcm.axera](https://github.com/AXERA-TECH/sd1.5-lcm.axera) — image generation (txt2img/img2img) service this interrogator is designed to complement.
- **Toolchain Documentation**: [Axera Pulsar2 Docs](https://pulsar2-docs.readthedocs.io/zh-cn/latest/)

## Licensing and Restrictions

The CLIP model itself is released under the [MIT License](LICENSE) by OpenAI. Axera-specific conversion tooling referenced by this repository is licensed under [BSD 3-Clause](LICENSE_AXERA). See [Disclaimer.md](Disclaimer.md) for usage caveats, particularly around demographic/sensitive classification categories.
