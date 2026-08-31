# Hugging Face Repository Plan: CLIP Zero-Shot Image Interrogator (Axera Optimized)

This document outlines the plan for creating and publishing a Hugging Face model repository for the Axera-optimized CLIP interrogation models used by `pi_axera_sd_service`.

## 1. Repository Identity
- **Proposed Name**: `clip-interrogator-axera-hw`
- **Local Build Directory**: `./clip-interrogator-axera-hw/`
- **License**: [MIT](clip-interrogator-axera-hw/LICENSE) (inherited from OpenAI CLIP) + [BSD 3-Clause](clip-interrogator-axera-hw/LICENSE_AXERA) for Axera-specific conversion tooling

## 2. Repository Structure

```text
clip-interrogator-axera-hw/
├── README.md                     # Model card
├── LICENSE                       # MIT (CLIP base model)
├── LICENSE_AXERA                 # BSD 3-Clause (conversion tools)
├── Disclaimer.md                 # Axera/CLIP-specific usage disclaimer
├── clip_base_vision.axmodel      # Compiled Vision Encoder (~94MB)
├── clip_base_text.axmodel        # Compiled Text Encoder (~136MB)
├── clip_interrogate.py           # Standalone inference wrapper
├── run_clip_interrogate.py       # CLI example
├── tags/
│   ├── imagenet_21k_labels_clean.txt   # 20,101-term vocabulary (opt-in, included)
│   └── clip_cache_<hash>.npy           # Precomputed text embeddings for the above (opt-in, included, ~40MB)
└── assets/                       # Sample classification outputs (to be added)
```

**Opt-in inclusion decision**: the precomputed tag vocabulary and embedding cache are included by default in this repository so users get instant classification out of the box, without the ~30-40s precompute step for 20k terms. This is documented in the README as optional/removable, along with how the hash-validated cache mechanism in `clip_interrogate.py` automatically regenerates the cache for a custom vocabulary.

## 3. Model Card (README.md) Content

Implemented in [clip-interrogator-axera-hw/README.md](clip-interrogator-axera-hw/README.md):
- Model Description (CLIP ViT-B/32, Pulsar2 U16 quant, AX650N/NPU3, zero-shot classification use case)
- Included Assets (axmodel files + optional tag cache, hash-validation behavior)
- Performance (vision ~30-50ms/image, text <5ms/term one-time, similarity <1ms CPU)
- Runtime Dependency Note (`axengine` + `transformers`; first-run Hugging Face download of `openai/clip-vit-base-patch32` for tokenizer/processor/projection weights; `HF_TOKEN` recommendation to avoid rate-limit-related transient failures)
- Usage Instructions (CLI + programmatic examples, including forced-choice `classify_categories`)
- Conversion Workflow Summary + lessons learned (static batch size, opset 14, eager attention, calibration dtypes)
- Credits and Citations
- Licensing and Restrictions

## 4. Action Checklist

- [x] Create local `clip-interrogator-axera-hw/` directory
- [x] Copy `clip_base_vision.axmodel` + `clip_base_text.axmodel` from `pi_axera_sd_service/models/`
- [x] Copy/adapt `clip_interrogate.py` + `run_clip_interrogate.py` (paths adjusted for new repo root; upgraded to the fuller `CLIPInterrogator` class from `pi_axera_sd_generator.py`, including cache/precompute and `classify_categories`)
- [x] Copy `imagenet_21k_labels_clean.txt` + `clip_cache_<hash>.npy` into `tags/` (opt-in inclusion, per decision above)
- [x] Write `README.md` per model card outline above
- [x] Add `LICENSE` + `LICENSE_AXERA` + `Disclaimer.md`
- [ ] Initialize git repo, push to new Hugging Face repo `clip-interrogator-axera-hw` (**requires Hugging Face account/auth — not yet done**)
- [ ] Add as a second git submodule in `sd1.5-lcm.axera` (alongside `realistic-vision-v6-axera-hw`), and update `pi_axera_sd_service` setup docs to reference the submodule as the canonical source of these two `.axmodel` files instead of "copy from a prior deployment" (**depends on previous step**)

## 5. Remaining Steps (need your input)

Publishing to Hugging Face and wiring up the submodule require your Hugging Face credentials and a decision on the exact remote URL, so these weren't run automatically:

1. Create the repo at `https://huggingface.co/<your-username>/clip-interrogator-axera-hw` (via the HF website or `huggingface-cli repo create`).
2. From `clip-interrogator-axera-hw/`: `git init`, `git lfs install`, `git lfs track "*.axmodel" "*.npy"`, commit, add the HF remote, and push.
3. From the `sd1.5-lcm.axera` repo root: `git submodule add https://huggingface.co/<your-username>/clip-interrogator-axera-hw clip-interrogator-axera-hw` (note: this will conflict with the local directory already present at that path — the existing local copy would need to be removed first and re-cloned as a submodule, or the submodule added under a different local path).
4. Update `pi_axera_sd_service/README.md` and `RASPBERRY_PI_SETUP.md` to reference the submodule path for `clip_base_vision.axmodel` / `clip_base_text.axmodel` instead of the current "copy from a prior deployment" language.
