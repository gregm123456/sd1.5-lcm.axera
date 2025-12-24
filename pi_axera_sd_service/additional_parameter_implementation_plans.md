# Implementation Plans for Additional Parameters

This document outlines the technical implementation plans for adding `strength`, `steps`, and `fit` parameters to the Pi Axera SD Service.

## 1. Denoising Strength (`denoising_strength`) for img2img

**Goal:** Allow users to control how much the input image is modified.
**Standard Parameter Name:** `denoising_strength` (compatible with SD WebUI API).
**Current State:** Hardcoded to start at timestep 499 (approx 0.5 strength).

### Technical Details
The model uses a fixed schedule of timesteps: `[999, 759, 499, 259]`.
The `time_input_txt2img.npy` file contains the time embeddings for all 4 steps (indices 0-3).
The `time_input_img2img.npy` file currently contains embeddings for indices 2 and 3 (499 and 259).

To implement variable strength, we will switch to using `time_input_txt2img.npy` as the master source for time embeddings and select the starting timestep based on the requested strength.

### Mapping Logic (Pinning)
We will accept any float value between 0.0 and 1.0 and map it to the nearest supported discrete step count.

| Input Range (`denoising_strength`) | Effective Strength | Starting Timestep | Index in Master NPY | Inference Steps | Description |
|------------------------------------|--------------------|-------------------|---------------------|-----------------|-------------|
| 0.00 - 0.35                        | **0.25**           | 259               | 3                   | 1               | Minimal changes, very fast |
| 0.36 - 0.60                        | **0.50**           | 499               | 2                   | 2               | Balanced (Current Default) |
| 0.61 - 0.85                        | **0.75**           | 759               | 1                   | 3               | Strong modification |
| 0.86 - 1.00                        | **1.00**           | 999               | 0                   | 4               | Complete reimagining |

### Implementation Steps
1.  Modify `pi_axera_sd_generator.py` to load `time_input_txt2img.npy` as the primary source for both modes.
2.  Update `generate_img2img` signature to accept `strength` (float, default 0.5).
3.  Update `/generate` endpoint to look for `denoising_strength` (or `strength` as alias) in the request body.
4.  Inside `generate_img2img`:
    *   Select the `start_index` based on the table above.
    *   Slice the `DEFAULT_TIMESTEPS` array: `timesteps = DEFAULT_TIMESTEPS[start_index:]`.
    *   Slice the time embeddings: `time_input = time_input_txt2img[start_index:]`.
    *   Use the first timestep of this new slice for the initial noise addition calculation.

---

## 2. Inference Steps (`steps`) for txt2img

**Goal:** Allow users to trade quality for speed by using fewer inference steps.
**Current State:** Fixed at 4 steps (`[999, 759, 499, 259]`).

### Technical Details
LCM models are robust at low step counts. We can allow users to request 1-4 steps. Since the time embeddings are pre-calculated for specific timestep values, we cannot generate arbitrary schedules (e.g., 20 steps). We must select a subset of the available 4 steps.

### Mapping Logic
| Requested Steps | Timesteps Used | Logic |
|-----------------|----------------|-------|
| 1               | `[999]`        | Fastest possible generation |
| 2               | `[999, 499]`   | Balanced speed/quality |
| 3               | `[999, 759, 259]` | Higher quality |
| 4 (Default)     | `[999, 759, 499, 259]` | Maximum quality |

*Note: The exact subset for 2 and 3 steps may need empirical testing to find the visually best combination. Alternatively, we can just truncate from the end or start, but skipping steps (striding) is usually better for LCM.*

### Implementation Steps
1.  Update `generate_txt2img` signature to accept `steps` (int, default 4).
2.  Inside `generate_txt2img`:
    *   Validate `steps` is between 1 and 4.
    *   Select the subset of `DEFAULT_TIMESTEPS` and corresponding rows from `time_input_txt2img`.
    *   Run the loop over this subset.

---

## 3. Resize Mode (`resize_mode`) for img2img

**Goal:** Handle input images that are not 512x512 without distorting them.
**Standard Parameter Name:** `resize_mode` (compatible with SD WebUI API).
**Current State:** "Stretch" (0) - blindly resizes input to 512x512.

### Technical Details
This is a pre-processing step using PIL before the image is passed to the model.

### Modes
| Value | Name | Description |
|-------|------|-------------|
| **0** (Default) | **Just resize** | Resize directly to 512x512. Ignores aspect ratio. (Stretch) |
| **1** | **Crop and resize** | Resize so the smallest dimension is 512, keeping aspect ratio, then center crop to 512x512. Preserves content shape but loses edges. (Cover) |
| **2** | **Resize and fill** | Resize so the largest dimension is 512, keeping aspect ratio, then pad the rest with background (black) to reach 512x512. Preserves all content but adds borders. (Contain) |

### Implementation Steps
1.  Update `/generate` endpoint to accept `resize_mode` parameter (int: 0, 1, 2).
2.  Implement a helper function `preprocess_image(image, mode)`:
    *   **0**: `image.resize((512, 512))`
    *   **1**: Calculate aspect ratio, resize, then `image.crop(...)`.
    *   **2**: Calculate aspect ratio, resize, create new 512x512 image, `new_img.paste(image, ...)`.
3.  Apply this helper before passing the image to `generate_img2img`.
