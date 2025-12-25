# Structured Interrogation & Demographics Plan

This document outlines the strategy for implementing a "Structured Interrogation" system on the Axera AX650N, leveraging the existing CLIP ImageNet-21K embedding cache to provide high-confidence, enum-based classification (e.g., Demographics).

## 1. The Problem: "Flat" vs. "Structured"
Currently, our interrogation system uses a **Flat Taxonomy** approach. All 20,101 terms compete in a single "race."
*   **Issue**: Visually dominant features (like "pink hair") can drown out high-level subjects (like "woman") because the raw similarity scores for specific colors/textures often exceed those of broad categories.
*   **Example**: A portrait of a woman with pink hair might return `fringed pink` (flower) as the top result because the "pinkness" is mathematically sharper than the "woman-ness."

## 2. The Solution: Categorized Softmax
Instead of one giant race, we divide the interrogation into **independent categories**.

### How it works:
1.  **Category Definition**: We define specific "buckets" (e.g., `Gender`, `Age`, `Subject`).
2.  **Embedding Slicing**: We extract the existing embeddings for the terms in each bucket from our RAM cache.
3.  **Softmax Normalization**: We apply the Softmax function to the scores *within each bucket*.
    *   This forces the scores in that bucket to sum to 1.0 (100%).
    *   It effectively "mutes" the noise from outside the bucket.

### Example:
**Image**: Woman with pink hair.
**Bucket: Gender** (`["man", "woman"]`)
*   Raw Scores: `man: 0.15`, `woman: 0.28`
*   **Softmax Result**: `man: 6%`, `woman: 94%` (Clear Winner)

## 3. Proposed API Endpoint: `/sdapi/v1/interrogate/structured`
We will implement a new endpoint that accepts a "Schema" of categories.

### Request Format:
```json
{
  "image": "base64_data...",
  "categories": {
    "subject": ["person", "animal", "vehicle", "landscape"],
    "gender": ["man", "woman", "boy", "girl"],
    "hair_color": ["pink", "blonde", "brown", "black"]
  }
}
```

### Response Format:
```json
{
  "results": {
    "subject": {"winner": "person", "confidence": 0.98},
    "gender": {"winner": "woman", "confidence": 0.94},
    "hair_color": {"winner": "pink", "confidence": 0.99}
  },
  "general_tags": ["portrait", "face", "long hair"] 
}
```

## 4. Technical Advantages
*   **Zero Re-encoding**: Uses the existing `clip_cache_*.npy` files. No NPU time is wasted re-computing embeddings.
*   **Sub-Millisecond Latency**: Slicing and Softmax are simple CPU matrix operations.
*   **Confidence Metrics**: The "confidence" value (0.0 to 1.0) allows the application to handle ambiguous cases (e.g., "If confidence < 0.6, mark as 'Unknown'").
*   **Dynamic Schemas**: Users can define new categories on-the-fly as long as the terms exist in our 20,101-word taxonomy.

## 5. Implementation Context
### Key Files and Components:
*   **Main Service**: `pi_axera_sd_service/pi_axera_sd_generator.py`
*   **CLIPInterrogator Class**: Located in the main service file (lines ~85-200)
*   **Existing Endpoint**: `/sdapi/v1/interrogate` (lines ~730-770) - returns flat Top-K results
*   **Embedding Cache**: `models/clip_cache/clip_cache_*.npy` (40MB, 20,101 x 512 matrix, pre-loaded in RAM)
*   **Label Source**: `pi_axera_sd_service/imagenet_21k_labels_clean.txt` (20,101 unique English terms)
*   **Prompt Template**: All labels are encoded as `"a photo of a {label}"` (defined in `precompute_text_embeddings`)

### Current CLIPInterrogator Methods:
*   `precompute_text_embeddings(labels)`: Encodes all labels and saves to cache
*   `interrogate(image, labels=None)`: Returns raw similarity scores for all cached labels
    *   If `labels=None`, uses the cached 20,101-term embeddings
    *   Returns: `(scores, current_labels)` tuple

### Implementation Strategy:
The new structured endpoint should:
1.  Accept a schema with category names and term lists
2.  For each category, look up the indices of those terms in `self.cached_labels`
3.  Slice the similarity scores for just those indices
4.  Apply Softmax to get probabilities
5.  Return the winner + confidence for each category

**Important**: Terms that don't exist in the 20,101-word vocabulary will need on-the-fly encoding (fallback path already exists in `interrogate` method).

## 6. Implementation Steps
1.  **Update `CLIPInterrogator`**: Add a `classify_categories(image, schema)` method.
2.  **Add Flask Route**: Implement `/sdapi/v1/interrogate/structured`.
3.  **Validation**: Test against the "Pink Hair" portrait to ensure `woman` is correctly identified as the primary subject.

## 7. Rich Category Examples (Reference: sample_texts.json)
The system is designed to handle complex, multi-dimensional classification. A sample rich set of categories can be found in [img2txt/sample_texts.json](img2txt/sample_texts.json), which includes:
*   **Sex/Gender**: Agender, Bigender, Cisgender, Female, Male, Non-binary, etc.
*   **Age**: Young Adult, Adult, Middle-aged, Senior, Elderly, etc.
*   **Socioeconomics**: Working Class, Middle Class, Wealthy, Affluent, etc.
*   **Politics**: Conservative, Liberal, Libertarian, Progressive, etc.
*   **Race**: African, Asian, Black, Hispanic, White, etc.
*   **Religion**: Buddhist, Christian, Hindu, Jewish, Muslim, etc.

These categories can be swapped or modified easily and often, as the system dynamically slices the existing 20,101-term embedding cache to resolve them.

## 8. Future Expansion: "The Demographics Form"
This architecture directly supports the goal of a "Form Filler" for security or analytics applications, where 100% adherence to a specific set of enums is required.
