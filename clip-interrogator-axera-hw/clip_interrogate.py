"""Standalone CLIP zero-shot image interrogator for Axera AX650N.

Loads the compiled clip_base_vision.axmodel / clip_base_text.axmodel and runs
zero-shot classification against a fixed or custom vocabulary of tags.
"""
import hashlib
import os
import time

import axengine
import numpy as np
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VISION_MODEL = os.path.join(SCRIPT_DIR, "clip_base_vision.axmodel")
DEFAULT_TEXT_MODEL = os.path.join(SCRIPT_DIR, "clip_base_text.axmodel")
DEFAULT_CACHE_DIR = os.path.join(SCRIPT_DIR, "tags")
TOKENIZER_DIR = "openai/clip-vit-base-patch32"  # fetched from Hugging Face Hub on first run


class CLIPInterrogator:
    def __init__(self, vision_model_path=DEFAULT_VISION_MODEL, text_model_path=DEFAULT_TEXT_MODEL, cache_dir=DEFAULT_CACHE_DIR):
        print(f"Loading CLIP Vision Model: {vision_model_path}")
        self.vision_session = axengine.InferenceSession(vision_model_path)
        print(f"Loading CLIP Text Model: {text_model_path}")
        self.text_session = axengine.InferenceSession(text_model_path)

        print(f"Loading CLIP Model for projections and tokenizer: {TOKENIZER_DIR}")
        self.model = CLIPModel.from_pretrained(TOKENIZER_DIR)
        self.processor = CLIPProcessor.from_pretrained(TOKENIZER_DIR)
        self.tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_DIR)

        # Projection weights aren't part of the exported axmodel, so apply them on CPU
        self.visual_projection = self.model.visual_projection.weight.detach().numpy().T
        self.text_projection = self.model.text_projection.weight.detach().numpy().T
        del self.model

        self.cached_embeddings = None
        self.cached_labels = None
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def precompute_text_embeddings(self, labels):
        """Compute (and cache to disk) text embeddings for a vocabulary of labels.

        The cache file name is a hash of the labels + prompt template, so any
        change to the vocabulary automatically invalidates the cache.
        """
        template = "a photo of a {}"
        labels_str = ",".join(labels) + "|" + template
        labels_hash = hashlib.sha256(labels_str.encode()).hexdigest()
        cache_file = None

        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"clip_cache_{labels_hash}.npy")
            if os.path.exists(cache_file):
                print(f"Loading cached embeddings from {cache_file}...")
                self.cached_embeddings = np.load(cache_file)
                self.cached_labels = labels
                return

        print(f"Pre-computing embeddings for {len(labels)} labels...")
        text_embs = []
        start_time = time.time()

        for i, label in enumerate(labels):
            if i % 100 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = (len(labels) - i) / rate
                print(f"  Processed {i}/{len(labels)} labels ({rate:.1f} labels/sec, ~{remaining:.0f}s remaining)")

            templated_label = template.format(label)
            text_inputs = self.tokenizer(templated_label, padding="max_length", max_length=77, truncation=True, return_tensors="np")
            t_out = self.text_session.run(None, {"input_ids": text_inputs["input_ids"].astype(np.int32)})
            t_emb_raw = t_out[1]  # pooler_output, shape (1, 512)
            t_emb = t_emb_raw @ self.text_projection
            text_embs.append(t_emb)

        self.cached_embeddings = np.concatenate(text_embs, axis=0)
        self.cached_embeddings /= np.linalg.norm(self.cached_embeddings, axis=-1, keepdims=True)
        self.cached_labels = labels

        if cache_file:
            print(f"Saving embeddings to cache: {cache_file}")
            np.save(cache_file, self.cached_embeddings)

        print(f"Embeddings pre-computed for {len(labels)} labels in {time.time() - start_time:.2f}s.")

    def interrogate(self, image, labels=None):
        """Return (scores, labels) - cosine similarity of the image against each label."""
        inputs = self.processor(images=image, return_tensors="np")
        v_out = self.vision_session.run(None, {"pixel_values": inputs["pixel_values"].astype(np.float32)})
        img_emb_raw = v_out[1]  # pooler_output, shape (1, 768)
        img_emb = img_emb_raw @ self.visual_projection
        img_emb /= np.linalg.norm(img_emb, axis=-1, keepdims=True)

        if labels is None:
            if self.cached_embeddings is None:
                raise ValueError("No labels provided and no cached embeddings found.")
            text_embs_norm = self.cached_embeddings
            current_labels = self.cached_labels
        elif labels == self.cached_labels:
            text_embs_norm = self.cached_embeddings
            current_labels = self.cached_labels
        else:
            text_embs = []
            for label in labels:
                templated_label = f"a photo of a {label}"
                text_inputs = self.tokenizer(templated_label, padding="max_length", max_length=77, truncation=True, return_tensors="np")
                t_out = self.text_session.run(None, {"input_ids": text_inputs["input_ids"].astype(np.int32)})
                t_emb_raw = t_out[1]
                t_emb = t_emb_raw @ self.text_projection
                text_embs.append(t_emb)
            text_embs = np.concatenate(text_embs, axis=0)
            text_embs_norm = text_embs / np.linalg.norm(text_embs, axis=-1, keepdims=True)
            current_labels = labels

        scores = img_emb @ text_embs_norm.T
        return scores, current_labels

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def classify_categories(self, image, schema):
        """Forced-choice classification: {"category": ["opt1", "opt2", ...], ...} -> winner + confidence."""
        inputs = self.processor(images=image, return_tensors="np")
        v_out = self.vision_session.run(None, {"pixel_values": inputs["pixel_values"].astype(np.float32)})
        img_emb_raw = v_out[1]
        img_emb = img_emb_raw @ self.visual_projection
        img_emb /= np.linalg.norm(img_emb, axis=-1, keepdims=True)

        results = {}
        label_to_idx = {}
        if self.cached_labels:
            label_to_idx = {label: i for i, label in enumerate(self.cached_labels)}

        for category, labels in schema.items():
            cat_embs = []
            valid_labels = []
            for label in labels:
                if label in label_to_idx:
                    cat_embs.append(self.cached_embeddings[label_to_idx[label]])
                    valid_labels.append(label)
                else:
                    templated_label = f"a photo of a {label}"
                    text_inputs = self.tokenizer(templated_label, padding="max_length", max_length=77, truncation=True, return_tensors="np")
                    t_out = self.text_session.run(None, {"input_ids": text_inputs["input_ids"].astype(np.int32)})
                    t_emb_raw = t_out[1]
                    t_emb = t_emb_raw @ self.text_projection
                    t_emb_norm = t_emb / np.linalg.norm(t_emb, axis=-1, keepdims=True)
                    cat_embs.append(t_emb_norm.flatten())
                    valid_labels.append(label)

            if not cat_embs:
                continue

            cat_embs = np.stack(cat_embs)
            scores = (img_emb @ cat_embs.T).flatten()
            probs = self._softmax(scores * 100.0)  # CLIP-standard temperature scaling
            winner_idx = np.argmax(probs)
            results[category] = {
                "winner": valid_labels[winner_idx],
                "confidence": float(probs[winner_idx]),
            }

        return results, img_emb
