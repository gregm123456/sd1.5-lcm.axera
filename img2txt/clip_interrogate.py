import axengine
import numpy as np
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel
import torch

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VISION_MODEL = os.path.join(SCRIPT_DIR, "../models/clip_base_vision.axmodel")
TEXT_MODEL = os.path.join(SCRIPT_DIR, "../models/clip_base_text.axmodel")
TOKENIZER_DIR = "openai/clip-vit-base-patch32" # Use standard HF tokenizer

class CLIPInterrogator:
    def __init__(self):
        print(f"Loading Vision Model: {VISION_MODEL}")
        self.vision_session = axengine.InferenceSession(VISION_MODEL)
        print(f"Loading Text Model: {TEXT_MODEL}")
        self.text_session = axengine.InferenceSession(TEXT_MODEL)
        
        print(f"Loading CLIP Model for projections and tokenizer: {TOKENIZER_DIR}")
        self.model = CLIPModel.from_pretrained(TOKENIZER_DIR)
        self.processor = CLIPProcessor.from_pretrained(TOKENIZER_DIR)
        self.tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_DIR)
        
        # Extract projection weights for CPU application
        self.visual_projection = self.model.visual_projection.weight.detach().numpy().T
        self.text_projection = self.model.text_projection.weight.detach().numpy().T
        
        # We can delete the model to save RAM if needed, but keeping it for now
        # del self.model

    def interrogate(self, image, labels):
        # 1. Image Pass
        inputs = self.processor(images=image, return_tensors="np")
        # The model returns [last_hidden_state, pooler_output]
        # We want pooler_output (index 1)
        v_out = self.vision_session.run(None, {"pixel_values": inputs['pixel_values'].astype(np.float32)})
        img_emb_raw = v_out[1] # Shape (1, 768)
        
        # Apply visual projection
        img_emb = img_emb_raw @ self.visual_projection # Shape (1, 512)
        
        # 2. Text Pass (Loop because model is static batch size 1)
        text_embs = []
        for label in labels:
            text_inputs = self.tokenizer(label, padding="max_length", max_length=77, truncation=True, return_tensors="np")
            t_out = self.text_session.run(None, {"input_ids": text_inputs['input_ids'].astype(np.int32)})
            t_emb_raw = t_out[1] # Shape (1, 512)
            
            # Apply text projection
            t_emb = t_emb_raw @ self.text_projection # Shape (1, 512)
            text_embs.append(t_emb)
        
        text_embs = np.concatenate(text_embs, axis=0)
        
        # 3. Similarity (CPU)
        # Normalize embeddings
        img_emb /= np.linalg.norm(img_emb, axis=-1, keepdims=True)
        text_embs /= np.linalg.norm(text_embs, axis=-1, keepdims=True)
        
        # Calculate cosine similarity
        # img_emb shape: (1, 512)
        # text_embs shape: (num_labels, 512)
        scores = img_emb @ text_embs.T
        return scores

# Example Usage
if __name__ == "__main__":
    ci = CLIPInterrogator()
    
    # Use a sample image from assets
    sample_img_path = os.path.join(SCRIPT_DIR, "../assets/txt2img_output_axe.png")
    if not os.path.exists(sample_img_path):
        print(f"Sample image not found at {sample_img_path}, creating a dummy one.")
        img = Image.new('RGB', (224, 224), color = (73, 109, 137))
    else:
        print(f"Using sample image: {sample_img_path}")
        img = Image.open(sample_img_path).convert("RGB")
        
    labels = ["a photo of a cat", "a photo of a dog", "a landscape", "a person", "an astronaut in space"]
    print(f"Interrogating with labels: {labels}")
    scores = ci.interrogate(img, labels)
    
    top_idx = np.argmax(scores[0])
    print(f"Top Label: {labels[top_idx]} (Score: {scores[0][top_idx]:.4f})")
    
    print("\nAll scores:")
    for label, score in zip(labels, scores[0]):
        print(f"  {label}: {score:.4f}")
