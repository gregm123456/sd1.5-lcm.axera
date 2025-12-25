import os
import sys
import argparse
import numpy as np
from PIL import Image
from clip_interrogate import CLIPInterrogator

# Curated list of tags for general image interrogation
DEFAULT_TAGS = [
    "man", "woman", "boy", "girl", "person", "group of people",
    "outdoor", "indoor", "nature", "city", "landscape", "interior",
    "day", "night", "sunset", "sunrise",
    "highly detailed", "masterpiece", "sharp focus", "blurry",
    "digital art", "photograph", "painting", "sketch", "anime",
    "forest", "mountain", "beach", "ocean", "street", "building",
    "cat", "dog", "horse", "bird", "animal",
    "car", "bicycle", "plane", "boat",
    "food", "drink", "fruit", "vegetable",
    "blue sky", "cloudy", "rainy", "snowy", "sunny",
    "red", "blue", "green", "yellow", "black", "white", "colorful"
]

def main():
    parser = argparse.ArgumentParser(description="Interrogate an image using CLIP on Axera NPU")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("--threshold", type=float, default=0.22, help="Similarity threshold (default: 0.22)")
    parser.add_argument("--top_k", type=int, default=10, help="Max number of tags to return (default: 10)")
    
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        sys.exit(1)

    print(f"Initializing CLIP Interrogator...")
    ci = CLIPInterrogator()

    print(f"Processing image: {args.image_path}")
    try:
        img = Image.open(args.image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)

    print(f"Interrogating with {len(DEFAULT_TAGS)} tags...")
    scores = ci.interrogate(img, DEFAULT_TAGS)[0]

    # Filter by threshold and sort by score
    results = []
    for i, score in enumerate(scores):
        if score >= args.threshold:
            results.append((DEFAULT_TAGS[i], score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Limit to top_k
    results = results[:args.top_k]

    if not results:
        print("No tags found above threshold.")
    else:
        print("\nDetected Tags:")
        for tag, score in results:
            print(f"  - {tag} ({score:.4f})")
        
        print("\nCaption-style output:")
        print(", ".join([r[0] for r in results]))

if __name__ == "__main__":
    main()
