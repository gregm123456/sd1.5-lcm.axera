import argparse
import os
import sys

from PIL import Image

from clip_interrogate import CLIPInterrogator, DEFAULT_CACHE_DIR

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TAGS_FILE = os.path.join(SCRIPT_DIR, "tags", "imagenet_21k_labels_clean.txt")

# Small curated fallback list, used only if the bundled tag file can't be found
FALLBACK_TAGS = [
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
    "red", "blue", "green", "yellow", "black", "white", "colorful",
]


def load_tags(tags_file):
    if tags_file and os.path.exists(tags_file):
        with open(tags_file, "r") as f:
            tags = [line.strip() for line in f if line.strip()]
        if tags:
            return tags
    print(f"Warning: tag file not found at {tags_file}, using small fallback vocabulary.")
    return FALLBACK_TAGS


def main():
    parser = argparse.ArgumentParser(description="Interrogate an image using CLIP on Axera NPU")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("--tags_file", type=str, default=DEFAULT_TAGS_FILE, help="Path to a newline-separated tag vocabulary")
    parser.add_argument("--threshold", type=float, default=0.22, help="Similarity threshold (default: 0.22)")
    parser.add_argument("--top_k", type=int, default=10, help="Max number of tags to return (default: 10)")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        sys.exit(1)

    tags = load_tags(args.tags_file)

    print("Initializing CLIP Interrogator...")
    ci = CLIPInterrogator(cache_dir=DEFAULT_CACHE_DIR)
    ci.precompute_text_embeddings(tags)

    print(f"Processing image: {args.image_path}")
    try:
        img = Image.open(args.image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)

    scores, used_tags = ci.interrogate(img)
    scores = scores[0]

    results = [(used_tags[i], score) for i, score in enumerate(scores) if score >= args.threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    results = results[: args.top_k]

    if not results:
        print("No tags found above threshold.")
    else:
        print("\nDetected Tags:")
        for tag, score in results:
            print(f"  - {tag} ({score:.4f})")

        print("\nCaption-style output:")
        print(", ".join(r[0] for r in results))


if __name__ == "__main__":
    main()
