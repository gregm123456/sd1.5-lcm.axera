#!/usr/bin/env python
"""One-time developer tool: build the offline CLIP asset bundle.

This is the only place that may touch the Hugging Face Hub. It loads
openai/clip-vit-base-patch32 (from the local HF cache, or the network if not
cached), and writes the few small files the interrogation service needs to
`clip_assets/`:

  clip_projections.npz   visual/text projection matrices (already transposed)
  preprocessor + tokenizer files
  MANIFEST.json          source revision and sha256 of every file

Usage: python pi_axera_sd_service/export_clip_assets.py [--revision REV]
"""
import argparse
import hashlib
import json
import os

import numpy as np
from transformers import CLIPModel, CLIPProcessor

REPO_ID = "openai/clip-vit-base-patch32"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clip_assets")
PROJECTIONS = "clip_projections.npz"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default=None, help="HF revision to export (default: main)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    model = CLIPModel.from_pretrained(REPO_ID, revision=args.revision)
    processor = CLIPProcessor.from_pretrained(REPO_ID, revision=args.revision)

    visual = model.visual_projection.weight.detach().numpy().T.astype(np.float32)
    text = model.text_projection.weight.detach().numpy().T.astype(np.float32)
    np.savez(os.path.join(OUT_DIR, PROJECTIONS), visual_projection=visual, text_projection=text)
    processor.save_pretrained(OUT_DIR)

    files = {
        name: sha256(os.path.join(OUT_DIR, name))
        for name in sorted(os.listdir(OUT_DIR))
        if name != "MANIFEST.json"
    }
    manifest = {
        "source": REPO_ID,
        "revision": args.revision or "main",
        "commit_hash": getattr(model.config, "_commit_hash", None),
        "files": files,
    }
    with open(os.path.join(OUT_DIR, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(files)} files to {OUT_DIR}")


if __name__ == "__main__":
    main()
