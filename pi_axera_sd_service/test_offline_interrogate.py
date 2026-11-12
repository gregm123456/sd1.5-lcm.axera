"""Offline CLIP interrogation checks.

Run from the repo root:  .venv/bin/python pi_axera_sd_service/test_offline_interrogate.py
(pytest also works). The parity test is skipped if the HF snapshot is not cached.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "clip_assets")
REPO_ID = "openai/clip-vit-base-patch32"


def test_manifest_matches_files():
    with open(os.path.join(ASSETS, "MANIFEST.json")) as f:
        manifest = json.load(f)
    assert "clip_projections.npz" in manifest["files"]
    for name, digest in manifest["files"].items():
        with open(os.path.join(ASSETS, name), "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == digest, name


def test_loads_with_no_hub_access():
    # Bundle must load with the Hub disabled and an empty HF cache.
    code = (
        "import os,tempfile;"
        "os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HOME']=tempfile.mkdtemp();"
        "from transformers import CLIPProcessor, CLIPTokenizer;"
        f"p=CLIPProcessor.from_pretrained({ASSETS!r},local_files_only=True);"
        f"t=CLIPTokenizer.from_pretrained({ASSETS!r},local_files_only=True);"
        "print(t('a photo of a dog')['input_ids'])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr


def test_parity_with_hf_snapshot():
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    code = (
        "from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer;import numpy as np,sys;"
        f"m=CLIPModel.from_pretrained({REPO_ID!r});"
        f"p=CLIPProcessor.from_pretrained({REPO_ID!r});t=CLIPTokenizer.from_pretrained({REPO_ID!r});"
        f"a=np.load({os.path.join(ASSETS, 'clip_projections.npz')!r});"
        "assert np.array_equal(a['visual_projection'],m.visual_projection.weight.detach().numpy().T);"
        "assert np.array_equal(a['text_projection'],m.text_projection.weight.detach().numpy().T);"
        f"lp=CLIPProcessor.from_pretrained({ASSETS!r},local_files_only=True);"
        f"lt=CLIPTokenizer.from_pretrained({ASSETS!r},local_files_only=True);"
        "from PIL import Image;im=Image.fromarray((np.random.RandomState(0).rand(300,400,3)*255).astype('uint8'));"
        "assert np.array_equal(p(images=im,return_tensors='np')['pixel_values'],lp(images=im,return_tensors='np')['pixel_values']);"
        "s='a photo of a golden retriever';"
        "assert t(s,padding='max_length',max_length=77)['input_ids']==lt(s,padding='max_length',max_length=77)['input_ids'];"
        "print('PARITY_OK')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    if "PARITY_OK" not in out.stdout and "LocalEntryNotFound" in out.stderr + out.stdout:
        print("SKIP: HF snapshot not cached locally")
        return
    assert "PARITY_OK" in out.stdout, out.stderr


def test_missing_or_corrupt_bundle_gives_clear_error():
    # Import only the verifier, without running the service's module-level model loading.
    src = open(os.path.join(HERE, "pi_axera_sd_generator.py")).read()
    start = src.index("def verify_clip_assets")
    end = src.index("class CLIPInterrogator")
    ns = {"os": os, "json": json, "hashlib": hashlib, "CLIP_PROJECTIONS_FILE": "clip_projections.npz"}
    exec(src[start:end], ns)
    verify = ns["verify_clip_assets"]

    with tempfile.TemporaryDirectory() as d:
        try:
            verify(d)
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "MANIFEST.json" in str(e)
        shutil.copytree(ASSETS, os.path.join(d, "b"))
        verify(os.path.join(d, "b"))
        with open(os.path.join(d, "b", "clip_projections.npz"), "ab") as f:
            f.write(b"x")
        try:
            verify(os.path.join(d, "b"))
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "sha256" in str(e)


def _load_validator():
    import textwrap

    src = open(os.path.join(HERE, "pi_axera_sd_generator.py")).read()
    start = src.index("    def _load_valid_cache")
    end = src.index("    def precompute_text_embeddings")
    ns = {"os": os, "np": np}
    exec(textwrap.dedent(src[start:end]), ns)

    class Stub:
        text_projection = np.zeros((512, 512), dtype=np.float32)

    return lambda path, n: ns["_load_valid_cache"](Stub(), path, n)


def test_invalid_embedding_caches_are_rejected_and_removed():
    load = _load_validator()
    good = np.random.RandomState(0).randn(4, 512).astype(np.float32)
    good /= np.linalg.norm(good, axis=-1, keepdims=True)
    bad_norm = good * 3
    nan = good.copy()
    nan[0, 0] = np.nan
    with tempfile.TemporaryDirectory() as d:
        def save(name, arr):
            path = os.path.join(d, name)
            np.save(path, arr)
            return path + ".npy"

        p = save("good", good)
        assert load(p, 4) is not None and os.path.exists(p)
        cases = {
            "wrong_rows": (save("rows", good), 5),
            "wrong_dim": (save("dim", good[:, :256]), 4),
            "unnormalized": (save("norm", bad_norm), 4),
            "nan": (save("nan", nan), 4),
            "int": (save("int", good.astype(np.int32)), 4),
        }
        trunc = os.path.join(d, "trunc.npy")
        raw = open(p, "rb").read()
        open(trunc, "wb").write(raw[: len(raw) // 2])
        cases["truncated"] = (trunc, 4)
        empty = os.path.join(d, "empty.npy")
        open(empty, "wb").close()
        cases["empty"] = (empty, 4)
        for name, (path, n) in cases.items():
            assert load(path, n) is None, name
            assert not os.path.exists(path), name


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
