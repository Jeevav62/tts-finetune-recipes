#!/usr/bin/env python3
"""
Step 0: Download the support models and build kokoro_base.pth
==============================================================
These files are too large for git and are listed in .gitignore, so a fresh
clone does not contain them:

  framework/StyleTTS2/Utils/        ASR aligner, JDC pitch extractor, PL-BERT
                                    (code + weights) from yl4579/StyleTTS2
  framework/training/kokoro_base.pth  Kokoro-82M base weights, converted from
                                    hexgrad/Kokoro-82M kokoro-v1_0.pth

Usage:
    python scripts/00_download_weights.py
    python scripts/00_download_weights.py --force     # re-download everything

Files that already exist are kept unless --force is given. Paths follow
framework.styletts2_dir and model.base_model in configs/config.yml.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

import yaml

# Pinned so everyone trains with the same support models.
STYLETTS2_COMMIT = "5cedc71c333f8d8b8551ca59378bdcc7af4c9529"
STYLETTS2_RAW = f"https://raw.githubusercontent.com/yl4579/StyleTTS2/{STYLETTS2_COMMIT}"

UTILS_FILES = [
    "Utils/__init__.py",
    "Utils/ASR/__init__.py",
    "Utils/ASR/config.yml",
    "Utils/ASR/layers.py",
    "Utils/ASR/models.py",
    "Utils/ASR/epoch_00080.pth",
    "Utils/JDC/__init__.py",
    "Utils/JDC/model.py",
    "Utils/JDC/bst.t7",
    "Utils/PLBERT/config.yml",
    "Utils/PLBERT/util.py",
    "Utils/PLBERT/step_1000000.t7",
]

KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
KOKORO_FILE = "kokoro-v1_0.pth"
# Modules of kokoro-v1_0.pth that the training scripts load from base_model.
KOKORO_MODULES = ["bert", "bert_encoder", "predictor", "text_encoder", "decoder"]


def resolve(recipe_root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (recipe_root / p).resolve()


def download(url: str, dst: Path, force: bool) -> None:
    if dst.exists() and dst.stat().st_size > 0 and not force:
        print(f"  keep  {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    print(f"  get   {url}")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.replace(dst)


def fetch_utils(utils_dir: Path, force: bool) -> None:
    print(f"StyleTTS2 Utils @ {STYLETTS2_COMMIT[:7]} -> {utils_dir}")
    for rel in UTILS_FILES:
        download(f"{STYLETTS2_RAW}/{rel}", utils_dir / Path(rel).relative_to("Utils"), force)


def build_base_model(dst: Path, force: bool) -> None:
    print(f"Kokoro base model -> {dst}")
    if dst.exists() and dst.stat().st_size > 0 and not force:
        print(f"  keep  {dst}")
        return

    import torch
    from huggingface_hub import hf_hub_download

    src = hf_hub_download(KOKORO_REPO, KOKORO_FILE, revision=KOKORO_REVISION)
    print(f"  from  {src}")
    raw = torch.load(src, map_location="cpu", weights_only=True)
    missing = [k for k in KOKORO_MODULES if k not in raw]
    if missing:
        sys.exit(f"[ERROR] {KOKORO_FILE} is missing modules {missing}; found {list(raw)}")

    # kokoro-v1_0.pth stores each module as a state dict saved from a
    # DataParallel wrapper ("module." prefix). The training scripts load
    # base_model with strict=False, so prefixed keys would be silently
    # skipped. Strip the prefix and wrap in the {"net": {...}} layout that
    # models.load_checkpoint expects.
    def strip(sd: dict) -> dict:
        return {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    torch.save({"net": {k: strip(raw[k]) for k in KOKORO_MODULES}}, tmp)
    tmp.replace(dst)
    print(f"  wrote {dst} ({dst.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/config.yml",
                    help="Path to config.yml (default: configs/config.yml)")
    ap.add_argument("--force", action="store_true", help="Re-download and overwrite existing files")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    recipe_root = Path(args.config).resolve().parent.parent
    styletts2_dir = resolve(recipe_root, cfg.get("framework", {}).get("styletts2_dir", "./framework/StyleTTS2"))
    base_model = resolve(recipe_root, cfg.get("model", {}).get("base_model", "./framework/training/kokoro_base.pth"))

    fetch_utils(styletts2_dir / "Utils", args.force)
    build_base_model(base_model, args.force)
    print("Done.")


if __name__ == "__main__":
    main()
