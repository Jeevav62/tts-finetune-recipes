#!/usr/bin/env python3
"""Check that a checkpoint loads the way the training scripts load it (CPU only).

Builds the StyleTTS2/Kokoro model exactly as train_first.py / train_second.py do,
then loads --ckpt the way the chosen --mode does, and reports per module how
many tensors were missing/unexpected AND how many tensors really equal the
checkpoint afterwards. With --old it also runs the pre-fix loader
(load_state_dict(params[key], strict=False), no prefix handling) for comparison.

Modes:
  stage1_base    kokoro_base.pth -> Stage 1 (accelerate-prepared, single process)
  stage1_resume  epoch_1st_N.pth -> Stage 1 resume (same path)
  stage2_first   first_stage.pth -> Stage 2 start (before DataParallel wrap, ignores PE/msd/mpd/wd/diffusion)
  stage2_resume  epoch_2nd_N.pth -> Stage 2 resume (before DataParallel wrap)

Usage (CPU only, safe to run next to a training job):
  python scripts/verify_checkpoint_load.py --config configs/config.yml \
      --ckpt output/kokoro-finetune/kokoro-custom-v1/epoch_2nd_00002.pth --mode stage2_resume --old
"""
import argparse, os, sys, copy, importlib.util
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import yaml, torch

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--mode", required=True, choices=["stage1_base", "stage1_resume", "stage2_first", "stage2_resume"])
ap.add_argument("--old", action="store_true", help="also run the pre-fix loader for comparison")
args = ap.parse_args()
args.ckpt = str(Path(args.ckpt).resolve())
torch.set_num_threads(4)

recipe = Path(args.config).resolve().parent.parent
spec = importlib.util.spec_from_file_location("t02", recipe / "scripts" / "02_train.py")
t02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t02)
cfg = t02.load_config(args.config)
sdir, tdir = t02.find_framework(cfg, recipe)
stage = 1 if args.mode.startswith("stage1") else 2
sc = yaml.safe_load(open(t02.make_styletts2_config(cfg, sdir, tdir, stage, recipe)))
os.chdir(sdir); sys.path.insert(0, str(sdir))
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
import models
from models import build_model, load_ASR_models, load_F0_models
from utils import recursive_munch
from Utils.PLBERT.util import load_plbert

def fresh():
    mp = recursive_munch(sc["model_params"])
    m = build_model(mp, load_ASR_models(sc["ASR_path"], sc["ASR_config"]),
                    load_F0_models(sc["F0_path"]), load_plbert(sc["PLBERT_dir"]))
    if stage == 1:  # train_first.py: accelerator.prepare(model[k]) before loading
        from accelerate import Accelerator
        acc = Accelerator(cpu=True, split_batches=True)
        for k in m: m[k] = acc.prepare(m[k])
    return m

ignore = ["predictor_encoder", "msd", "mpd", "wd", "diffusion"] if args.mode == "stage2_first" else []
net = _orig_load(args.ckpt, map_location="cpu", weights_only=False)["net"]
print("checkpoint modules:", {k: (len(v), next(iter(v))[:40]) for k, v in net.items()})

def equal_report(m, tag):
    tot = eq = 0
    for key in m:
        if key not in net or key in ignore: continue
        own = m[key].state_dict()
        ck = models.match_module_prefix(net[key], list(own))
        e = sum(1 for k, v in own.items() if k in ck and v.shape == ck[k].shape and torch.equal(v, ck[k].to(v.dtype)))
        # weight_norm compat: checkpoint weight_g/weight_v map to parametrizations.*.original0/1
        if e < len(own):
            for k, v in own.items():
                if k.endswith("parametrizations.weight.original0") or k.endswith("parametrizations.weight.original1"):
                    base = k.rsplit("parametrizations.weight.original", 1)[0]
                    src = ck.get(base + ("weight_g" if k.endswith("0") else "weight_v"))
                    if src is not None and src.shape == v.shape and torch.equal(v, src): e += 1
        tot += len(own); eq += e
        print("  [%s] %-18s %4d/%4d tensors equal to checkpoint" % (tag, key, e, len(own)))
    print("[%s] TOTAL %d/%d tensors equal to checkpoint (%.1f%%)" % (tag, eq, tot, 100.0 * eq / max(tot, 1)))
    return eq, tot

if args.old:
    m = fresh(); miss = tot = 0
    for key in m:
        if key in net and key not in ignore:
            r = m[key].load_state_dict(net[key], strict=False)
            n = len(m[key].state_dict()); miss += len(r.missing_keys); tot += n
            print("  [old] %-18s missing %4d/%4d  unexpected %4d" % (key, len(r.missing_keys), n, len(r.unexpected_keys)))
    print("[old] TOTAL missing %d/%d" % (miss, tot))
    equal_report(m, "old")

m = fresh()
try:
    models.load_checkpoint(m, None, args.ckpt, load_only_params=True, ignore_modules=ignore)
    eq, tot = equal_report(m, "new")
    print("RESULT: %s" % ("PASS" if eq == tot else "FAIL (%d tensors differ)" % (tot - eq)))
except models.CheckpointLoadError as e:
    print("RESULT: LOADER REFUSED:", e)
