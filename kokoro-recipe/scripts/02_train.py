#!/usr/bin/env python3
"""
Step 2: Launch Kokoro Fine-Tuning (Stage 1 or Stage 2)
=======================================================
Training is launched fully detached from the terminal — it survives SSH
disconnects, terminal closes, and script crashes.

Usage:
    python scripts/02_train.py --stage 1 --config configs/config.yml --gpu 0
    python scripts/02_train.py --stage 2 --config configs/config.yml --gpu 0

    # Multi-GPU (DataParallel — note: WavLM discriminator has a DDP bug,
    #            so use a single GPU unless you disable joint_epoch GAN):
    python scripts/02_train.py --stage 2 --config configs/config.yml --gpu 0,1

Monitor:
    tail -f logs/stage1.log
    tail -f logs/stage2.log

Kill:
    kill $(cat logs/stage1.pid)
    kill $(cat logs/stage2.pid)
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    return yaml.safe_load(Path(config_path).read_text())


def find_framework(cfg: dict, recipe_root: Path) -> tuple[Path, Path]:
    """Return (styletts2_dir, training_dir) from config."""
    fw = cfg.get("framework", {})

    # New bundled layout: styletts2_dir + training_dir
    styletts2_raw = fw.get("styletts2_dir", "")
    training_raw  = fw.get("training_dir", "")

    def resolve(raw: str, fallback: Path) -> Path:
        p = Path(raw) if raw else None
        if p:
            return p if p.is_absolute() else (recipe_root / p).resolve()
        return fallback

    styletts2_dir = resolve(styletts2_raw, recipe_root / "framework" / "StyleTTS2")
    training_dir  = resolve(training_raw,  recipe_root / "framework" / "training")

    if not (styletts2_dir / "train_first.py").exists():
        # Legacy fallback: kokoro_deutsch_dir
        kd_raw = fw.get("kokoro_deutsch_dir", "")
        if kd_raw:
            kd = Path(kd_raw)
            if kd.exists():
                styletts2_dir = kd / "StyleTTS2"
                training_dir  = kd / "training"
        if not (styletts2_dir / "train_first.py").exists():
            print("[ERROR] Cannot find StyleTTS2 training scripts.")
            print(f"  Expected: {styletts2_dir / 'train_first.py'}")
            print("  Check framework.styletts2_dir in configs/config.yml")
            sys.exit(1)

    return styletts2_dir, training_dir


def make_styletts2_config(cfg: dict, styletts2_dir: Path, training_dir: Path,
                          stage: int, recipe_root: Path) -> Path:
    """Write a temporary StyleTTS2 config.yml with paths from our config.yml."""
    model_cfg   = cfg["model"]
    stage1_cfg  = cfg["stage1"]
    stage2_cfg  = cfg["stage2"]
    audio_cfg   = cfg.get("audio", {})

    train_list   = training_dir / "train_list.txt"
    val_list     = training_dir / "val_list.txt"
    ood_file     = training_dir / "OOD_texts.txt"
    utils_dir    = styletts2_dir / "Utils"

    # OOD_texts.txt fallback: use recipe's eval_texts.txt if not in training_dir
    if not ood_file.exists():
        ood_file = recipe_root / "training" / "eval_texts.txt"

    log_dir  = recipe_root / model_cfg.get("log_dir", "output/kokoro-finetune")
    run_name = model_cfg.get("run_name", "kokoro-custom-v1")

    s2 = stage2_cfg
    slm = s2.get("slmadv") or {}
    loss = s2.get("loss", {})
    opt  = s2.get("optimizer", {})

    sc = {
        "batch_size":   (stage1_cfg if stage == 1 else stage2_cfg).get("batch_size", 2),
        # Stage 1 may set its own crop window; otherwise it uses stage2.max_len as before.
        "max_len":      (stage1_cfg.get("max_len", s2.get("max_len", 180)) if stage == 1
                         else s2.get("max_len", 180)),
        "epochs":       (stage1_cfg if stage == 1 else stage2_cfg).get("epochs", 10),
        "epochs_1st":   stage1_cfg.get("epochs", 2),
        "epochs_2nd":   stage2_cfg.get("epochs", 10),
        "save_freq":    (stage1_cfg if stage == 1 else stage2_cfg).get("save_freq", 2),
        "pretrained_model":             str((recipe_root / model_cfg.get("base_model", "framework/training/kokoro_base.pth")).resolve()),
        "first_stage_path":             "first_stage.pth",
        "load_only_params":             s2.get("load_only_params", True),
        "second_stage_load_pretrained": s2.get("second_stage_load_pretrained", True),
        "log_dir":                      str(log_dir / run_name),
        "data_params": {
            "train_data":   str(train_list),
            "val_data":     str(val_list),
            "root_path":    str(recipe_root / cfg["dataset"].get("wavs_dir", "data/wavs")),
            "OOD_data":     str(ood_file),
            "min_length":   50,
            "num_workers":  (stage1_cfg if stage == 1 else stage2_cfg).get("num_workers", 4),
        },
        "preprocess_params": {
            "sr": audio_cfg.get("sr", 24000),
            "spect_params": {
                "n_fft":       audio_cfg.get("n_fft", 2048),
                "win_length":  audio_cfg.get("win_length", 1200),
                "hop_length":  audio_cfg.get("hop_length", 300),
                "n_mels":      audio_cfg.get("n_mels", 80),
                "fmin":        audio_cfg.get("fmin", 0),
                "fmax":        audio_cfg.get("fmax", 8000),
            },
        },
        "model_params": {
            "dim_in": 64, "n_token": 178, "hidden_dim": 512, "style_dim": 128,
            "max_dur": 50, "multispeaker": False, "n_mels": 80, "dropout": 0.2,
            "n_layer": 3, "text_encoder_kernel_size": 5,
            "decoder": {
                "type": "istftnet",
                "upsample_rates": [10, 6],
                "upsample_kernel_sizes": [20, 12],
                "upsample_initial_channel": 512,
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "gen_istft_n_fft": 20,
                "gen_istft_hop_size": 5,
            },
            "diffusion": {
                "embedding_mask_proba": 0.1,
                "transformer": {"num_layers": 3, "num_heads": 8, "head_features": 64, "multiplier": 2},
                "dist": {"sigma_data": 0.2, "estimate_sigma_data": True, "mean": -3.0, "std": 1.0},
            },
            "plbert": {
                "hidden_size": 768, "num_attention_heads": 12, "intermediate_size": 2048,
                "max_position_embeddings": 512, "num_hidden_layers": 12, "dropout": 0.1,
            },
            "slm": {
                "model": "microsoft/wavlm-base-plus", "sr": 16000,
                "hidden": 768, "nlayers": 13, "initial_channel": 64,
            },
        },
        "loss_params": {
            "lambda_gen":  loss.get("lambda_gen", 1.0),
            "lambda_mel":  loss.get("lambda_mel", 5.0),
            "lambda_dur":  loss.get("lambda_dur", 1.0),
            "lambda_ce":   loss.get("lambda_ce", 20.0),
            "lambda_F0":   loss.get("lambda_F0", 2.0),
            "lambda_norm": loss.get("lambda_norm", 1.0),
            "lambda_s2s":  loss.get("lambda_s2s", 1.0),
            "lambda_mono": loss.get("lambda_mono", 1.0),
            "lambda_slm":  loss.get("lambda_slm", 1.0),
            "lambda_diff": loss.get("lambda_diff", 0.0),
            "lambda_sty":  loss.get("lambda_sty", 0.0),
            "TMA_epoch":   0,
            "diff_epoch":  999,
            "joint_epoch": s2.get("joint_epoch", 99),
        },
        "optimizer_params": {
            "lr":      opt.get("lr", 0.00005),
            "bert_lr": opt.get("bert_lr", 0.000005),
            "ft_lr":   opt.get("ft_lr", 0.00005),
        },
        "F0_path":    str(utils_dir / "JDC" / "bst.t7"),
        "ASR_config": str(utils_dir / "ASR" / "config.yml"),
        "ASR_path":   str(utils_dir / "ASR" / "epoch_00080.pth"),
        "PLBERT_dir": str(utils_dir / "PLBERT"),
        "slmadv_params": {
            "min_len": slm.get("min_len", 100),
            "max_len": slm.get("max_len", 500),
            # SLMAdversarialLoss returns None unless it collects >= 2 samples, so the
            # SLM step only runs if batch_percentage * batch_size > 1
            # (with the default 0.2 that means batch_size >= 6).
            "batch_percentage": slm.get("batch_percentage", 0.2),
            "iter": 10, "thresh": 5, "scale": 0.01, "sig": 1.5,
        },
    }

    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
    yaml.dump(sc, tf, default_flow_style=False, allow_unicode=True)
    tf.close()
    return Path(tf.name)


def warn_if_slm_inactive(cfg: dict) -> None:
    """Warn when the SLM (WavLM) adversarial step can never run in Stage 2."""
    s2 = cfg["stage2"]
    epochs = s2.get("epochs", 10)
    joint_epoch = s2.get("joint_epoch", 99)
    batch_size = s2.get("batch_size", 2)
    pct = (s2.get("slmadv") or {}).get("batch_percentage", 0.2)
    if joint_epoch < epochs and pct * batch_size <= 1:
        print(f"[warn] stage2.slmadv.batch_percentage ({pct}) x stage2.batch_size ({batch_size}) <= 1,")
        print("       so the SLM adversarial step will be skipped on every step after joint_epoch")
        print("       (and those steps won't be logged, because that path skips the rest of the step).")
        print("       Use batch_size >= 2 with batch_percentage 1.0 (or batch_size >= 6 with 0.2).")


_CUDNN_CRASH_PATTERNS = (
    "illegal memory access",
    "unable to find an engine",
    "CUDNN_STATUS",
    "cuDNN error",
)


def decide_slm_disable_cudnn(cfg: dict, recipe_root: Path) -> bool:
    """stage2.slm_disable_cudnn: false (default) | true | auto.

    A fallback for torch builds older than the pinned 2.8.0. With torch 2.6.0
    (cuDNN 9.1.0) the SLM generator backward crashes with a CUDA illegal memory
    access; torch 2.8.0 (cuDNN 9.10.2) does not.

    true runs the SLM-adversarial forward/backward with cuDNN disabled (native
    CUDA kernels; whole Stage 2 steps were about 1.6x slower in testing). auto
    turns that on only if the previous Stage 2 launch's log (logs/stage2.log,
    read before it is overwritten) shows a cuDNN / illegal-memory-access crash,
    so relaunching after such a crash takes the safe path while a clean run
    keeps full speed. auto looks at the previous launch only; use true to keep
    the fallback on across relaunches.
    """
    val = cfg["stage2"].get("slm_disable_cudnn", False)
    if isinstance(val, bool):
        return val
    if str(val).lower() != "auto":
        return str(val).lower() in ("1", "true", "yes", "on")
    prev_log = recipe_root / "logs" / "stage2.log"
    try:
        text = prev_log.read_text(errors="replace")
    except OSError:
        return False
    for pat in _CUDNN_CRASH_PATTERNS:
        if pat in text:
            print(f"[auto] {prev_log} shows a '{pat}' crash -> SLM step will run without cuDNN")
            return True
    return False


def launch(stage: int, gpus: str, config_path: str) -> None:
    cfg = load_config(config_path)
    recipe_root = Path(config_path).parent.parent.resolve()
    styletts2_dir, training_dir = find_framework(cfg, recipe_root)

    script = styletts2_dir / ("train_first.py" if stage == 1 else "train_second.py")
    if not script.exists():
        print(f"[ERROR] Training script not found: {script}")
        print("  Check framework.styletts2_dir in configs/config.yml")
        sys.exit(1)

    # Use the system python (requirements.txt installs all deps)
    python = sys.executable

    styletts_config = make_styletts2_config(cfg, styletts2_dir, training_dir, stage, recipe_root)
    if stage == 2:
        warn_if_slm_inactive(cfg)
        slm_no_cudnn = decide_slm_disable_cudnn(cfg, recipe_root)
        sc = yaml.safe_load(styletts_config.read_text())
        sc["slm_disable_cudnn"] = slm_no_cudnn
        styletts_config.write_text(yaml.dump(sc, default_flow_style=False, allow_unicode=True))
        print(f"SLM step cuDNN guard: {'ON' if slm_no_cudnn else 'off'}")

    logs_dir = recipe_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"stage{stage}.log"
    pid_file = logs_dir / f"stage{stage}.pid"

    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            try:
                os.kill(pid, 0)
                print(f"[warn] Stage {stage} may still be running (PID {pid}).")
                print(f"       Kill with: kill {pid}")
                resp = input("Launch anyway? [y/N] ").strip().lower()
                if resp not in ("y", "yes"):
                    sys.exit(0)
            except OSError:
                pass
        except ValueError:
            pass
        pid_file.unlink(missing_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            [python, "-u", str(script), "--config_path", str(styletts_config)],
            cwd=str(styletts2_dir),
            stdin=subprocess.DEVNULL,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    pid_file.write_text(str(proc.pid))
    print(f"Launched Stage {stage}   PID={proc.pid}   GPU(s)={gpus}")
    print(f"Log:  {log_file}")
    print()
    print("Monitor:  tail -f " + str(log_file))
    print("Kill:     kill $(cat " + str(pid_file) + ")")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True,
                    help="1 = acoustic warmup  |  2 = prosody fine-tuning")
    ap.add_argument("--gpu", default="0",
                    help="Comma-separated CUDA device indices (default: 0)")
    ap.add_argument("--config", default="configs/config.yml",
                    help="Path to config.yml (default: configs/config.yml)")
    args = ap.parse_args()
    launch(args.stage, args.gpu, args.config)


if __name__ == "__main__":
    main()
