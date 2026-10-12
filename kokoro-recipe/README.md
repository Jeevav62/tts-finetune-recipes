# Kokoro Fine-Tuning Recipe

Fine-tune **Kokoro-82M** (a StyleTTS2-based TTS model) on your own speaker data.
The result is a model that speaks in your target voice, with correct pronunciation
for your domain vocabulary.

> **Example checkpoint trained with this recipe:**
> [![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-kokoro--82m--indian--en-ff6b00?style=flat-square)](https://huggingface.co/jeevav62/kokoro-82m-indian-en)
> — Kokoro-82M fine-tuned on Indian-English (male voice, 1058 clips)
>
> 🔗 https://huggingface.co/jeevav62/kokoro-82m-indian-en

---

## Credits

- **Kokoro-82M** — [hexgrad](https://huggingface.co/hexgrad/Kokoro-82M) (Apache 2.0)
  The base 82M-parameter TTS model this recipe fine-tunes.
- **StyleTTS2** — [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) (MIT)
  The underlying architecture (encoder, decoder, predictors, discriminators).
- **kokoro-deutsch** — [semidark/kokoro-deutsch](https://github.com/semidark/kokoro-deutsch) (Apache 2.0)
  The training fork of StyleTTS2 adapted for Kokoro fine-tuning. Contains the
  `train_first.py`, `train_second.py`, pretrained support models (ASR, JDC, PLBERT),
  and the `kokoro` inference library.
- **misaki** — [hexgrad/misaki](https://github.com/hexgrad/misaki) (Apache 2.0)
  G2P engine used for phoneme conversion.
- **monotonic_align** — [resemble-ai/monotonic_align](https://github.com/resemble-ai/monotonic_align)
  Cython extension for fast monotonic alignment during training.

---

## Hardware Requirements

| Setup | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 12 GB | 24 GB (RTX 4090) |
| RAM | 16 GB | 32 GB |
| Python | **3.12** (required — spacy lacks 3.13 wheels) | 3.12 |
| PyTorch | **2.8.0** (pinned; CUDA 12.8 wheels, see Step 1) | 2.8.0 |
| Storage | 10 GB | 20 GB |

**Training time** (1000 clips, Stage 1 + Stage 2 with 10 epochs each):
- RTX 4090 (24 GB): ~75 min Stage 1 + ~3 hours Stage 2
  - Stage 1: ~75 min
  - Stage 2 (10 epochs, GAN on): ~3 hours

---

## What's Included

The StyleTTS2 training scripts are **bundled** in `framework/`, so there is no
need to clone StyleTTS2. The large weight files are **not** in git (they are
listed in `.gitignore`). Run `scripts/00_download_weights.py` once to fetch them
(Quick Start step 3):

```
framework/
├── StyleTTS2/
│   ├── train_first.py, train_second.py   ← training scripts
│   ├── models.py, losses.py, ...         ← architecture
│   └── Utils/                            ← downloaded: from yl4579/StyleTTS2 (pinned commit)
│       ├── ASR/epoch_00080.pth           ← alignment model (91 MB)
│       ├── JDC/bst.t7                    ← pitch extractor (21 MB)
│       └── PLBERT/step_1000000.t7        ← PL-BERT (25 MB; Kokoro's own BERT weights come from kokoro_base.pth)
└── training/
    ├── config.json                       ← Kokoro phoneme vocab
    └── kokoro_base.pth                   ← generated: converted from hexgrad/Kokoro-82M kokoro-v1_0.pth (~330 MB)
```

`kokoro_base.pth` keeps the `bert`, `bert_encoder`, `predictor`, `text_encoder`
and `decoder` modules of `kokoro-v1_0.pth`, with the `module.` prefix stripped,
under a `net` key. The training scripts load it with `strict=False`, so a file
that still has the `module.` prefix would load **nothing** without any error.
Use the script rather than pointing `model.base_model` at `kokoro-v1_0.pth` directly.

---

## Quick Start

```bash
# 1. Python 3.12 is required
python3.12 --version   # must show 3.12.x

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the support models (StyleTTS2 Utils/) and build kokoro_base.pth (one-time)
python scripts/00_download_weights.py

# 4. Check that monotonic_align imports (pip already builds the extension)
python -c "from monotonic_align import maximum_path; print('monotonic_align OK')"

# 5. Add your dataset (see data/DATASET_FORMAT.md)
#    Place WAV files in data/wavs/ and create data/train.csv

# 6. (Optional) Edit configs/config.yml to change run name, epochs, batch_size
#    The default paths already point at the files from step 3.

# 7. Prepare dataset (text normalization + G2P)
python scripts/01_prepare_dataset.py --config configs/config.yml

# 8. Stage 1 training (~8 min on RTX 4090)
python scripts/02_train.py --stage 1 --config configs/config.yml --gpu 0

# 9. Stage 2 training (~65 min for 10 epochs on RTX 4090)
python scripts/02_train.py --stage 2 --config configs/config.yml --gpu 0

# 10. Evaluate all checkpoints (synthesizes test sentences for each epoch)
python scripts/03_eval_all_epochs.py --config configs/config.yml

# 11. Inference with the best epoch
python scripts/05_infer.py \
    --voicepack  eval/epoch08/voicepack.pt \
    --checkpoint eval/epoch08/kokoro_converted.pth \
    --text "Hello, this is my custom voice." \
    --out  output.wav
```

---

## Dataset Format

See **[data/DATASET_FORMAT.md](data/DATASET_FORMAT.md)** for full details.

**Quick summary:**
```
data/
├── train.csv              # CSV: id,target_audio,text (with header row)
└── wavs/
    ├── 0001.wav           # 24 kHz mono WAV, ~3-25 sec
    └── ...
```

---

## Configuration Reference

All training parameters live in **`configs/config.yml`**.

### Paths

| Key | Default | Description |
|---|---|---|
| `dataset.csv_path` | `./data/train.csv` | Your transcript CSV |
| `dataset.wavs_dir` | `./data/wavs/` | Your WAV files directory |
| `model.base_model` | `./framework/training/kokoro_base.pth` | Kokoro-82M base weights (created by `scripts/00_download_weights.py`) |
| `model.log_dir` | `./output/kokoro-finetune` | Where checkpoints are saved |
| `framework.styletts2_dir` | `./framework/StyleTTS2` | Training scripts (bundled) + `Utils/` (downloaded). Don't change |
| `framework.training_dir` | `./framework/training` | Vocab (bundled) + base weights (generated). Don't change |

### Stage 1 (acoustic warmup)

| Key | Default | Description |
|---|---|---|
| `stage1.epochs` | `2` | Number of epochs (2 is usually enough) |
| `stage1.batch_size` | `4` | Reduce to 2 if you run out of GPU memory |
| `stage1.save_freq` | `1` | Save every N epochs |
| `stage1.max_len` | `stage2.max_len` | Mel frames per Stage 1 crop window. Falls back to `stage2.max_len` if unset |

### Stage 2 (prosody fine-tuning)

| Key | Default | Description |
|---|---|---|
| `stage2.epochs` | `10` | Training epochs |
| `stage2.batch_size` | `2` | Reduce to 1 if you run out of GPU memory |
| `stage2.max_len` | `180` | Mel frames per window (~2.25 sec). Reduce for short clips |
| `stage2.joint_epoch` | `99` | Epoch when GAN discriminators turn on. `99` = disabled (saves ~4 GB VRAM). Set to `3` to enable |
| `stage2.load_only_params` | `true` | Reset optimizer when resuming. Keep `true` |
| `stage2.slmadv.batch_percentage` | `0.2` | Share of each batch used for the SLM (WavLM) adversarial step after `joint_epoch`. The step only runs if `batch_percentage * batch_size > 1`, so use `1.0` with `batch_size: 2`. It can never run with `batch_size: 1` |
| `stage2.slmadv.min_len` / `max_len` | `100` / `500` | Min/max length of the SLM adversarial crop |
| `stage2.slm_disable_cudnn` | `false` | Fallback for torch older than the pinned 2.8.0, where the SLM adversarial step can crash in cuDNN (see Troubleshooting). `true` runs only the SLM adversarial forward/backward without cuDNN (whole Stage 2 steps ran ~1.6x slower in testing). `auto` turns it on only when the previous `logs/stage2.log` shows a cuDNN / illegal-memory-access crash. `KOKORO_SLM_NO_CUDNN=1`/`0` overrides it. Not needed with torch 2.8.0 |
| `stage2.loss.lambda_F0` | `2.0` | Pitch loss weight. Increase to `3.0` for sharper pitch |
| `stage2.loss.lambda_mel` | `5.0` | Mel reconstruction weight (main loss) |
| `stage2.loss.lambda_ce` | `20.0` | Duration cross-entropy (phoneme timing) |

### G2P

| Key | Default | Description |
|---|---|---|
| `g2p.language` | `en-gb` | espeak language code. Use `en-us` for American English |
| `g2p.lexicon` | `./scripts/lexicon.json` | Custom pronunciation overrides |

---

## Training Walkthrough

### Step 1 — Install

```bash
pip install -r requirements.txt
python scripts/00_download_weights.py
```

`requirements.txt` pins `torch==2.8.0` and `torchaudio==2.8.0`. On Linux x86_64 the
default PyPI wheels for 2.8.0 are CUDA 12.8 builds that bundle cuDNN 9.10.2, so the
driver must support CUDA 12.8 (`nvidia-smi` should show "CUDA Version: 12.8" or
higher). To install exactly that build explicitly, or on a machine where pip would
pick a different one:
```bash
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
Older torch builds are not recommended: with torch 2.6.0 (CUDA 12.4, cuDNN 9.1.0),
Stage 2 crashes in the SLM adversarial step once `joint_epoch` is reached (see
Troubleshooting). To upgrade an existing environment, re-run
`pip install -r requirements.txt`; existing checkpoints load unchanged.

`00_download_weights.py` downloads `Utils/` (ASR, JDC, PL-BERT code and weights)
from [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) at a pinned commit and
converts `kokoro-v1_0.pth` from [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
into `framework/training/kokoro_base.pth`. Existing files are kept; pass `--force`
to re-download.

The per-epoch TensorBoard audio samples use misaki's English G2P, which needs the
spaCy model `en_core_web_sm`. misaki downloads it automatically the first time it
runs (this needs `pip` in the environment and internet access). To install it up front:
```bash
python -m spacy download en_core_web_sm
```

Then check that the compiled `monotonic_align` extension imports. Current
`resemble-ai/monotonic_align` builds it during `pip install`; the installed
package no longer contains a `setup.py`, so a separate `build_ext` step is not needed:
```bash
python -c "from monotonic_align import maximum_path; print('monotonic_align OK')"
```

Performance tips:
- Cap CPU threads, e.g. `export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`. On large
  cloud hosts PyTorch otherwise sizes its thread pool from the host's physical
  core count (128 threads on a 2x 64-core EPYC host), not from the vCPUs your
  container was given (e.g. 36 vCPUs on a Runpod 4090 pod).
- On Runpod, create the virtualenv on the container disk (e.g. `/root/venv`), not
  on the `/workspace` network volume. With the venv on the network volume, Stage 2
  ran several times slower in testing.

### Step 2 — Prepare your dataset

1. Place your WAV files in `data/wavs/` (24 kHz mono, 3-25 sec each)
2. Create `data/train.csv` with columns `id,target_audio,text` (include header row)
3. Edit `scripts/lexicon.json` to add any domain-specific word pronunciations
4. Edit `framework/training/eval_texts.txt` with sentences from your domain for testing

### Step 3 — Configure

Open `configs/config.yml`. The framework and base model paths are already set —
you only need to adjust:
- `model.run_name` — change this to give your training run a name
- `stage2.epochs`, `stage2.batch_size`, `stage2.joint_epoch` as needed
- `g2p.language` — if your dataset is not British English

**Memory tuning:**
- 24 GB GPU: `batch_size: 2`, `joint_epoch: 99` (GAN off)
- 24 GB GPU + GAN: `batch_size: 1`, `joint_epoch: 3` (the SLM adversarial step
  can't run at batch size 1; see `stage2.slmadv.batch_percentage`)
- 48 GB GPU + GAN + SLM: `batch_size: 2`, `max_len: 960`, `slmadv.batch_percentage: 1.0`
  (peaked at about 38,800 MiB in nvidia-smi on an RTX 6000 Ada with `joint_epoch: 0`)
- 16 GB GPU: `batch_size: 1`, `joint_epoch: 99`

### Step 4 — Data prep

```bash
python scripts/01_prepare_dataset.py --config configs/config.yml
```

This normalizes your text, runs G2P, and writes:
- `framework/training/train_list.txt` — 95% of your data for training
- `framework/training/val_list.txt` — 5% held out for validation
- `framework/training/normalization_samples.txt` — 20 random rows to spot-check

**Check `normalization_samples.txt`** to make sure text normalization looks right
before starting training.

### Step 5 — Stage 1

```bash
python scripts/02_train.py --stage 1 --config configs/config.yml --gpu 0
```

- Trains the decoder (vocoder), style_encoder, text_aligner, pitch_extractor
- Takes ~5-10 min on RTX 4090
- Monitor progress: `tail -f logs/stage1.log`
- Output: `output/kokoro-finetune/kokoro-custom-v1/first_stage.pth`

### Step 6 — Stage 2

```bash
python scripts/02_train.py --stage 2 --config configs/config.yml --gpu 0
```

- Trains the predictor (duration, F0, energy), predictor_encoder, refines decoder
- Takes ~65 min for 10 epochs on RTX 4090
- Monitor progress: `tail -f logs/stage2.log`
- Output: `output/kokoro-finetune/kokoro-custom-v1/epoch_2nd_00000.pth` ... `epoch_2nd_00009.pth`

### Step 7 — Evaluate all epochs

```bash
python scripts/03_eval_all_epochs.py --config configs/config.yml
```

For each epoch this will:
1. Extract a voicepack
2. Synthesize all sentences from `framework/training/eval_texts.txt`
3. Save audio to `eval/epochNN/`

Listen to `eval/epoch00/` through `eval/epoch09/` and pick the best one.

### Step 8 — Pick the best checkpoint

There is no automatic "best" picker — listen to the eval audio and judge.

**What to listen for:**
- Correct pronunciation (especially custom words from your lexicon)
- Natural prosody (not robotic or flat)
- Voice consistency (sounds like the target speaker throughout)
- No buzzing, clicking, or audio artifacts

Typical result: epochs 6-9 sound best in a 10-epoch Stage 2 run.
Earlier epochs may have better pronunciation but flatter prosody.

### Step 9 — Inference

```bash
python scripts/05_infer.py \
    --voicepack  eval/epoch08/voicepack.pt \
    --checkpoint eval/epoch08/kokoro_converted.pth \
    --text "Your custom text here." \
    --out  output.wav
```

Interactive mode (type multiple sentences without re-loading the model each time):
```bash
python scripts/05_infer.py \
    --voicepack  eval/epoch08/voicepack.pt \
    --checkpoint eval/epoch08/kokoro_converted.pth
```

---

## Monitoring with TensorBoard

```bash
tensorboard --logdir output/
# Open http://localhost:6006 in your browser
```

Watch `train/loss` and `val/loss` — both should decrease over epochs.

---

## Custom Pronunciation Lexicon

Edit `scripts/lexicon.json` to add words your G2P engine mispronounces:

```json
{
  "_comment": "Add your custom word pronunciations here (IPA format)",
  "Bengaluru": "bˈɛŋɡəluɹu",
  "YourProduct": "jˈʊɹ pɹˈɒdʌkt"
}
```

Test your lexicon before training:
```python
import sys; sys.path.insert(0, 'scripts')
from g2p_helper import g2p_with_lexicon
print(g2p_with_lexicon("Bengaluru YourProduct"))
```

**Important:** The same lexicon is used at training prep AND inference.
If you add words after training, re-run `01_prepare_dataset.py` and retrain.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'kokoro'`**
→ Run `pip install -r requirements.txt` to install all dependencies.
  Make sure you are using Python 3.12.

**`No module named 'monotonic_align'` or Cython build errors**
→ Reinstall it with `pip install --force-reinstall --no-deps "monotonic_align @ git+https://github.com/resemble-ai/monotonic_align.git"`
  (needs a C compiler and the Python 3.12 headers), then run the import check from Quick Start Step 4.

**`No module named 'Utils'`, or file not found for `Utils/...` or `kokoro_base.pth`**
→ These files are not in git. Run `python scripts/00_download_weights.py` (Quick Start Step 3).

**`Could not load English G2P for TensorBoard inference: No module named 'spacy'`**
→ Training still runs, but TensorBoard audio samples are skipped. Reinstall with
  `pip install -r requirements.txt` (it installs `misaki[en]`, which brings in spaCy).

**CUDA out of memory during Stage 2**
→ Reduce `stage2.batch_size` to `1` and ensure `stage2.joint_epoch: 99` (GAN off).
  The WavLM discriminator (activated at joint_epoch) uses ~4 GB extra VRAM.

**Stage 2 crashes after `joint_epoch` with `CUDA error: an illegal memory access` or cuDNN `GET was unable to find an engine to execute this computation`**
→ This is a cuDNN failure in the backward pass of the SLM (WavLM) adversarial step
  (`loss_gen_lm.backward()` in `train_second.py`). It happens with torch 2.6.0
  (CUDA 12.4, cuDNN 9.1.0), typically within a few hundred SLM steps, and did not
  happen with torch 2.8.0 (cuDNN 9.10.2). Check your version with
  `python -c "import torch; print(torch.__version__, torch.backends.cudnn.version())"`
  (the Stage 2 log also prints it), and upgrade with `pip install -r requirements.txt`.
  If you must stay on an older torch, set `stage2.slm_disable_cudnn: true` (or
  `KOKORO_SLM_NO_CUDNN=1`, or `auto` to switch it on only after a crash) and resume
  from the last `epoch_2nd_*.pth`. Only the SLM adversarial forward/backward runs
  without cuDNN; whole Stage 2 steps ran about 1.6x slower with it on in testing.

**Stage 1 checkpoint not found for Stage 2**
→ Stage 1 writes `epoch_1st_00001.pth`. `02_train.py` looks for `first_stage.pth`.
  Rename it manually:
  ```bash
  cp output/kokoro-finetune/kokoro-custom-v1/epoch_1st_00001.pth \
     output/kokoro-finetune/kokoro-custom-v1/first_stage.pth
  ```

**`CheckpointLoadError: Checkpoint ... did not load cleanly`**
→ More than 2% of a module's tensors were missing from (or unused in) the checkpoint, so
  training stopped instead of silently starting from untrained weights. `load_checkpoint`
  already handles the `module.` prefix that Stage 2 checkpoints carry (they are saved from
  `DataParallel`-wrapped modules). Check the per-module counts it prints, or run
  `python scripts/verify_checkpoint_load.py --config configs/config.yml --ckpt <file> --mode stage2_resume --old`.
  If the mismatch is intended, raise the limit with `KOKORO_MAX_MISSING_FRAC=0.1`.

**Pronunciation is wrong for custom names after training**
→ Add them to `scripts/lexicon.json` BEFORE running `01_prepare_dataset.py`.
  The lexicon is baked into the training data at prep time — adding words after
  training won't help. Always set up your lexicon first.

**G2P produces phonemes not in Kokoro's vocab**
→ `01_prepare_dataset.py` will print "Unknown phoneme chars" if it detects any.
  These are silently dropped during training — usually harmless, but if you hear
  skipped sounds, adjust the lexicon to avoid those characters.

**`Cannot find StyleTTS2 training scripts` error**
→ The `framework/StyleTTS2/` directory must be present inside the recipe folder.
  If you moved or renamed it, update `framework.styletts2_dir` in `configs/config.yml`.

---

## Output Structure

```
kokoro-recipe/
├── output/
│   └── kokoro-finetune/kokoro-custom-v1/
│       ├── first_stage.pth          ← Stage 1 final checkpoint
│       ├── epoch_1st_00000.pth      ← Stage 1 per-epoch checkpoints
│       ├── epoch_2nd_00000.pth      ← Stage 2 per-epoch checkpoints
│       ├── epoch_2nd_00001.pth
│       └── ...
├── eval/
│   ├── epoch00/
│   │   ├── voicepack.pt             ← speaker style tensor [510, 1, 256]
│   │   ├── kokoro_converted.pth     ← Kokoro-format inference weights
│   │   ├── 01.wav                   ← synthesized test sentences
│   │   ├── 02.wav
│   │   └── manifest.txt             ← which text maps to which wav
│   └── epoch01/ ... epoch09/
├── logs/
│   ├── stage1.log, stage1.pid
│   └── stage2.log, stage2.pid
└── framework/training/              ← generated by 01_prepare_dataset.py
    ├── train_list.txt               ← StyleTTS2 training data list
    └── val_list.txt                 ← validation data list
```
