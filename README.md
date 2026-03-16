# CV-SVNIT — TripleEnsemble Image Denoising

**Restormer + NAFNet-w64 + NAFNet-w32** with learned softmax mixing weights,
EMA, Lookahead-AdamW, and 8-fold TTA.

---

## Repo structure

```
CV-SVNIT/
├── config.py              ← all paths & hyper-params (env-var overridable)
├── train.py               ← training entry point
├── inference.py           ← TTA inference + NTIRE ZIP entry point
├── requirements.txt
│
├── models/
│   └── ensemble.py        ← TripleEnsemble + backbone loaders
│
├── datasets/
│   └── dataset.py         ← DIV2KDataset (train / val / test)
│
├── engine/
│   ├── trainer.py         ← Lookahead, WarmupCosine, train_one_epoch
│   └── validator.py       ← tiled forward, validate()
│
├── utils/
│   ├── losses.py          ← Charbonnier + MSE
│   ├── metrics.py         ← PSNR
│   ├── ema.py             ← EMA
│   └── checkpoint.py      ← save_ckpt / load_ckpt
│
├── Restormer/             ← git submodule
├── NAFNet/                ← git submodule
│
├── checkpoints/           ← put pre-trained .pth files here
│   ├── nafnet_best_sigma50.pth
│   ├── restormer_best_sigma50.pth
│   └── NAFNet-SIDD-width32.pth
│
├── datasets/
│   ├── train/             ← training PNGs
│   └── val/               ← validation PNGs
│
└── outputs/               ← created automatically
    ├── checkpoints/best.pth
    ├── results/
    └── submission.zip
```

---

## Quick start (local / Colab / HPC)

```bash
git clone --recurse-submodules https://github.com/aagamjnug20/CV-SVNIT
cd CV-SVNIT
pip install -r requirements.txt

python train.py
python inference.py
```

---

## Kaggle usage

```python
# In a Kaggle notebook cell:
!git clone --recurse-submodules https://github.com/aagamjnug20/CV-SVNIT
%cd CV-SVNIT
!pip install -r requirements.txt -q
```

```python
import os
os.environ["TRAIN_DIRS"]  = "/kaggle/input/div2k/train,/kaggle/input/lsdir/shard-00"
os.environ["VAL_DIRS"]    = "/kaggle/input/div2k/val"
os.environ["TEST_DIR"]    = "/kaggle/input/lsdir-div2k-testing"
os.environ["CKPT_DIR"]    = "/kaggle/input/pretrained"
os.environ["OUTPUT_DIR"]  = "/kaggle/working"

!python train.py
!python inference.py
```

---

## Environment variables (all optional)

| Variable        | Default                   | Description                              |
|-----------------|---------------------------|------------------------------------------|
| `TRAIN_DIRS`    | `datasets/train`          | Comma-separated training dirs            |
| `VAL_DIRS`      | `datasets/val`            | Comma-separated validation dirs          |
| `TEST_DIR`      | `datasets/test`           | Test images for inference                |
| `CKPT_DIR`      | `checkpoints`             | Pre-trained backbone checkpoints         |
| `OUTPUT_DIR`    | `outputs`                 | All outputs (checkpoints, results, zip)  |
| `RESUME`        | *(empty)*                 | Path to checkpoint to resume from        |
| `USE_TTA`       | `1`                       | `0` to disable 8-fold TTA               |
| `INFER_TILE`    | `256`                     | Tile size for inference (reduce if OOM)  |
| `INFER_OVERLAP` | `32`                      | Overlap between tiles                    |
| `BATCH_SIZE`    | `2`                       | Training batch size                      |
| `NUM_EPOCHS`    | `100`                     | Total training epochs                    |
| `SIGMA`         | `50.0`                    | AWGN noise level (0–255 scale)           |

---

## Method summary

- **Architecture** : TripleEnsemble — three denoising networks whose outputs are blended with
  learned softmax weights (`logits` parameter).
- **Training** :
  - Epochs 0–4  : backbones frozen; only mixing logits trained.
  - Epochs 5–99 : full fine-tuning with differential LR
    (5e-5 for backbones, 1e-4 for logits).
  - Optimizer : Lookahead(AdamW) + WarmupCosine schedule.
  - Loss : 0.5 × MSE + 0.5 × Charbonnier.
  - EMA decay = 0.9999 applied during validation and inference.
- **Inference** : 8-fold TTA (4 rotations × 2 flips) + overlapping tiled processing.
