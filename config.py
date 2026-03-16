"""
config.py — Central configuration for ML-SVNIT pipeline.

Every path is resolved from environment variables so the same code
runs on Kaggle, Colab, a local GPU, or an HPC cluster without
editing a single line.

Usage
-----
# defaults (local clone)
python train.py

# Kaggle / custom paths
TRAIN_DIRS=/kaggle/input/div2k/train,/kaggle/input/lsdir/shard-00 \
CKPT_DIR=/kaggle/input/pretrained                                  \
OUTPUT_DIR=/kaggle/working                                         \
python train.py
"""

import os
import math

# ── Repo root (wherever this file lives) ──────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))

# ── Directories (all overridable via env) ─────────────────────────────────────
TRAIN_DIRS  = os.getenv("TRAIN_DIRS",  os.path.join(BASE_DIR, "datasets/train")).split(",")
VAL_DIRS    = os.getenv("VAL_DIRS",    os.path.join(BASE_DIR, "datasets/val")).split(",")
TEST_DIR    = os.getenv("TEST_DIR",    os.path.join(BASE_DIR, "datasets/test"))
CKPT_DIR    = os.getenv("CKPT_DIR",   os.path.join(BASE_DIR, "checkpoints"))
OUTPUT_DIR  = os.getenv("OUTPUT_DIR", os.path.join(BASE_DIR, "outputs"))

# ── Sub-dirs derived from OUTPUT_DIR ──────────────────────────────────────────
SAVE_DIR   = os.path.join(OUTPUT_DIR, "checkpoints")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
ZIP_PATH   = os.path.join(OUTPUT_DIR, "submission.zip")

# ── Pre-trained backbone checkpoints ──────────────────────────────────────────
NAF_CKPT     = os.getenv("NAF_CKPT",    os.path.join(CKPT_DIR, "nafnet_best_sigma50.pth"))
REST_CKPT    = os.getenv("REST_CKPT",   os.path.join(CKPT_DIR, "restormer_best_sigma50.pth"))
MODEL_C_CKPT = os.getenv("MODEL_C_CKPT",os.path.join(CKPT_DIR, "NAFNet-SIDD-width32.pth"))

# ── Arch source files (relative to repo root) ─────────────────────────────────
REST_ARCH    = os.path.join(BASE_DIR, "Restormer/basicsr/models/archs/restormer_arch.py")
NAF_ARCH     = os.path.join(BASE_DIR, "NAFNet/basicsr/models/archs/NAFNet_arch.py")
NAF_DEP_ARCH = os.path.join(BASE_DIR, "NAFNet/basicsr/models/archs/arch_util.py")
NAF_DEP_LOC  = os.path.join(BASE_DIR, "NAFNet/basicsr/models/archs/local_arch.py")

# ── Training hyper-params ─────────────────────────────────────────────────────
PATCH_SIZE    = int(os.getenv("PATCH_SIZE",   128))
BATCH_SIZE    = int(os.getenv("BATCH_SIZE",   2))
NUM_EPOCHS    = int(os.getenv("NUM_EPOCHS",   100))
SIGMA         = float(os.getenv("SIGMA",      50.0))
WARMUP_STEPS  = int(os.getenv("WARMUP_STEPS", 2000))
LR_PRETRAINED = float(os.getenv("LR_PRE",    5e-5))
LR_NEW        = float(os.getenv("LR_NEW",    1e-4))
WEIGHT_DECAY  = float(os.getenv("WD",        1e-4))
LR_MIN        = float(os.getenv("LR_MIN",    1e-7))
EMA_DECAY     = float(os.getenv("EMA_DECAY", 0.9999))
GRAD_CLIP     = float(os.getenv("GRAD_CLIP", 0.5))
NUM_WORKERS   = int(os.getenv("NUM_WORKERS",  0))

# ── Validation ────────────────────────────────────────────────────────────────
VAL_TILE      = int(os.getenv("VAL_TILE",    128))
VAL_OVERLAP   = int(os.getenv("VAL_OVERLAP", 8))
VAL_EVERY     = int(os.getenv("VAL_EVERY",   1))
VAL_SUBSET    = int(os.getenv("VAL_SUBSET",  10))

# ── Inference / TTA ───────────────────────────────────────────────────────────
USE_TTA       = os.getenv("USE_TTA",  "1") == "1"
INFER_TILE    = int(os.getenv("INFER_TILE",    256))
INFER_OVERLAP = int(os.getenv("INFER_OVERLAP", 32))

# ── Resume ────────────────────────────────────────────────────────────────────
RESUME = os.getenv("RESUME", "")   # path to a .pth to resume from; empty = fresh

# ── NTIRE readme fields ───────────────────────────────────────────────────────
NTIRE_USES_GPU   = 1
NTIRE_EXTRA_DATA = 1
NTIRE_DESC = (
    "TripleEnsemble of Restormer + NAFNet-w64 + NAFNet-w32 with learned "
    "softmax mixing weights, fine-tuned on DIV2K + LSDIR for AWGN "
    "denoising at sigma=50. Inference uses 8-fold TTA (flip x 4 rotations) "
    "and tiled processing with overlap blending."
)
