# ML-SVNIT — TripleEnsemble Image Denoising

**Restormer + NAFNet-w64 + NAFNet-w32** with learned softmax mixing weights,
EMA stabilization, Lookahead-AdamW optimization, and 8-fold test-time augmentation.

Designed for **AWGN denoising (σ = 50)** and compatible with **NTIRE submission format**.

---

# Repository structure

```
ML-SVNIT/
│
├── train.py            # training pipeline
├── inference.py        # inference + NTIRE submission generator
├── patch.py            # fixes torchvision ≥ 0.16 basicsr issue
│
├── Restormer/          # Restormer repository
├── NAFNet/             # NAFNet repository
│
├── checkpoints/        # pretrained backbone checkpoints
│   ├── nafnet_best_sigma50.pth
│   ├── restormer_best_sigma50.pth
│   └── NAFNet-SIDD-width32.pth
│
├── results/            # inference outputs
│
└── submission.zip      # NTIRE submission package
```

---

# Installation

### Clone repository

```bash
git clone https://github.com/aagamjnug20/ML-SVNIT
cd ML-SVNIT
```

### Clone backbone repositories

```bash
git clone https://github.com/swz30/Restormer.git
git clone https://github.com/megvii-research/NAFNet.git
```
---

### Install dependencies

```bash
pip install torch torchvision einops timm lmdb imageio tqdm
```

### ⚠️ Compatibility Patch (MUST RUN)

`basicsr` (used by Restormer & NAFNet) is **incompatible with torchvision ≥ 0.16**.

Without this fix, training and inference will crash with import errors.


### Run patch

```bash
python patch.py
```

---

# Training

Training uses:

* **TripleEnsemble architecture**
* **Lookahead-AdamW optimizer**
* **Warmup Cosine LR schedule**
* **EMA weights**
* **Charbonnier + MSE loss**

---

## Training command

```bash
python train.py \
--train_dirs \
/path/to/DIV2K_train_HR \
/path/to/LSDIR/shard-00 \
\
--val_dirs \
/path/to/DIV2K_valid_HR \
\
--naf_ckpt checkpoints/nafnet_best_sigma50.pth \
--rest_ckpt checkpoints/restormer_best_sigma50.pth \
--modelc_ckpt checkpoints/NAFNet-SIDD-width32.pth \
\
--patch_size 128 \
--batch_size 2 \
--epochs 100 \
\
--sigma 50 \
\
--lr_pretrained 5e-5 \
--lr_new 1e-4 \
\
--warmup_steps 2000 \
\
--num_workers 4 \
\
--save_dir checkpoints
```

---

## Resume training

```bash
python train.py \
--train_dirs ... \
--val_dirs ... \
--naf_ckpt ... \
--rest_ckpt ... \
--modelc_ckpt ... \
--resume checkpoints/best.pth
```

---

# Inference

Inference supports:

* **8-fold TTA**
* **tiled inference for large images**
* **automatic NTIRE submission ZIP generation**

---

## Inference command

```bash
python inference.py \
--ensemble_ckpt checkpoints/best.pth \
\
--naf_ckpt checkpoints/nafnet_best_sigma50.pth \
--rest_ckpt checkpoints/restormer_best_sigma50.pth \
--modelc_ckpt checkpoints/NAFNet-SIDD-width32.pth \
\
--test_dir /path/to/test_images \
\
--out_dir results \
--zip_path submission.zip \
\
--tile 256 \
--overlap 32 \
\
--tta
```

---

## Disable TTA (faster inference)

Remove the flag:

```
--tta
```

This reduces runtime by **~8×**.

---

# Kaggle Usage

Example Kaggle setup:

```python
!git clone https://github.com/aagamjnug20/ML-SVNIT
%cd ML-SVNIT

!git clone https://github.com/swz30/Restormer.git
!git clone https://github.com/megvii-research/NAFNet.git

!pip install einops timm lmdb imageio tqdm -q
```

Run training:

```python
!python train.py \
--train_dirs /kaggle/input/div2k/train \
--val_dirs /kaggle/input/div2k/val \
--naf_ckpt /kaggle/input/checkpoints/nafnet_best_sigma50.pth \
--rest_ckpt /kaggle/input/checkpoints/restormer_best_sigma50.pth \
--modelc_ckpt /kaggle/working/NAFNet-SIDD-width32.pth
```

Run inference:

```python
!python inference.py \
--ensemble_ckpt /kaggle/working/checkpoints/best.pth \
--naf_ckpt /kaggle/input/checkpoints/nafnet_best_sigma50.pth \
--rest_ckpt /kaggle/input/checkpoints/restormer_best_sigma50.pth \
--modelc_ckpt /kaggle/working/NAFNet-SIDD-width32.pth \
--test_dir /kaggle/input/lsdir-div2k-testing \
--tta
```

---

# Method Summary

### Architecture

TripleEnsemble:

```
Restormer
   │
NAFNet-w64
   │
NAFNet-w32
```

Outputs are combined via **learned softmax mixing weights**:

```
output = w1 * Restormer
       + w2 * NAFNet-w64
       + w3 * NAFNet-w32
```

where:

```
w = softmax(logits)
```

---

### Training schedule

| Phase   | Epochs | Description                           |
| ------- | ------ | ------------------------------------- |
| Stage 1 | 0–4    | freeze backbones, train mixing logits |
| Stage 2 | 5–100  | full fine-tuning                      |

---

### Optimization

* **Optimizer:** Lookahead(AdamW)
* **LR schedule:** Warmup Cosine
* **Backbone LR:** 5e-5
* **Logits LR:** 1e-4

---

### Loss

```
Loss = 0.5 × MSE + 0.5 × Charbonnier
```

---

### EMA

Exponential moving average:

```
decay = 0.9999
```

EMA weights are used during:

* validation
* inference

---

### Inference

* **8-fold TTA**
* **tiled inference**
* **overlap blending**

---

# Output

Inference produces:

```
results/
   image_001.png
   image_002.png
   ...
readme.txt
submission.zip
```

The ZIP file is **NTIRE submission compatible**.

---
