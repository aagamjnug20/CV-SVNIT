"""
train.py — Training entry point for CV-SVNIT.

Quick start
-----------
# local (after pip install -r requirements.txt)
python train.py

# resume from a checkpoint
RESUME=outputs/checkpoints/best.pth python train.py

# Kaggle (set paths via env)
TRAIN_DIRS=/kaggle/input/div2k/train,/kaggle/input/lsdir/shard-00 \\
VAL_DIRS=/kaggle/input/div2k/val                                   \\
CKPT_DIR=/kaggle/input/pretrained                                  \\
OUTPUT_DIR=/kaggle/working                                         \\
python train.py
"""

import os
import sys
import time

import torch
from torch.utils.data import DataLoader

# ── Make Restormer / NAFNet source importable ──────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "Restormer"))
sys.path.insert(0, os.path.join(BASE_DIR, "NAFNet"))

import config
from models.ensemble  import build_ensemble, freeze_backbones, unfreeze_backbones
from datasets.dataset import DIV2KDataset
from engine.trainer   import train_one_epoch, build_optimizer_and_scheduler
from engine.validator import validate
from utils.ema        import EMA
from utils.checkpoint import save_ckpt, load_ckpt


def main() -> None:
    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(config.SAVE_DIR,   exist_ok=True)
    os.makedirs(config.RESULT_DIR, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_ensemble().to(device)

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_ds = DIV2KDataset(config.TRAIN_DIRS, config.PATCH_SIZE, mode="train", sigma=config.SIGMA)
    val_ds   = DIV2KDataset(config.VAL_DIRS,   config.PATCH_SIZE, mode="val",   sigma=config.SIGMA)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    total_steps = config.NUM_EPOCHS * len(train_loader)
    print(f"Train batches/epoch : {len(train_loader)}  |  Total steps : {total_steps}")

    # ── Optimizer / scheduler / EMA / scaler ──────────────────────────────────
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        total_steps=total_steps,
        warmup_steps=config.WARMUP_STEPS,
    )
    ema    = EMA(model, decay=config.EMA_DECAY)
    scaler = torch.amp.GradScaler("cuda")

    # ── Optional resume ───────────────────────────────────────────────────────
    start_epoch, best_psnr = 0, 0.0
    if config.RESUME and os.path.isfile(config.RESUME):
        start_epoch, best_psnr = load_ckpt(config.RESUME, model, optimizer, ema, scheduler)

    best_path = os.path.join(config.SAVE_DIR, "best.pth")

    print(f"\n{'='*60}")
    print(f"Training   epochs={config.NUM_EPOCHS}  sigma={config.SIGMA}  patch={config.PATCH_SIZE}  bs={config.BATCH_SIZE}")
    print(f"Loss: Charbonnier + MSE  |  LR_pre={config.LR_PRETRAINED}  LR_new={config.LR_NEW}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        t0 = time.time()

        # Epoch 0 : freeze backbones — train only the mixing logits
        if epoch == 0 and start_epoch == 0:
            freeze_backbones(model)

        # Epoch 5 : unfreeze everything — rebuild opt/scheduler for full fine-tune
        if epoch == 5 and start_epoch <= 5:
            unfreeze_backbones(model)
            optimizer, scheduler = build_optimizer_and_scheduler(
                model,
                total_steps=total_steps,
                warmup_steps=0,
                resume_steps=epoch * len(train_loader),
            )

        # ── Train ─────────────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, ema, device
        )

        # ── Validate ──────────────────────────────────────────────────────────
        if (epoch + 1) % config.VAL_EVERY == 0 or epoch == 0:
            val_psnr = validate(model, val_loader, ema, device)
        else:
            val_psnr = best_psnr

        # ── Log ───────────────────────────────────────────────────────────────
        elapsed = time.time() - t0
        w       = model.weights
        print(
            f"Epoch [{epoch+1:3d}/{config.NUM_EPOCHS}]  "
            f"Loss={train_metrics['loss']:.6f}  "
            f"TrainPSNR={train_metrics['psnr']:.4f}  "
            f"ValPSNR={val_psnr:.4f}  "
            f"w=[{w[0]:.3f},{w[1]:.3f},{w[2]:.3f}]  "
            f"[{elapsed:.0f}s]",
            flush=True,
        )

        # ── Save best ─────────────────────────────────────────────────────────
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_ckpt(best_path, model, optimizer, ema, scheduler, epoch + 1, best_psnr)

    print(f"\nDone. Best Val PSNR : {best_psnr:.4f} dB")


if __name__ == "__main__":
    main()
