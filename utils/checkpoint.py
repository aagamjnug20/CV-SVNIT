"""utils/checkpoint.py — Save and load training checkpoints."""

import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

from utils.ema import EMA


def save_ckpt(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    ema: EMA,
    scheduler,
    epoch: int,
    best_psnr: float,
) -> None:
    """Save a full training checkpoint and mirror it to <output_dir>/best.pth."""
    data = {
        "epoch":      epoch,
        "best_psnr":  best_psnr,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
        "ema_shadow": ema.shadow,
    }
    torch.save(data, path)
    best_mirror = os.path.join(os.path.dirname(path), "best.pth")
    shutil.copy(path, best_mirror)
    print(f"  [ckpt] saved → {path}  (epoch={epoch}  best_psnr={best_psnr:.4f})", flush=True)


def load_ckpt(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    ema: EMA,
    scheduler,
) -> tuple[int, float]:
    """Load a checkpoint.  Returns (start_epoch, best_psnr)."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    device = next(model.parameters()).device
    ema.shadow = {k: v.to(device) for k, v in ckpt["ema_shadow"].items()}
    print(f"  [ckpt] resumed ← {path}  (epoch={ckpt['epoch']}  best_psnr={ckpt['best_psnr']:.4f})")
    return ckpt["epoch"], ckpt["best_psnr"]
