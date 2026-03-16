"""engine/validator.py — Tiled forward pass and validation loop."""

import torch
import torch.nn as nn

import config
from utils.metrics import psnr


# ── Tiled forward (validation — tensors already on device) ───────────────────

@torch.no_grad()
def forward_tiled(
    model:   nn.Module,
    x:       torch.Tensor,
    tile:    int = 128,
    overlap: int = 8,
) -> torch.Tensor:
    """
    Split a single (1,C,H,W) on-device tensor into overlapping tiles,
    run the model on each, and blend the results.
    Used during validation where images are already on GPU.
    """
    _, _, h, w = x.shape
    stride = tile - overlap
    out    = torch.zeros_like(x, dtype=torch.float32)
    count  = torch.zeros_like(x, dtype=torch.float32)

    for top in range(0, h, stride):
        for left in range(0, w, stride):
            t = min(top,  h - tile)
            l = min(left, w - tile)
            patch = x[:, :, t : t + tile, l : l + tile]
            with torch.amp.autocast("cuda"):
                pred = model(patch).float()
            out  [:, :, t : t + tile, l : l + tile] += pred
            count[:, :, t : t + tile, l : l + tile] += 1.0

    return out / count.clamp(min=1.0)


# ── Validation loop ───────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, ema, device: torch.device) -> float:
    """
    Evaluate up to VAL_SUBSET images from *loader* using EMA weights.
    Returns average PSNR.
    """
    ema.apply()
    model.eval()
    total_psnr, count = 0.0, 0

    for i, (noisy, clean) in enumerate(loader):
        if i >= config.VAL_SUBSET:
            break
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        pred  = forward_tiled(model, noisy, tile=config.VAL_TILE, overlap=config.VAL_OVERLAP)
        total_psnr += psnr(pred, clean)
        count      += 1

    torch.cuda.empty_cache()
    ema.restore()
    return total_psnr / max(count, 1)
