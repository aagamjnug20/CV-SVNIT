"""utils/metrics.py — Evaluation metrics."""

import math
import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1))
        return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse.item())
