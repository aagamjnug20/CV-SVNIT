"""utils/losses.py — Loss functions."""

import torch
import torch.nn.functional as F


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """0.5 × MSE + 0.5 × Charbonnier."""
    return 0.5 * F.mse_loss(pred, target) + 0.5 * charbonnier_loss(pred, target)
