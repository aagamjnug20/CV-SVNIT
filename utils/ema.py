"""utils/ema.py — Exponential Moving Average of model weights."""

import torch
import torch.nn as nn


class EMA:
    """
    Maintains a shadow copy of every trainable parameter and blends it
    toward the live weights after each update step.

    Usage
    -----
    ema = EMA(model, decay=0.9999)

    # after each optimizer step:
    ema.update()

    # before validation / inference:
    ema.apply()
    ...
    ema.restore()
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model  = model
        self.decay  = decay
        self.shadow: dict = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup: dict = {}

    @torch.no_grad()
    def update(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

    def apply(self) -> None:
        """Swap live weights → EMA weights (saves live weights for restore)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore live weights after EMA evaluation."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()
