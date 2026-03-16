"""
engine/trainer.py — Training utilities.

Contains:
  - get_param_groups        : differential LRs for backbone vs new params
  - Lookahead               : optimizer wrapper
  - WarmupCosineScheduler   : LR schedule
  - train_one_epoch         : single epoch loop
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from utils.losses   import combined_loss
from utils.metrics  import psnr
from utils.ema      import EMA


# ── Param groups (differential LR) ───────────────────────────────────────────

def get_param_groups(model: nn.Module) -> list[dict]:
    """
    Four groups:
      1. backbone weights  + weight-decay
      2. backbone norm/bias — no weight-decay
      3. new weights       + weight-decay
      4. new norm/bias     — no weight-decay
    """
    backbone_ids = {
        id(p)
        for p in (
            list(model.restormer.parameters())
            + list(model.nafnet.parameters())
            + list(model.model_c.parameters())
        )
    }

    no_wd_ids: set[int] = set()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d)):
            for p in m.parameters():
                no_wd_ids.add(id(p))
    for name, p in model.named_parameters():
        if name.endswith(".bias"):
            no_wd_ids.add(id(p))

    bb_wd, bb_nowd, new_wd, new_nowd = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bb    = id(p) in backbone_ids
        is_no_wd = id(p) in no_wd_ids
        if is_bb:
            (bb_nowd if is_no_wd else bb_wd).append(p)
        else:
            (new_nowd if is_no_wd else new_wd).append(p)

    groups = [
        {"params": bb_wd,    "lr": config.LR_PRETRAINED, "weight_decay": config.WEIGHT_DECAY},
        {"params": bb_nowd,  "lr": config.LR_PRETRAINED, "weight_decay": 0.0},
        {"params": new_wd,   "lr": config.LR_NEW,        "weight_decay": config.WEIGHT_DECAY},
        {"params": new_nowd, "lr": config.LR_NEW,        "weight_decay": 0.0},
    ]
    total = sum(len(g["params"]) for g in groups)
    print(f"[ParamGroups] {total} tensors across 4 groups")
    return groups


# ── Lookahead ─────────────────────────────────────────────────────────────────

class Lookahead(torch.optim.Optimizer):
    """
    Lookahead wrapper (Zhang et al., 2019).
    Wraps any base optimizer; every k inner steps, slow weights are
    interpolated toward the fast weights.
    """

    def __init__(self, base_optimizer: torch.optim.Optimizer, alpha: float = 0.5, k: int = 5):
        self.base_optimizer = base_optimizer
        self.alpha          = alpha
        self.k              = k
        self._step_count    = 0
        self.param_groups   = base_optimizer.param_groups
        self.state          = base_optimizer.state
        self.defaults       = base_optimizer.defaults
        self.slow_weights   = [
            [p.clone().detach().requires_grad_(False) for p in g["params"]]
            for g in base_optimizer.param_groups
        ]

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self._step_count += 1
        if self._step_count % self.k == 0:
            for group, slow_group in zip(self.base_optimizer.param_groups, self.slow_weights):
                for p_fast, p_slow in zip(group["params"], slow_group):
                    if p_fast.grad is None:
                        continue
                    p_slow.data.add_(self.alpha * (p_fast.data - p_slow.data))
                    p_fast.data.copy_(p_slow.data)
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, sd):
        self.base_optimizer.load_state_dict(sd)
        self.param_groups = self.base_optimizer.param_groups

    def __getattr__(self, name):
        return getattr(self.base_optimizer, name)


# ── Scheduler ─────────────────────────────────────────────────────────────────

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warm-up followed by cosine annealing down to min_lr."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr       = min_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch
        lrs  = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                lr = base_lr * (step + 1) / max(1, self.warmup_steps)
            else:
                prog   = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * prog))
                lr     = self.min_lr + (base_lr - self.min_lr) * cosine
            lrs.append(lr)
        return lrs


# ── One-epoch training loop ───────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler:    torch.amp.GradScaler,
    ema:       EMA,
    device:    torch.device,
) -> dict:
    model.train()
    total_loss, total_psnr, n = 0.0, 0.0, 0

    for step, (noisy, clean) in enumerate(loader):
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            pred = model(noisy)
            loss = combined_loss(pred, clean)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update()

        step_psnr  = psnr(pred.detach(), clean)
        total_loss += loss.item()
        total_psnr += step_psnr
        n          += 1

        if step % 50 == 0:
            print(
                f"  step {step}/{len(loader)}"
                f"  loss={loss.item():.6f}"
                f"  psnr={step_psnr:.4f} dB",
                flush=True,
            )

    torch.cuda.empty_cache()
    return {"loss": total_loss / n, "psnr": total_psnr / n}


# ── Optimizer factory (used at epoch 0 and epoch 5) ──────────────────────────

def build_optimizer_and_scheduler(
    model,
    total_steps: int,
    warmup_steps: int = 0,
    resume_steps: int = 0,
) -> tuple[Lookahead, WarmupCosineScheduler]:
    """Return a fresh (Lookahead-wrapped AdamW, WarmupCosineScheduler) pair."""
    param_groups = get_param_groups(model)
    base_opt     = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    optimizer    = Lookahead(base_opt, alpha=0.5, k=5)
    scheduler    = WarmupCosineScheduler(
        optimizer.base_optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps - resume_steps,
        min_lr=config.LR_MIN,
        last_epoch=-1,
    )
    return optimizer, scheduler
