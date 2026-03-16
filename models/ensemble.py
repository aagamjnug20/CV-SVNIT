"""
models/ensemble.py — TripleEnsemble and backbone loader functions.

Each backbone is loaded from its own pre-trained checkpoint.
The ensemble learns three softmax-normalised mixing weights (logits)
on top of the frozen (then unfrozen) backbones.
"""

import sys
import math
import importlib
import importlib.util

import torch
import torch.nn as nn

import config


# ── Module loader helper ──────────────────────────────────────────────────────

def _load_module(mod_name: str, path: str):
    """Dynamically load a Python file as a named module."""
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _extract_weights(raw: dict) -> dict:
    """Pull weights from any common checkpoint format."""
    for key in ("params_ema", "params", "model", "state_dict"):
        if key in raw:
            sd = raw[key]
            return {k.replace("module.", ""): v for k, v in sd.items()}
    return {k.replace("module.", ""): v for k, v in raw.items()}


# ── Backbone builders ─────────────────────────────────────────────────────────

def load_restormer(ckpt_path: str) -> nn.Module:
    mod = _load_module("restormer_arch", config.REST_ARCH)
    net = mod.Restormer(
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type="BiasFree", dual_pixel_task=False,
    )
    raw = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(_extract_weights(raw), strict=True)
    print(f"[Restormer]  ✓  {sum(p.numel() for p in net.parameters())/1e6:.1f}M params")
    return net


def load_nafnet_w64(ckpt_path: str) -> nn.Module:
    # Load optional dependencies first (arch_util, local_arch)
    for dep in (config.NAF_DEP_ARCH, config.NAF_DEP_LOC):
        if dep and __import__("os").path.exists(dep):
            name = "basicsr.models.archs." + __import__("os").path.basename(dep)[:-3]
            _load_module(name, dep)
    mod = _load_module("nafnet_w64_arch", config.NAF_ARCH)
    net = mod.NAFNet(
        img_channel=3, width=64, middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2],
    )
    raw = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(_extract_weights(raw), strict=True)
    print(f"[NAFNet-w64] ✓  {sum(p.numel() for p in net.parameters())/1e6:.1f}M params")
    return net


def load_nafnet_w32(ckpt_path: str) -> nn.Module:
    mod = _load_module("nafnet_w32_arch", config.NAF_ARCH)
    net = mod.NAFNet(
        img_channel=3, width=32, middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2],
    )
    raw = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(_extract_weights(raw), strict=True)
    print(f"[NAFNet-w32] ✓  {sum(p.numel() for p in net.parameters())/1e6:.1f}M params")
    return net


# ── Ensemble ──────────────────────────────────────────────────────────────────

class TripleEnsemble(nn.Module):
    """
    Weighted sum of three denoisers with learnable softmax mixing weights.

        output = w[0]*Restormer(x) + w[1]*NAFNet-w64(x) + w[2]*NAFNet-w32(x)

    where  w = softmax(logits),  logits ∈ ℝ³  (trainable parameter).
    """

    def __init__(
        self,
        restormer: nn.Module,
        nafnet:    nn.Module,
        model_c:   nn.Module,
        weights_init: tuple = (0.34, 0.33, 0.33),
    ):
        super().__init__()
        self.restormer = restormer
        self.nafnet    = nafnet
        self.model_c   = model_c
        logits = torch.tensor([math.log(w + 1e-8) for w in weights_init])
        self.logits = nn.Parameter(logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.logits, dim=0)
        return w[0] * self.restormer(x) + w[1] * self.nafnet(x) + w[2] * self.model_c(x)

    @property
    def weights(self) -> list[float]:
        return torch.softmax(self.logits, dim=0).tolist()


# ── Convenience builder ───────────────────────────────────────────────────────

def build_ensemble(weights_init=(0.34, 0.33, 0.33)) -> TripleEnsemble:
    """Build and return a TripleEnsemble from checkpoints defined in config."""
    restormer = load_restormer(config.REST_CKPT)
    nafnet    = load_nafnet_w64(config.NAF_CKPT)
    model_c   = load_nafnet_w32(config.MODEL_C_CKPT)
    model     = TripleEnsemble(restormer, nafnet, model_c, weights_init)
    print(f"[Ensemble] initial weights = {[f'{w:.4f}' for w in model.weights]}")
    return model


# ── Freeze / unfreeze helpers ─────────────────────────────────────────────────

def freeze_backbones(model: TripleEnsemble) -> None:
    for p in (
        list(model.restormer.parameters())
        + list(model.nafnet.parameters())
        + list(model.model_c.parameters())
    ):
        p.requires_grad_(False)
    print("[Freeze] Backbones frozen — training logits only")


def unfreeze_backbones(model: TripleEnsemble) -> None:
    for p in (
        list(model.restormer.parameters())
        + list(model.nafnet.parameters())
        + list(model.model_c.parameters())
    ):
        p.requires_grad_(True)
    print("[Unfreeze] Backbones unfrozen — full fine-tuning")
