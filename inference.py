"""
inference.py — TTA inference entry point for ML-SVNIT.

Quick start
-----------
python inference.py

# Kaggle
TEST_DIR=/kaggle/input/lsdir-div2k-testing \\
OUTPUT_DIR=/kaggle/working                  \\
CKPT_DIR=/kaggle/input/pretrained           \\
python inference.py

# Disable TTA (faster, slightly lower PSNR)
USE_TTA=0 python inference.py

# Reduce tile size if you hit OOM
INFER_TILE=128 python inference.py
"""

import os
import sys
import glob
import time
import zipfile
import textwrap

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from PIL import Image
from tqdm import tqdm

# ── Make Restormer / NAFNet source importable ─────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "Restormer"))
sys.path.insert(0, os.path.join(BASE_DIR, "NAFNet"))

import config
from models.ensemble import build_ensemble


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ── Tiled inference (CPU tensors in, CPU tensors out) ─────────────────────────

@torch.no_grad()
def infer_tiled(
    model:   nn.Module,
    noisy_t: torch.Tensor,
    tile:    int,
    overlap: int,
    device:  torch.device,
) -> torch.Tensor:
    """
    noisy_t : (1,3,H,W) float32 on CPU
    returns : same shape, float32 on CPU, values in [0,1]
    """
    _, _, H, W = noisy_t.shape
    step = tile - overlap
    out  = torch.zeros_like(noisy_t)
    cnt  = torch.zeros_like(noisy_t)

    ys = sorted(set(list(range(0, max(H - tile, 0), step)) + [max(H - tile, 0)]))
    xs = sorted(set(list(range(0, max(W - tile, 0), step)) + [max(W - tile, 0)]))

    for y in ys:
        y2 = min(y + tile, H);  y1 = y2 - tile
        for x in xs:
            x2 = min(x + tile, W);  x1 = x2 - tile
            patch = noisy_t[:, :, y1:y2, x1:x2].to(device)
            with autocast():
                pred = model(patch).float().cpu()
            out[:, :, y1:y2, x1:x2] += pred
            cnt[:, :, y1:y2, x1:x2] += 1

    return torch.clamp(out / cnt.clamp(min=1), 0.0, 1.0)


# ── 8-fold TTA (4 rotations × 2 flips) ───────────────────────────────────────

@torch.no_grad()
def tta_forward(
    model:   nn.Module,
    noisy_t: torch.Tensor,
    tile:    int,
    overlap: int,
    device:  torch.device,
) -> torch.Tensor:
    preds = []
    for flip in (False, True):
        t = noisy_t.flip(-1) if flip else noisy_t
        for k in range(4):
            aug  = torch.rot90(t, k, dims=[-2, -1])
            pred = infer_tiled(model, aug, tile, overlap, device)
            pred = torch.rot90(pred, -k, dims=[-2, -1])
            if flip:
                pred = pred.flip(-1)
            preds.append(pred)
    return torch.stack(preds).mean(0)


# ── Checkpoint loader (applies EMA weights) ───────────────────────────────────

def load_for_inference(model: nn.Module, ckpt_path: str, device: torch.device) -> nn.Module:
    print(f"\nLoading checkpoint: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu")

    if isinstance(raw, dict):
        saved_ep   = raw.get("epoch",     "?")
        saved_psnr = raw.get("best_psnr", "?")
        print(f"  Saved epoch : {saved_ep}  |  best_psnr : {saved_psnr}")

        # Load model weights (backbone fine-tune if present, otherwise keep pre-trained)
        model_sd = raw.get("model", {})
        if model_sd:
            missing, unexpected = model.load_state_dict(
                {k.replace("module.", ""): v for k, v in model_sd.items()}, strict=False
            )
            print(f"  model.state_dict loaded  (missing={len(missing)}, unexpected={len(unexpected)})")

        # Override with EMA weights where available
        ema_sd  = raw.get("ema_shadow", {})
        applied = 0
        for name, param in model.named_parameters():
            if name in ema_sd:
                param.data.copy_(ema_sd[name].to(device))
                applied += 1
        if applied:
            print(f"  EMA weights applied to {applied} parameters ✓")

    model.eval().to(device)
    w = model.weights
    print(
        f"  Ensemble weights → "
        f"Restormer:{w[0]:.4f}  NAFNet-w64:{w[1]:.4f}  NAFNet-w32:{w[2]:.4f}"
    )
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Device  : {device}")
    if device.type == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  TTA     : {config.USE_TTA}")
    print(f"  Tile    : {config.INFER_TILE}  Overlap : {config.INFER_OVERLAP}")
    print(f"{'='*60}\n")

    os.makedirs(config.RESULT_DIR, exist_ok=True)

    # ── Build & load model ────────────────────────────────────────────────────
    model    = build_ensemble().to(device)
    best_pth = os.path.join(config.SAVE_DIR, "best.pth")
    if not os.path.isfile(best_pth):
        raise FileNotFoundError(
            f"No checkpoint found at {best_pth}\n"
            "Run train.py first, or set SAVE_DIR to the directory containing best.pth."
        )
    model = load_for_inference(model, best_pth, device)

    # ── Collect test images ───────────────────────────────────────────────────
    paths: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        paths += glob.glob(os.path.join(config.TEST_DIR, ext))
    paths = sorted(paths)
    print(f"  Images found : {len(paths)}")
    if not paths:
        raise FileNotFoundError(f"No images found in TEST_DIR={config.TEST_DIR}")

    infer_fn   = tta_forward if config.USE_TTA else infer_tiled
    total_time = 0.0
    outer      = tqdm(paths, desc="Inference", unit="img")

    for i, src in enumerate(outer):
        fname  = os.path.basename(src)
        dst    = os.path.join(config.RESULT_DIR, fname)
        outer.set_postfix(file=fname)

        img_pil        = Image.open(src).convert("RGB")
        orig_W, orig_H = img_pil.size
        noisy_np       = np.array(img_pil, dtype=np.float32) / 255.0
        noisy_t        = torch.from_numpy(noisy_np).permute(2, 0, 1).unsqueeze(0)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        out_t = infer_fn(model, noisy_t, config.INFER_TILE, config.INFER_OVERLAP, device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed     = time.perf_counter() - t0
        total_time += elapsed

        out_np = (out_t.squeeze(0).permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(out_np).save(dst, format="PNG", compress_level=0)   # lossless

        avg = total_time / (i + 1)
        eta = avg * (len(paths) - i - 1)
        tqdm.write(
            f"  [{i+1:>3}/{len(paths)}] {fname}  {orig_W}×{orig_H}"
            f"  | {elapsed:.2f}s  | avg {avg:.2f}s  | ETA {eta/60:.1f} min"
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── readme.txt ────────────────────────────────────────────────────────────
    avg_time = total_time / max(len(paths), 1)
    w        = model.weights
    readme   = textwrap.dedent(f"""\
        runtime per image [s] : {avg_time:.2f}
        CPU[1] / GPU[0] : {1 - config.NTIRE_USES_GPU}
        Extra Data [1] / No Extra Data [0] : {config.NTIRE_EXTRA_DATA}
        Other description : {config.NTIRE_DESC} \
Ensemble weights: Restormer={w[0]:.3f}, NAFNet-w64={w[1]:.3f}, NAFNet-w32={w[2]:.3f}.
    """)
    readme_path = os.path.join(config.RESULT_DIR, "readme.txt")
    with open(readme_path, "w") as f:
        f.write(readme)
    print(f"\n  readme.txt written.\n{readme}")

    # ── ZIP ───────────────────────────────────────────────────────────────────
    pngs = sorted(glob.glob(os.path.join(config.RESULT_DIR, "*.png")))
    with zipfile.ZipFile(config.ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in pngs:
            zf.write(p, arcname=os.path.basename(p))
        zf.write(readme_path, arcname="readme.txt")

    size_mb = os.path.getsize(config.ZIP_PATH) / 1e6
    print(f"  ZIP   : {config.ZIP_PATH}  ({size_mb:.1f} MB)")
    print(f"  Files : {len(pngs)} PNGs + readme.txt")
    print(f"\n  ✓ Submission ready at : {config.ZIP_PATH}\n")


if __name__ == "__main__":
    main()
