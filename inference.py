# import subprocess
# subprocess.run(["git", "clone", "https://github.com/swz30/Restormer.git", "/kaggle/working/Restormer"], check=False)
# subprocess.run(["git", "clone", "https://github.com/megvii-research/NAFNet.git",  "/kaggle/working/NAFNet"],    check=False)
# subprocess.run(["pip", "install", "einops", "timm", "lmdb", "imageio", "-q"],      check=False)

# ============================================================
#  ensemble_inference.py — TripleEnsemble  |  NTIRE Submission
#  Kaggle-ready  •  8-fold TTA  •  Lossless PNG  •  Auto ZIP
# ============================================================

import os, sys, glob, math, zipfile, time, textwrap, importlib, importlib.util
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

sys.path.insert(0, "/kaggle/working/Restormer")
sys.path.insert(0, "/kaggle/working/NAFNet")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse

def parse_args():

    parser = argparse.ArgumentParser("TripleEnsemble Inference")

    parser.add_argument("--ensemble_ckpt", required=True)

    parser.add_argument("--naf_ckpt", required=True)
    parser.add_argument("--rest_ckpt", required=True)
    parser.add_argument("--modelc_ckpt", required=True)

    parser.add_argument("--test_dir", required=True)

    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--zip_path", default="submission.zip")

    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)

    parser.add_argument("--tta", action="store_true")

    return parser.parse_args()



# ============================================================
#  CONFIG  ← only edit this block
# ============================================================
args = parse_args()

CFG = dict(
    ensemble_ckpt=args.ensemble_ckpt,
    naf_ckpt=args.naf_ckpt,
    rest_ckpt=args.rest_ckpt,
    model_c_ckpt=args.modelc_ckpt,

    rest_arch="Restormer/basicsr/models/archs/restormer_arch.py",
    naf_arch="NAFNet/basicsr/models/archs/NAFNet_arch.py",

    naf_deps=[
        "NAFNet/basicsr/models/archs/arch_util.py",
        "NAFNet/basicsr/models/archs/local_arch.py"
    ],

    test_dir=args.test_dir,

    out_dir=args.out_dir,
    zip_path=args.zip_path,

    use_tta=args.tta,

    tile_size=args.tile,
    tile_overlap=args.overlap,

    uses_gpu=1,
    extra_data=1,

    other_desc="TripleEnsemble Restormer + NAFNet-w64 + NAFNet-w32"
)

os.makedirs(CFG["out_dir"], exist_ok=True)

# ============================================================
#  Arch loader helper
# ============================================================
def _load_mod(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

def _extract_sd(raw):
    """Pull weights from any common checkpoint format."""
    if not isinstance(raw, dict):
        return raw
    for key in ("params_ema", "params", "state_dict"):
        if key in raw:
            return {k.replace("module.", ""): v for k, v in raw[key].items()}
    # plain dict — return as-is after stripping module prefix
    return {k.replace("module.", ""): v for k, v in raw.items()}

# ============================================================
#  Sub-network builders  (load from their OWN checkpoints)
# ============================================================
def build_restormer(cfg):
    mod = _load_mod("restormer_arch", cfg["rest_arch"])
    net = mod.Restormer(
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=[4,6,6,8], num_refinement_blocks=4,
        heads=[1,2,4,8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type="BiasFree", dual_pixel_task=False,
    )
    raw = torch.load(cfg["rest_ckpt"], map_location="cpu")
    sd  = raw.get("model", raw.get("params", raw.get("state_dict", raw)))
    sd  = {k.replace("module.", ""): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)
    print(f"  [Restormer]  ✓  {sum(p.numel() for p in net.parameters())/1e6:.1f}M params")
    return net

def build_nafnet_w64(cfg):
    for fpath in cfg["naf_deps"]:
        if os.path.exists(fpath):
            _load_mod("basicsr.models.archs." + os.path.basename(fpath)[:-3], fpath)
    mod = _load_mod("nafnet_w64_arch", cfg["naf_arch"])
    net = mod.NAFNet(img_channel=3, width=64, middle_blk_num=12,
                     enc_blk_nums=[2,2,4,8], dec_blk_nums=[2,2,2,2])
    raw = torch.load(cfg["naf_ckpt"], map_location="cpu")
    sd  = raw.get("model", raw.get("params", raw.get("params_ema", raw.get("state_dict", raw))))
    sd  = {k.replace("module.", ""): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)
    print(f"  [NAFNet-w64] ✓  {sum(p.numel() for p in net.parameters())/1e6:.1f}M params")
    return net

def build_nafnet_w32(cfg):
    mod = _load_mod("nafnet_w32_arch", cfg["naf_arch"])
    net = mod.NAFNet(img_channel=3, width=32, middle_blk_num=12,
                     enc_blk_nums=[2,2,4,8], dec_blk_nums=[2,2,2,2])
    raw = torch.load(cfg["model_c_ckpt"], map_location="cpu")
    sd  = raw.get("model", raw.get("params", raw.get("params_ema", raw.get("state_dict", raw))))
    sd  = {k.replace("module.", ""): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)
    print(f"  [NAFNet-w32] ✓  {sum(p.numel() for p in net.parameters())/1e6:.1f}M params")
    return net

# ============================================================
#  TripleEnsemble  (identical to training code)
# ============================================================
class TripleEnsemble(nn.Module):
    def __init__(self, restormer, nafnet, model_c, weights_init=(0.34, 0.33, 0.33)):
        super().__init__()
        self.restormer = restormer
        self.nafnet    = nafnet
        self.model_c   = model_c
        logits = torch.tensor([math.log(w + 1e-8) for w in weights_init])
        self.logits = nn.Parameter(logits)   # trained mixing weights
        self.alpha  = nn.Parameter(logits.clone())  # old key alias — kept for compat

    def forward(self, x):
        w = torch.softmax(self.logits, dim=0)
        return w[0]*self.restormer(x) + w[1]*self.nafnet(x) + w[2]*self.model_c(x)

    @property
    def weights(self):
        return torch.softmax(self.logits, dim=0).tolist()

# ============================================================
#  Load ensemble checkpoint  (handles frozen / partial saves)
# ============================================================
def load_ensemble(cfg, device):
    print("\nBuilding sub-networks from pre-trained checkpoints...")
    restormer = build_restormer(cfg)
    nafnet    = build_nafnet_w64(cfg)
    model_c   = build_nafnet_w32(cfg)

    model = TripleEnsemble(restormer, nafnet, model_c).to(device)

    print(f"\nLoading ensemble checkpoint: {cfg['ensemble_ckpt']}")
    raw = torch.load(cfg["ensemble_ckpt"], map_location="cpu")

    if isinstance(raw, dict):
        print(f"  Top-level keys : {list(raw.keys())}")
        saved_ep   = raw.get("epoch",     "?")
        saved_psnr = raw.get("best_psnr", "?")
        print(f"  Saved epoch    : {saved_ep}  |  best_psnr : {saved_psnr}")

        model_sd = raw.get("model", {})
        ema_sd   = raw.get("ema_shadow", {})

        # ── apply fine-tuned backbone weights if present ──
        has_rest = any(k.startswith("restormer.") for k in model_sd)
        has_naf  = any(k.startswith("nafnet.")    for k in model_sd)
        has_mc   = any(k.startswith("model_c.")   for k in model_sd)
        if has_rest or has_naf or has_mc:
            print(f"  Fine-tuned backbone weights found → loading (strict=False)")
            miss, unexp = model.load_state_dict(
                {k.replace("module.", ""): v for k, v in model_sd.items()}, strict=False)
            print(f"  Missing: {len(miss)}  Unexpected: {len(unexp)}")
        else:
            print("  No backbone weights in checkpoint — using pre-trained backbones as-is")

        # ── load mixing scalar (logits / alpha) ──
        for sd, label in [(model_sd, "model_sd"), (ema_sd, "ema_shadow")]:
            for key in ("logits", "alpha"):
                if key in sd:
                    val = sd[key].to(device)
                    model.logits.data.copy_(val)
                    model.alpha.data.copy_(val)
                    print(f"  Mixing weights from {label}['{key}'] → {val.tolist()}")
                    break

        # ── apply EMA weights if available ──
        if ema_sd:
            applied = 0
            for name, param in model.named_parameters():
                if name in ema_sd:
                    param.data.copy_(ema_sd[name].to(device))
                    applied += 1
            if applied:
                print(f"  EMA weights applied to {applied} parameters ✓")

    model.eval()
    w = model.weights
    print(f"\n  Ensemble weights → Restormer:{w[0]:.4f}  NAFNet-w64:{w[1]:.4f}  NAFNet-w32:{w[2]:.4f}")
    print(f"  Total params     : {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model

# ============================================================
#  Tiled inference
# ============================================================
@torch.no_grad()
def infer_tiled(model, noisy_t, tile, overlap, device):
    """noisy_t : (1,3,H,W) float32 CPU  →  same shape CPU"""
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

# ============================================================
#  8-fold TTA  (flip × 4 rotations = 8 predictions)
# ============================================================
def tta_forward(model, noisy_t, tile, overlap, device):
    preds = []
    for flip in [False, True]:
        t = noisy_t.flip(-1) if flip else noisy_t
        for k in range(4):
            aug  = torch.rot90(t, k, dims=[-2, -1])
            pred = infer_tiled(model, aug, tile, overlap, device)
            pred = torch.rot90(pred, -k, dims=[-2, -1])   # undo rotation
            if flip:
                pred = pred.flip(-1)                       # undo flip
            preds.append(pred)
    return torch.stack(preds).mean(0)

# ============================================================
#  MAIN
# ============================================================
def main():
    cfg    = CFG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  TTA    : {cfg['use_tta']}")
    print(f"  Tile   : {cfg['tile_size']}  Overlap: {cfg['tile_overlap']}")
    print(f"{'='*60}")

    model = load_ensemble(cfg, device)

    # collect images
    paths = []
    for e in ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"):
        paths += glob.glob(os.path.join(cfg["test_dir"], e))
    paths = sorted(paths)
    print(f"\n  Images found: {len(paths)}")
    assert len(paths) > 0, f"No images found in {cfg['test_dir']}"

    total_time = 0.0
    outer = tqdm(paths, desc="Inference", unit="img")

    for i, src in enumerate(outer):
        fname = os.path.basename(src)
        dst   = os.path.join(cfg["out_dir"], fname)
        outer.set_postfix(file=fname)

        img_pil        = Image.open(src).convert("RGB")
        orig_W, orig_H = img_pil.size
        noisy_np       = np.array(img_pil, dtype=np.float32) / 255.0
        noisy_t        = torch.from_numpy(noisy_np).permute(2,0,1).unsqueeze(0)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        out_t = (tta_forward if cfg["use_tta"] else infer_tiled)(
            model, noisy_t, cfg["tile_size"], cfg["tile_overlap"], device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        out_np  = (out_t.squeeze(0).permute(1,2,0).numpy() * 255.0).clip(0,255).astype(np.uint8)
        out_pil = Image.fromarray(out_np)

        assert out_pil.size == (orig_W, orig_H), \
            f"Size mismatch! input={orig_W,orig_H} output={out_pil.size}"

        out_pil.save(dst, format="PNG", compress_level=0)   # lossless, fastest

        avg = total_time / (i + 1)
        eta = avg * (len(paths) - i - 1)
        tqdm.write(
            f"  [{i+1:>3}/{len(paths)}] {fname}  {orig_W}×{orig_H}"
            f"  | {elapsed:.2f}s  | avg {avg:.2f}s  | ETA {eta/60:.1f} min"
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────
    avg_time = total_time / max(len(paths), 1)
    w = model.weights
    print(f"\n  Total : {total_time:.1f}s  |  Avg/image : {avg_time:.2f}s")

    # ── readme.txt  (NTIRE format, matches your LoRA submission) ──────
    readme_content = textwrap.dedent(f"""\
        runtime per image [s] : {avg_time:.2f}
        CPU[1] / GPU[0] : {1 - cfg['uses_gpu']}
        Extra Data [1] / No Extra Data [0] : {cfg['extra_data']}
        Other description : {cfg['other_desc']} Ensemble weights: Restormer={w[0]:.3f}, NAFNet-w64={w[1]:.3f}, NAFNet-w32={w[2]:.3f}.
    """)
    readme_path = os.path.join(cfg["out_dir"], "readme.txt")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"\n  readme.txt written.")
    print(readme_content)

    # ── flat ZIP  (PNGs + readme.txt, no subfolders) ──────
    pngs = sorted(glob.glob(os.path.join(cfg["out_dir"], "*.png")))
    with zipfile.ZipFile(cfg["zip_path"], "w", zipfile.ZIP_DEFLATED) as zf:
        for p in pngs:
            zf.write(p, arcname=os.path.basename(p))   # flat — no folder path
        zf.write(readme_path, arcname="readme.txt")

    size_mb = os.path.getsize(cfg["zip_path"]) / 1e6
    print(f"  ZIP  : {cfg['zip_path']}  ({size_mb:.1f} MB)")
    print(f"  Files: {len(pngs)} PNGs + readme.txt")
    print(f"\n  ✓ Done. Submission ready at: {cfg['zip_path']}\n")


if __name__ == "__main__":
    main()
