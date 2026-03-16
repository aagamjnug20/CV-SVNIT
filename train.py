import os, sys, math, copy, glob, time, importlib, importlib.util, subprocess
import numpy as np
import imageio.v3 as imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, "/kaggle/working/Restormer")
sys.path.insert(0, "/kaggle/working/NAFNet")

import argparse

def parse_args():

    parser = argparse.ArgumentParser("TripleEnsemble Training")

    # datasets
    parser.add_argument("--train_dirs", nargs="+", required=True)
    parser.add_argument("--val_dirs", nargs="+", required=True)

    # checkpoints
    parser.add_argument("--naf_ckpt", required=True)
    parser.add_argument("--rest_ckpt", required=True)
    parser.add_argument("--modelc_ckpt", required=True)

    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--resume", default=None)

    # hyperparams
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--sigma", type=float, default=50)

    parser.add_argument("--lr_pretrained", type=float, default=5e-5)
    parser.add_argument("--lr_new", type=float, default=1e-4)

    parser.add_argument("--warmup_steps", type=int, default=2000)

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--mode", choices=["train","test"], default="train")

    parser.add_argument("--test_dir", default=None)
    parser.add_argument("--result_dir", default="results")

    return parser.parse_args()
    

# ── Paths ─────────────────────────────────────────────────
args = parse_args()

TRAIN_DIRS = args.train_dirs
VAL_DIRS   = args.val_dirs

NAF_CKPT     = args.naf_ckpt
REST_CKPT    = args.rest_ckpt
MODEL_C_CKPT = args.modelc_ckpt

SAVE_DIR   = args.save_dir
RESULT_DIR = args.result_dir

PATCH_SIZE = args.patch_size
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
SIGMA      = args.sigma

LR_PRETRAINED = args.lr_pretrained
LR_NEW        = args.lr_new
WARMUP_STEPS  = args.warmup_steps

NUM_WORKERS = args.num_workers

RESUME = args.resume
MODE   = args.mode
TEST_DIR = args.test_dir

os.makedirs(SAVE_DIR,   exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

VAL_TILE      = 128
VAL_OVERLAP   = 8
VAL_EVERY     = 1
VAL_SUBSET    = 10

# ════════════════════════════════════════════════════════
# 1. LOSS
# ════════════════════════════════════════════════════════
def charbonnier_loss(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))

def combined_loss(pred, target):
    return 0.5 * F.mse_loss(pred, target) + 0.5 * charbonnier_loss(pred, target)

# ════════════════════════════════════════════════════════
# 2. NOISE
# ════════════════════════════════════════════════════════
def add_noise_np(image_np, sigma=50.0):
    img = image_np.astype(np.float64) / 255.0
    noise = np.random.normal(0, sigma / 255.0, img.shape)
    return (img + noise) * 255.0

def crop_to_multiple(image_np, s=8):
    h, w = image_np.shape[:2]
    return image_np[: h - h % s, : w - w % s]

def np_to_tensor(image_np):
    arr = np.clip(image_np, 0, 255).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)

def tensor_to_np(t):
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return np.round(arr * 255).astype(np.uint8)

# ════════════════════════════════════════════════════════
# 3. DATASET
# ════════════════════════════════════════════════════════
class DIV2KDataset(Dataset):
    def __init__(self, dirs, patch_size=256, mode="train", sigma=50.0):
        # Accept either a single path string or a list of paths
        if isinstance(dirs, str):
            dirs = [dirs]
        self.files = []
        for d in dirs:
            found = sorted(glob.glob(os.path.join(d, "*.png")))
            self.files.extend(found)
        self.files = sorted(self.files)
        self.patch_size = patch_size
        self.mode       = mode
        self.sigma      = sigma
        assert len(self.files) > 0, f"No PNG files found in: {dirs}"
        print(f"[Dataset:{mode}] {len(self.files)} images from {len(dirs)} dir(s)")
        if mode in ("val", "test"):
            print(f"[Dataset:{mode}] Preloading...", flush=True)
            self.cache = [imageio.imread(f) for f in self.files]
            print(f"[Dataset:{mode}] Preload done ✓", flush=True)
        else:
            self.cache = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_np = self.cache[idx] if self.cache is not None \
                 else imageio.imread(self.files[idx])
        fname  = os.path.basename(self.files[idx])
        if img_np.ndim == 2:
            img_np = np.stack([img_np]*3, axis=-1)

        if self.mode == "train":
            h, w = img_np.shape[:2]
            if h < self.patch_size:
                img_np = np.array(Image.fromarray(img_np).resize(
                    (max(w, self.patch_size), self.patch_size), Image.BICUBIC))
                h = self.patch_size
            if w < self.patch_size:
                img_np = np.array(Image.fromarray(img_np).resize(
                    (self.patch_size, max(h, self.patch_size)), Image.BICUBIC))
                w = self.patch_size
            h, w = img_np.shape[:2]
            top  = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
            img_np = img_np[top:top+self.patch_size, left:left+self.patch_size]

            if np.random.rand() < 0.5:
                img_np = np.fliplr(img_np).copy()
            if np.random.rand() < 0.5:
                img_np = np.flipud(img_np).copy()
            if np.random.rand() < 0.5:
                k = np.random.randint(1, 4)
                img_np = np.rot90(img_np, k).copy()
        else:
            img_np = crop_to_multiple(img_np, s=8)

        noisy_np = add_noise_np(img_np, self.sigma)
        clean = np_to_tensor(img_np.astype(np.float32))
        noisy = np_to_tensor(noisy_np)

        if self.mode == "test":
            return noisy, clean, fname
        return noisy, clean

# ════════════════════════════════════════════════════════
# 4. PSNR
# ════════════════════════════════════════════════════════
def psnr(pred, target):
    with torch.no_grad():
        mse = F.mse_loss(pred.clamp(0,1), target.clamp(0,1))
        return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse.item())

# ════════════════════════════════════════════════════════
# 5. MODEL LOADING
# ════════════════════════════════════════════════════════
def _load_module_from_file(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

def load_restormer(ckpt_path):
    arch_file = "/kaggle/working/Restormer/basicsr/models/archs/restormer_arch.py"
    mod = _load_module_from_file("restormer_arch", arch_file)
    net = mod.Restormer(
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=[4,6,6,8], num_refinement_blocks=4,
        heads=[1,2,4,8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type="BiasFree", dual_pixel_task=False,
    )
    ckpt    = torch.load(ckpt_path, map_location="cpu")
    weights = ckpt.get("model", ckpt.get("params", ckpt.get("state_dict", ckpt)))
    weights = {k.replace("module.", ""): v for k, v in weights.items()}
    net.load_state_dict(weights, strict=True)
    print(f"[Restormer]  ✓  {sum(p.numel() for p in net.parameters())/1e6:.1f}M params")
    return net

def load_nafnet(ckpt_path):
    # ← Uses unique mod_name "nafnet_w64_arch" to avoid sys.modules collision
    # with the width-32 loader below
    NAFNET_ARCH_DIR = "/kaggle/working/NAFNet/basicsr/models/archs"
    for fname in ["arch_util.py", "local_arch.py"]:
        fpath   = os.path.join(NAFNET_ARCH_DIR, fname)
        modname = "basicsr.models.archs." + fname[:-3]
        if os.path.exists(fpath):
            spec = importlib.util.spec_from_file_location(modname, fpath)
            mod  = importlib.util.module_from_spec(spec)
            sys.modules[modname] = mod
            spec.loader.exec_module(mod)
    arch_file = "/kaggle/working/NAFNet/basicsr/models/archs/NAFNet_arch.py"
    mod       = _load_module_from_file("nafnet_w64_arch", arch_file)
    net = mod.NAFNet(
        img_channel=3, width=64, middle_blk_num=12,
        enc_blk_nums=[2,2,4,8], dec_blk_nums=[2,2,2,2],
    )
    ckpt    = torch.load(ckpt_path, map_location="cpu")
    weights = ckpt.get("model", ckpt.get("params", ckpt.get("params_ema", ckpt.get("state_dict", ckpt))))
    weights = {k.replace("module.", ""): v for k, v in weights.items()}
    net.load_state_dict(weights, strict=True)
    print(f"[NAFNet-w64] ✓  {sum(p.numel() for p in net.parameters())/1e6:.1f}M params")
    return net

def load_model_c(ckpt_path):
    # NAFNet width-32 — uses a different mod_name to avoid overwriting w64 in sys.modules
    arch_file = "/kaggle/working/NAFNet/basicsr/models/archs/NAFNet_arch.py"
    mod       = _load_module_from_file("nafnet_w32_arch", arch_file)
    net = mod.NAFNet(
        img_channel=3, width=32, middle_blk_num=12,
        enc_blk_nums=[2,2,4,8], dec_blk_nums=[2,2,2,2],
    )
    ckpt    = torch.load(ckpt_path, map_location="cpu")
    weights = ckpt.get("model", ckpt.get("params", ckpt.get("params_ema", ckpt.get("state_dict", ckpt))))
    weights = {k.replace("module.", ""): v for k, v in weights.items()}
    net.load_state_dict(weights, strict=True)
    print(f"[NAFNet-w32] ✓  {sum(p.numel() for p in net.parameters())/1e6:.1f}M params")
    return net

# ════════════════════════════════════════════════════════
# 6. ENSEMBLE
# ════════════════════════════════════════════════════════
class TripleEnsemble(nn.Module):
    def __init__(self, restormer, nafnet, model_c,
                 weights_init=(0.34, 0.33, 0.33)):
        super().__init__()
        self.restormer = restormer
        self.nafnet    = nafnet
        self.model_c   = model_c
        # 3 free logits; softmax keeps them summing to 1
        logits = torch.tensor([math.log(w + 1e-8) for w in weights_init])
        self.logits = nn.Parameter(logits)

    def forward(self, x):
        w     = torch.softmax(self.logits, dim=0)   # [3], sums to 1
        out_r = self.restormer(x)
        out_n = self.nafnet(x)
        out_c = self.model_c(x)
        return w[0] * out_r + w[1] * out_n + w[2] * out_c

    @property
    def weights(self):
        return torch.softmax(self.logits, dim=0).tolist()

# ════════════════════════════════════════════════════════
# 6b. TILED FORWARD  (was missing — needed by validate + inference)
# ════════════════════════════════════════════════════════
@torch.no_grad()
def forward_tiled(model, x, tile=128, overlap=8):
    b, c, h, w = x.shape
    assert b == 1
    stride = tile - overlap
    out   = torch.zeros_like(x, dtype=torch.float32)
    count = torch.zeros_like(x, dtype=torch.float32)
    for top in range(0, h, stride):
        for left in range(0, w, stride):
            # clamp so the tile never goes out of bounds
            t = min(top,  h - tile)
            l = min(left, w - tile)
            patch = x[:, :, t:t+tile, l:l+tile]
            with torch.amp.autocast('cuda'):
                pred_patch = model(patch).float()
            out  [:, :, t:t+tile, l:l+tile] += pred_patch
            count[:, :, t:t+tile, l:l+tile] += 1.0
    return out / count.clamp(min=1.0)

# ════════════════════════════════════════════════════════
# 7. PARAM GROUPS
# ════════════════════════════════════════════════════════
def get_param_groups(model):
    backbone_ids = {
        id(p)
        for p in (list(model.restormer.parameters())
                + list(model.nafnet.parameters())
                + list(model.model_c.parameters()))
    }
    no_wd_ids = set()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm,
                          nn.GroupNorm, nn.InstanceNorm2d)):
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
        {"params": bb_wd,    "lr": LR_PRETRAINED, "weight_decay": WEIGHT_DECAY},
        {"params": bb_nowd,  "lr": LR_PRETRAINED, "weight_decay": 0.0},
        {"params": new_wd,   "lr": LR_NEW,        "weight_decay": WEIGHT_DECAY},
        {"params": new_nowd, "lr": LR_NEW,        "weight_decay": 0.0},
    ]
    print(f"[ParamGroups] {sum(len(g['params']) for g in groups)} tensors")
    return groups

# ════════════════════════════════════════════════════════
# 8. LOOKAHEAD
# ════════════════════════════════════════════════════════
class Lookahead(torch.optim.Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=5):
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
            for group, slow_group in zip(self.base_optimizer.param_groups,
                                         self.slow_weights):
                for p_fast, p_slow in zip(group["params"], slow_group):
                    if p_fast.grad is None:
                        continue
                    p_slow.data.add_(self.alpha * (p_fast.data - p_slow.data))
                    p_fast.data.copy_(p_slow.data)
        return loss

    def zero_grad(self, set_to_none=True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, sd):
        self.base_optimizer.load_state_dict(sd)
        self.param_groups = self.base_optimizer.param_groups

    def __getattr__(self, name):
        return getattr(self.base_optimizer, name)

# ════════════════════════════════════════════════════════
# 9. SCHEDULER
# ════════════════════════════════════════════════════════
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps,
                 min_lr=1e-7, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr       = min_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        step = self.last_epoch
        out  = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                lr = base_lr * (step + 1) / max(1, self.warmup_steps)
            else:
                prog   = (step - self.warmup_steps) / max(
                            1, self.total_steps - self.warmup_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * prog))
                lr     = self.min_lr + (base_lr - self.min_lr) * cosine
            out.append(lr)
        return out

# ════════════════════════════════════════════════════════
# 10. EMA
# ════════════════════════════════════════════════════════
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model  = model
        self.decay  = decay
        self.shadow = {name: param.data.clone()
                       for name, param in model.named_parameters()
                       if param.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (self.decay * self.shadow[name]
                                     + (1.0 - self.decay) * param.data)

    def apply(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

# ════════════════════════════════════════════════════════
# 11. FREEZE HELPERS
# ════════════════════════════════════════════════════════
def freeze_backbones(model):
    for p in (list(model.restormer.parameters())
            + list(model.nafnet.parameters())
            + list(model.model_c.parameters())):
        p.requires_grad_(False)
    print("[Freeze] Backbones frozen")

def unfreeze_backbones(model):
    for p in (list(model.restormer.parameters())
            + list(model.nafnet.parameters())
            + list(model.model_c.parameters())):
        p.requires_grad_(True)
    print("[Unfreeze] Backbones unfrozen")

# ════════════════════════════════════════════════════════
# 12. CHECKPOINT
# ════════════════════════════════════════════════════════
def save_ckpt(path, model, optimizer, ema, epoch, best_psnr, scheduler):
    import shutil
    data = {
        "epoch":      epoch,
        "best_psnr":  best_psnr,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
        "ema_shadow": ema.shadow,
    }
    torch.save(data, path)
    shutil.copy(path, "/kaggle/working/best.pth")
    print(f"  [ckpt] saved epoch={epoch}  best_psnr={best_psnr:.4f}", flush=True)

def load_ckpt(path, model, optimizer, ema, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    ema.shadow = {k: v.to(next(model.parameters()).device)
                  for k, v in ckpt["ema_shadow"].items()}
    print(f"  [ckpt] Resumed epoch={ckpt['epoch']}  best_psnr={ckpt['best_psnr']:.4f}")
    return ckpt["epoch"], ckpt["best_psnr"]

# ════════════════════════════════════════════════════════
# 13. TRAIN ONE EPOCH
# ════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, scheduler, scaler, ema, device):
    model.train()
    total_loss, total_psnr, n = 0.0, 0.0, 0

    for step, (noisy, clean) in enumerate(loader):
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            pred = model(noisy)
            loss = combined_loss(pred, clean)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update()

        step_psnr = psnr(pred.detach(), clean)
        total_loss += loss.item()
        total_psnr += step_psnr
        n += 1

        if step % 50 == 0:
            print(f"  step {step}/{len(loader)}  loss={loss.item():.6f}  psnr={step_psnr:.4f} dB", flush=True)

    torch.cuda.empty_cache()
    return {"loss": total_loss / n, "psnr": total_psnr / n}

# ════════════════════════════════════════════════════════
# 14. VALIDATE
# ════════════════════════════════════════════════════════
@torch.no_grad()
def validate(model, loader, ema, device):
    ema.apply()
    model.eval()
    total_psnr = 0.0
    count = 0
    for i, (noisy, clean) in enumerate(loader):
        if i >= VAL_SUBSET:
            break
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        pred = forward_tiled(model, noisy, tile=VAL_TILE, overlap=VAL_OVERLAP)
        total_psnr += psnr(pred, clean)
        count += 1
    torch.cuda.empty_cache()
    ema.restore()
    return total_psnr / count

# ════════════════════════════════════════════════════════
# 15. INFERENCE
# ════════════════════════════════════════════════════════
@torch.no_grad()
def run_inference_only(model, ckpt_path, device):
    print(f"\n{'='*60}\nINFERENCE — {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    # Apply EMA weights for inference
    for name, param in model.named_parameters():
        if param.requires_grad and name in ckpt["ema_shadow"]:
            param.data.copy_(ckpt["ema_shadow"][name].to(device))
    model.eval().to(device)

    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.png")))
    print(f"Found {len(files)} images")

    for fpath in files:
        fname    = os.path.basename(fpath)
        img_np   = imageio.imread(fpath)
        if img_np.ndim == 2:
            img_np = np.stack([img_np]*3, axis=-1)
        img_np   = crop_to_multiple(img_np, s=8)
        noisy    = np_to_tensor(img_np.astype(np.float32)).unsqueeze(0).to(device)

        pred     = forward_tiled(model, noisy, tile=VAL_TILE, overlap=VAL_OVERLAP)
        pred_np  = tensor_to_np(pred[0].cpu())

        save_path = os.path.join(RESULT_DIR, f"denoised_{fname}")
        imageio.imwrite(save_path, pred_np)
        print(f"  {fname} → {save_path}", flush=True)
        torch.cuda.empty_cache()

    print(f"\nDone. Results in {RESULT_DIR}")

# ════════════════════════════════════════════════════════
# 16. MAIN
# ════════════════════════════════════════════════════════
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    restormer = load_restormer(REST_CKPT)
    nafnet    = load_nafnet(NAF_CKPT)
    model_c   = load_model_c(MODEL_C_CKPT)

    model = TripleEnsemble(
        restormer, nafnet, model_c,
        weights_init=(0.34, 0.33, 0.33),
    ).to(device)
    print(f"[Ensemble] weights = {[f'{w:.4f}' for w in model.weights]}")

    best_path = os.path.join(SAVE_DIR, "best.pth")

    # ── Test / inference mode ─────────────────────────────────────────
    if MODE == "test":
        assert os.path.isfile(best_path), f"No checkpoint at {best_path}"
        run_inference_only(model, best_path, device)   # ← was run_test (fixed)
        return

    # ── Training mode ─────────────────────────────────────────────────
    train_ds = DIV2KDataset(TRAIN_DIRS, PATCH_SIZE, mode="train", sigma=SIGMA)
    val_ds   = DIV2KDataset(VAL_DIRS,   PATCH_SIZE, mode="val",   sigma=SIGMA)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    total_steps = NUM_EPOCHS * len(train_loader)
    print(f"Train batches/epoch: {len(train_loader)}  |  Total steps: {total_steps}")

    param_groups = get_param_groups(model)
    base_opt     = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    optimizer    = Lookahead(base_opt, alpha=0.5, k=5)
    scheduler    = WarmupCosineScheduler(
        optimizer.base_optimizer,
        warmup_steps=WARMUP_STEPS,
        total_steps=total_steps,
        min_lr=LR_MIN,
    )
    ema    = EMA(model, decay=EMA_DECAY)
    scaler = torch.amp.GradScaler('cuda')

    start_epoch, best_psnr = 0, 0.0
    if RESUME and os.path.isfile(RESUME):
        start_epoch, best_psnr = load_ckpt(RESUME, model, optimizer, ema, scheduler)

    print(f"\n{'='*60}")
    print(f"Training  epochs={NUM_EPOCHS}  sigma={SIGMA}  patch={PATCH_SIZE}  bs={BATCH_SIZE}")
    print(f"Loss: Charbonnier + MSE  |  LR_pretrained={LR_PRETRAINED}  LR_new={LR_NEW}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()

        # Freeze backbones for the first 5 epochs so only the logits train.
        # Guard with start_epoch <= epoch so this doesn't re-trigger on resume.
        if epoch == 0 and start_epoch == 0:
            freeze_backbones(model)

        # ← Fixed: only rebuild optimizer/scheduler when we actually cross
        # epoch 5 for the first time (not on every resume past epoch 5)
        if epoch == 5 and start_epoch <= 5:
            unfreeze_backbones(model)
            param_groups = get_param_groups(model)
            base_opt     = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
            optimizer    = Lookahead(base_opt, alpha=0.5, k=5)
            scheduler    = WarmupCosineScheduler(
                optimizer.base_optimizer,
                warmup_steps=0,
                total_steps=total_steps - epoch * len(train_loader),
                min_lr=LR_MIN,
                last_epoch=-1,
            )

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, ema, device)

        if (epoch + 1) % VAL_EVERY == 0 or epoch == 0:
            val_psnr = validate(model, val_loader, ema, device)
        else:
            val_psnr = best_psnr

        elapsed = time.time() - t0
        w = model.weights
        print(
            f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}]  "
            f"Loss={train_metrics['loss']:.6f}  "
            f"TrainPSNR={train_metrics['psnr']:.4f}  "
            f"ValPSNR={val_psnr:.4f}  "
            f"w=[{w[0]:.3f},{w[1]:.3f},{w[2]:.3f}]  "
            f"[{elapsed:.0f}s]",
            flush=True,
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_ckpt(best_path, model, optimizer, ema, epoch+1, best_psnr, scheduler)

    print(f"\nDone. Best Val PSNR: {best_psnr:.4f} dB")
    # Final inference run after training completes
    run_inference_only(model, best_path, device)   # ← was run_test (fixed)


if __name__ == "__main__":
    main()
