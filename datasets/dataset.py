"""datasets/dataset.py — DIV2K / LSDIR dataset with on-the-fly noise synthesis."""

import os
import glob

import numpy as np
import imageio.v3 as imageio
import torch
from torch.utils.data import Dataset
from PIL import Image


# ── Helpers ───────────────────────────────────────────────────────────────────

def add_noise_np(image_np: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    img   = image_np.astype(np.float64) / 255.0
    noise = np.random.normal(0, sigma / 255.0, img.shape)
    return (img + noise) * 255.0


def crop_to_multiple(image_np: np.ndarray, s: int = 8) -> np.ndarray:
    h, w = image_np.shape[:2]
    return image_np[: h - h % s, : w - w % s]


def np_to_tensor(image_np: np.ndarray) -> torch.Tensor:
    arr = np.clip(image_np, 0, 255).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return np.round(arr * 255).astype(np.uint8)


# ── Dataset ───────────────────────────────────────────────────────────────────

class DIV2KDataset(Dataset):
    """
    Loads clean PNG images from one or more directories and synthesises
    AWGN noise on-the-fly.

    Parameters
    ----------
    dirs        : str or list of str — directories containing PNG files
    patch_size  : random-crop size used during training
    mode        : "train" | "val" | "test"
    sigma       : noise standard deviation (0–255 scale)
    """

    def __init__(
        self,
        dirs,
        patch_size: int = 128,
        mode: str = "train",
        sigma: float = 50.0,
    ):
        if isinstance(dirs, str):
            dirs = [dirs]

        self.files: list[str] = []
        for d in dirs:
            self.files.extend(sorted(glob.glob(os.path.join(d, "*.png"))))
        self.files = sorted(self.files)

        self.patch_size = patch_size
        self.mode       = mode
        self.sigma      = sigma

        if not self.files:
            raise FileNotFoundError(f"No PNG files found in: {dirs}")
        print(f"[Dataset:{mode}] {len(self.files)} images from {len(dirs)} dir(s)")

        # Pre-load val/test images to avoid repeated disk I/O
        if mode in ("val", "test"):
            print(f"[Dataset:{mode}] Preloading…", flush=True)
            self.cache = [imageio.imread(f) for f in self.files]
            print(f"[Dataset:{mode}] Preload done ✓", flush=True)
        else:
            self.cache = None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        img_np = (
            self.cache[idx]
            if self.cache is not None
            else imageio.imread(self.files[idx])
        )
        fname = os.path.basename(self.files[idx])

        # Ensure 3-channel
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)

        if self.mode == "train":
            img_np = self._random_crop(img_np)
            img_np = self._augment(img_np)
        else:
            img_np = crop_to_multiple(img_np, s=8)

        noisy_np = add_noise_np(img_np, self.sigma)
        clean    = np_to_tensor(img_np.astype(np.float32))
        noisy    = np_to_tensor(noisy_np)

        if self.mode == "test":
            return noisy, clean, fname
        return noisy, clean

    # ── Private helpers ───────────────────────────────────────────────────────

    def _random_crop(self, img_np: np.ndarray) -> np.ndarray:
        ps = self.patch_size
        h, w = img_np.shape[:2]
        if h < ps:
            img_np = np.array(
                Image.fromarray(img_np).resize((max(w, ps), ps), Image.BICUBIC)
            )
            h = ps
        if w < ps:
            img_np = np.array(
                Image.fromarray(img_np).resize((ps, max(h, ps)), Image.BICUBIC)
            )
            w = ps
        h, w   = img_np.shape[:2]
        top    = np.random.randint(0, h - ps + 1)
        left   = np.random.randint(0, w - ps + 1)
        return img_np[top : top + ps, left : left + ps]

    @staticmethod
    def _augment(img_np: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            img_np = np.fliplr(img_np).copy()
        if np.random.rand() < 0.5:
            img_np = np.flipud(img_np).copy()
        if np.random.rand() < 0.5:
            img_np = np.rot90(img_np, np.random.randint(1, 4)).copy()
        return img_np
