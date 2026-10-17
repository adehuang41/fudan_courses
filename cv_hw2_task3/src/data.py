"""Oxford-IIIT Pet trimap segmentation dataset & loaders.

trimap pixel values returned by torchvision.datasets.OxfordIIITPet are 1/2/3:
    1 = pet (foreground)
    2 = background
    3 = boundary
We remap to 0/1/2 to use as plain class indices for CrossEntropy / Dice.

Image and mask are augmented synchronously via custom joint transforms — no
pretrained backbone, so we normalize to [0,1] (no ImageNet mean/std).
"""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as T
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as TF

NUM_CLASSES = 3  # foreground / background / boundary
CLASS_NAMES = ["foreground", "background", "boundary"]


def _remap_trimap(mask: Image.Image) -> torch.Tensor:
    """trimap PIL (values 1/2/3) -> LongTensor(H,W) with values 0/1/2."""
    arr = np.array(mask, dtype=np.int64) - 1  # 1->0, 2->1, 3->2
    arr = np.clip(arr, 0, 2)
    return torch.from_numpy(arr)


class JointTransform:
    """Apply identical geometric transforms to (image, mask)."""

    def __init__(self, img_size: int = 256, train: bool = True):
        self.img_size = img_size
        self.train = train

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.train:
            # Random resized crop, identical params for both
            i, j, h, w = T.RandomResizedCrop.get_params(
                img, scale=(0.7, 1.0), ratio=(0.85, 1.18)
            )
            img = TF.resized_crop(img, i, j, h, w, [self.img_size, self.img_size],
                                  interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resized_crop(mask, i, j, h, w, [self.img_size, self.img_size],
                                   interpolation=TF.InterpolationMode.NEAREST)
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            # Color jitter on image only
            img = T.ColorJitter(0.2, 0.2, 0.2)(img)
        else:
            img = TF.resize(img, [self.img_size, self.img_size],
                            interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [self.img_size, self.img_size],
                             interpolation=TF.InterpolationMode.NEAREST)

        img_t = TF.to_tensor(img)  # [0,1], CHW
        mask_t = _remap_trimap(mask)
        return img_t, mask_t


class SegWrapper(Dataset):
    """Wrap an OxfordIIITPet(target_types='segmentation') with a JointTransform."""

    def __init__(self, base: OxfordIIITPet, joint: JointTransform):
        self.base = base
        self.joint = joint

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask = self.base[idx]
        return self.joint(img, mask)


def _raw_dataset(root: str, split: str) -> OxfordIIITPet:
    return OxfordIIITPet(
        root=root,
        split=split,
        target_types="segmentation",
        download=True,
    )


def build_loaders(
    root: str = "./data",
    batch_size: int = 16,
    num_workers: int = 4,
    img_size: int = 256,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    train_joint = JointTransform(img_size=img_size, train=True)
    eval_joint = JointTransform(img_size=img_size, train=False)

    full = _raw_dataset(root, "trainval")
    test_raw = _raw_dataset(root, "test")

    n = len(full)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    g = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(range(n), [n_train, n_val], generator=g)
    train_idx = list(train_idx)
    val_idx = list(val_idx)

    train_set = SegWrapper(_Subset(full, train_idx), train_joint)
    val_set = SegWrapper(_Subset(full, val_idx), eval_joint)
    test_set = SegWrapper(test_raw, eval_joint)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             persistent_workers=num_workers > 0)
    return train_loader, val_loader, test_loader


class _Subset(Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]
