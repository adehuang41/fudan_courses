from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 37


def build_transforms(img_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_loaders(root: str, batch_size: int = 64, num_workers: int = 4,
                  img_size: int = 224, val_ratio: float = 0.1, seed: int = 42):
    train_tf, eval_tf = build_transforms(img_size)

    full_train = OxfordIIITPet(root=root, split="trainval", target_types="category",
                               transform=train_tf, download=True)
    val_view = OxfordIIITPet(root=root, split="trainval", target_types="category",
                             transform=eval_tf, download=False)
    test_set = OxfordIIITPet(root=root, split="test", target_types="category",
                             transform=eval_tf, download=True)

    n = len(full_train)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    import torch
    g = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_train, [n_train, n_val], generator=g)
    val_subset.dataset = val_view  # use eval transforms

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             persistent_workers=num_workers > 0)
    return train_loader, val_loader, test_loader
