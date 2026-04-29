"""
datasets.py — Dataset loading for image and 2D flow matching experiments.

Image datasets : 'cifar10'
2D datasets    : '8gaussians', '40gaussians', 'moons', 'circles', 'checkerboard'
"""
import math
import torch
import numpy as np
from torchvision import datasets, transforms

IMAGE_DATASETS = {'cifar10'}
DATASETS_2D    = {'8gaussians', '40gaussians', 'moons', 'circles', 'checkerboard'}

X0_SHAPE = {
    'cifar10':      (3, 32, 32),
    '8gaussians':   (2,),
    '40gaussians':  (2,),
    'moons':        (2,),
    'circles':      (2,),
    'checkerboard': (2,),
}


def is_image(name):
    return name in IMAGE_DATASETS


def get_x0_shape(name):
    if name not in X0_SHAPE:
        raise ValueError(f'Unknown dataset "{name}". '
                         f'Choose: {sorted(IMAGE_DATASETS | DATASETS_2D)}')
    return X0_SHAPE[name]


# ── CIFAR-10 ───────────────────────────────────────────────────────────────────

def get_cifar10_loaders(data_dir, batch_size, num_workers,
                        distributed=False, rank=0, world_size=1):
    """
    Returns (train_loader, eval_loader, train_sampler).
    eval_loader uses no augmentation and is used for FID reference stats.
    """
    from torch.utils.data.distributed import DistributedSampler

    tfm_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    tfm_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    ds_train = datasets.CIFAR10(root=data_dir, train=True,
                                download=True,  transform=tfm_train)
    ds_eval  = datasets.CIFAR10(root=data_dir, train=True,
                                download=False, transform=tfm_eval)

    train_sampler = (DistributedSampler(ds_train, num_replicas=world_size, rank=rank)
                     if distributed else None)

    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=batch_size,
        sampler=train_sampler, shuffle=(not distributed),
        num_workers=num_workers, drop_last=True, pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(
        ds_eval, batch_size=256, shuffle=False, num_workers=num_workers)

    return train_loader, eval_loader, train_sampler


def collect_images(loader, max_n=2000):
    """Collect up to max_n CIFAR-10 images (CPU) for speed estimation."""
    imgs = []
    for x, _ in loader:
        imgs.append(x)
        if sum(i.shape[0] for i in imgs) >= max_n:
            break
    return torch.cat(imgs, 0)[:max_n]


# ── 2D toy datasets ────────────────────────────────────────────────────────────

def sample_2d(name, n):
    """Sample n points from a 2D distribution. Returns (n, 2) float32 tensor."""
    fns = {
        '8gaussians':   _sample_8gaussians,
        '40gaussians':  _sample_40gaussians,
        'moons':        _sample_moons,
        'circles':      _sample_circles,
        'checkerboard': _sample_checkerboard,
    }
    if name not in fns:
        raise ValueError(f'Unknown 2D dataset "{name}". Choose: {sorted(fns)}')
    return fns[name](n)


def _sample_8gaussians(n):
    s   = 4.0
    sq2 = 1.0 / math.sqrt(2)
    centers = torch.tensor([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [sq2, sq2], [sq2, -sq2], [-sq2, sq2], [-sq2, -sq2],
    ], dtype=torch.float32) * s
    idx = torch.randint(0, 8, (n,))
    return centers[idx] + 0.5 * torch.randn(n, 2)


def _sample_40gaussians(n):
    angles  = torch.linspace(0, 2 * math.pi, 41)[:-1]
    centers = torch.stack([4 * torch.cos(angles), 4 * torch.sin(angles)], dim=1)
    idx     = torch.randint(0, 40, (n,))
    return centers[idx] + 0.3 * torch.randn(n, 2)


def _sample_moons(n):
    try:
        from sklearn.datasets import make_moons
        x, _ = make_moons(n_samples=n, noise=0.1)
        return torch.tensor(x, dtype=torch.float32) * 2 - 1
    except ImportError:
        raise ImportError('sklearn is required for the moons dataset: pip install scikit-learn')


def _sample_circles(n):
    try:
        from sklearn.datasets import make_circles
        x, _ = make_circles(n_samples=n, noise=0.05, factor=0.5)
        return torch.tensor(x, dtype=torch.float32) * 2
    except ImportError:
        raise ImportError('sklearn is required for the circles dataset')


def _sample_checkerboard(n):
    out = []
    while sum(x.shape[0] for x in out) < n:
        x  = torch.rand(n, 2) * 4 - 2
        ok = ((x[:, 0].long() + x[:, 1].long()) % 2 == 0)
        out.append(x[ok])
    return torch.cat(out, 0)[:n]
