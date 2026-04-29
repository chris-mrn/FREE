"""
VAE training for CIFAR-10 — best-practice KL-regularized autoencoder.

Architecture (LDM-style, but trained from scratch):
  Encoder:  32×32×3  →  8×8×4  (4× spatial downsampling, 4 latent channels)
  Decoder:  8×8×4    →  32×32×3
  Attention at 8×8 bottleneck.
  GroupNorm + SiLU throughout.

Loss composition (all four components are well-established best practices):
  1. L1 reconstruction              (pixel-level fidelity)
  2. β-KL                           (latent regularisation; β=1e-6 very small)
  3. VGG-16 perceptual loss          (high-frequency / texture quality)
  4. PatchGAN adversarial loss       (sharpness, avoids blurring)
     • starts at step `disc_start` (default 10 K)
     • adaptive weight: ||∇_last_dec L_rec|| / ||∇_last_dec L_adv|| (VQGAN style)

Best practices implemented:
  ✓ 4× spatial downsampling  (8×8 latents, 256-dim) — better than SD's 8× for 32×32
  ✓ Per-channel latent normalisation stats saved at end (mean, std over train set)
  ✓ EMA on encoder+decoder
  ✓ KL warmup (0 → β over 5 K steps, prevents posterior collapse early on)
  ✓ Discriminator hinge loss + adaptive adversarial weight
  ✓ Random horizontal flip data augmentation
  ✓ Gradient clipping for both VAE and discriminator optimisers
  ✓ GroupNorm (stable across small batch sizes)
  ✓ Residual blocks + self-attention at bottleneck

Outputs:
  out_dir/vae_best.pt         — best reconstruction-loss checkpoint (encoder+decoder)
  out_dir/vae_ema_best.pt     — best EMA checkpoint
  out_dir/vae_final.pt        — final checkpoint
  out_dir/vae_ema_final.pt    — final EMA checkpoint
  out_dir/latent_stats.pt     — per-channel {mean, std} over training latents
  out_dir/recon_*.png         — reconstruction grids (train + fixed val batch)
  out_dir/loss.csv

Usage:
  python train_vae_cifar10.py --out_dir outputs/cifar10_vae [options]
"""

import os
import sys
import copy
import csv
import time
import argparse
import math

sys.path.insert(0, '/nfs/ghome/live/cmarouani/FREE')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def _num_groups(channels):
    """Largest divisor of `channels` that is ≤ 32 and ≥ 1."""
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    """Pre-activation residual block: GroupNorm → SiLU → Conv → GN → SiLU → Conv + skip."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1, stride=stride) if (in_ch != out_ch or stride != 1) else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Single-head self-attention with GroupNorm (spatial dims kept flat)."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.qkv  = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h    = self.norm(x)
        qkv  = self.qkv(h).reshape(B, 3, C, H * W).permute(1, 0, 2, 3)  # (3,B,C,HW)
        q, k, v = qkv[0], qkv[1], qkv[2]                                  # (B,C,HW)
        scale = C ** -0.5
        attn  = torch.softmax(torch.bmm(q.transpose(1, 2), k) * scale, dim=-1)  # (B,HW,HW)
        h     = torch.bmm(v, attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(h)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


# ─────────────────────────────────────────────────────────────────────────────
# Encoder and Decoder
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """32×32×3 → 8×8×(2*z_ch)   [mean‖logvar over z_ch channels]."""

    def __init__(self, in_ch=3, base_ch=128, ch_mult=(1, 2, 2), z_ch=4):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Downsampling blocks
        self.down = nn.ModuleList()
        cur_ch = base_ch
        for i, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            stride = 2 if i < len(ch_mult) - 1 else 1   # last level: no stride
            self.down.append(nn.Sequential(
                ResBlock(cur_ch, out_ch, stride=stride),
                ResBlock(out_ch, out_ch),
            ))
            cur_ch = out_ch

        # Bottleneck + attention
        self.mid = nn.Sequential(
            ResBlock(cur_ch, cur_ch),
            AttnBlock(cur_ch),
            ResBlock(cur_ch, cur_ch),
        )

        self.norm_out = nn.GroupNorm(_num_groups(cur_ch), cur_ch)
        self.conv_out = nn.Conv2d(cur_ch, 2 * z_ch, 3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for blk in self.down:
            h = blk(h)
        h = self.mid(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h   # (B, 2*z_ch, H/4, W/4)


class Decoder(nn.Module):
    """8×8×z_ch → 32×32×3."""

    def __init__(self, out_ch=3, base_ch=128, ch_mult=(1, 2, 2), z_ch=4):
        super().__init__()
        ch_mult_rev  = list(reversed(ch_mult))
        top_ch       = base_ch * ch_mult_rev[0]

        self.conv_in = nn.Conv2d(z_ch, top_ch, 3, padding=1)

        # Bottleneck + attention
        self.mid = nn.Sequential(
            ResBlock(top_ch, top_ch),
            AttnBlock(top_ch),
            ResBlock(top_ch, top_ch),
        )

        # Upsampling blocks
        self.up = nn.ModuleList()
        cur_ch = top_ch
        for i, mult in enumerate(ch_mult_rev):
            out_c = base_ch * mult
            upsample = i < len(ch_mult_rev) - 1   # no upsample at last level
            layers = [ResBlock(cur_ch, out_c), ResBlock(out_c, out_c)]
            if upsample:
                layers.append(Upsample(out_c))
            self.up.append(nn.Sequential(*layers))
            cur_ch = out_c

        self.norm_out = nn.GroupNorm(_num_groups(cur_ch), cur_ch)
        self.conv_out = nn.Conv2d(cur_ch, out_ch, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid(h)
        for blk in self.up:
            h = blk(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return torch.tanh(h)


# ─────────────────────────────────────────────────────────────────────────────
# Full KL-regularised autoencoder
# ─────────────────────────────────────────────────────────────────────────────

class KLVAE(nn.Module):
    def __init__(self, z_ch=4, base_ch=128, ch_mult=(1, 2, 2)):
        super().__init__()
        self.encoder = Encoder(base_ch=base_ch, ch_mult=ch_mult, z_ch=z_ch)
        self.decoder = Decoder(base_ch=base_ch, ch_mult=ch_mult, z_ch=z_ch)
        self.z_ch = z_ch

    def encode(self, x):
        h       = self.encoder(x)
        mean, logvar = h.chunk(2, dim=1)
        logvar  = logvar.clamp(-30.0, 20.0)
        return mean, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        std  = torch.exp(0.5 * logvar)
        eps  = torch.randn_like(std)
        z    = mean + std * eps
        xhat = self.decode(z)
        return xhat, mean, logvar

    def kl_loss(self, mean, logvar):
        """KL( N(mean,exp(logvar)) || N(0,1) ) summed over dims, mean over batch."""
        return 0.5 * (-1.0 - logvar + mean.pow(2) + logvar.exp()).sum(dim=[1, 2, 3]).mean()


# ─────────────────────────────────────────────────────────────────────────────
# PatchGAN discriminator (3-layer, ~1 M params for 32×32 images)
# ─────────────────────────────────────────────────────────────────────────────

class PatchDiscriminator(nn.Module):
    """NLayer PatchGAN: outputs logits over spatial patches."""

    def __init__(self, in_ch=3, base_ch=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1),
                  nn.LeakyReLU(0.2, inplace=True)]
        cur = base_ch
        for i in range(1, n_layers):
            nxt = min(cur * 2, 512)
            layers += [nn.Conv2d(cur, nxt, 4, stride=2 if i < n_layers - 1 else 1,
                                 padding=1),
                       nn.GroupNorm(min(32, nxt), nxt),
                       nn.LeakyReLU(0.2, inplace=True)]
            cur = nxt
        layers.append(nn.Conv2d(cur, 1, 4, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# VGG perceptual loss (no LPIPS dependency)
# ─────────────────────────────────────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    """L1 distance in VGG-16 feature space (relu1_2, relu2_2, relu3_3)."""

    def __init__(self, device):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features
        vgg.eval().requires_grad_(False).to(device)
        # Slices: relu1_2=4, relu2_2=9, relu3_3=16
        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:4]),    # relu1_2
            nn.Sequential(*list(vgg.children())[4:9]),   # relu2_2
            nn.Sequential(*list(vgg.children())[9:16]),  # relu3_3
        ])
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def _norm(self, x):
        """[-1,1] → ImageNet-normalised."""
        return (x * 0.5 + 0.5 - self.mean) / self.std

    def forward(self, pred, target):
        pred_n, target_n = self._norm(pred), self._norm(target)
        loss = 0.0
        hp, ht = pred_n, target_n
        for sl in self.slices:
            hp = sl(hp)
            ht = sl(ht)
            loss = loss + F.l1_loss(hp, ht.detach())
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive adversarial weight (VQGAN / LDM style)
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_adv_weight(nll_loss, g_loss, last_layer_weight, disc_weight=1.0):
    """Scale adversarial loss so its gradient magnitude matches reconstruction."""
    nll_grads = torch.autograd.grad(nll_loss, last_layer_weight, retain_graph=True)[0]
    g_grads   = torch.autograd.grad(g_loss,   last_layer_weight, retain_graph=True)[0]
    w = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    w = torch.clamp(w, 0.0, 1e4).detach()
    return w * disc_weight


# ─────────────────────────────────────────────────────────────────────────────
# Hinge losses
# ─────────────────────────────────────────────────────────────────────────────

def disc_hinge_loss(logits_real, logits_fake):
    """Discriminator hinge loss."""
    loss_real = F.relu(1.0 - logits_real).mean()
    loss_fake = F.relu(1.0 + logits_fake).mean()
    return 0.5 * (loss_real + loss_fake)


def gen_hinge_loss(logits_fake):
    """Generator hinge loss."""
    return -logits_fake.mean()


# ─────────────────────────────────────────────────────────────────────────────
# EMA helper
# ─────────────────────────────────────────────────────────────────────────────

def ema_update(src, tgt, decay):
    for sp, tp in zip(src.parameters(), tgt.parameters()):
        tp.data.mul_(decay).add_(sp.data, alpha=1 - decay)


# ─────────────────────────────────────────────────────────────────────────────
# Latent statistics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_latent_stats(vae, dataloader, device):
    """Compute per-channel mean and std of all training latents."""
    print('  Computing latent stats over training set …', flush=True)
    all_latents = []
    for x, _ in tqdm(dataloader, desc='Encoding', leave=False):
        mean, _ = vae.encode(x.to(device))
        all_latents.append(mean.cpu())
    all_latents = torch.cat(all_latents, dim=0)   # (N, z_ch, H, W)
    mean = all_latents.mean(dim=[0, 2, 3], keepdim=True)   # (1, z_ch, 1, 1)
    std  = all_latents.std(dim=[0, 2, 3], keepdim=True).clamp(min=1e-6)
    print(f'  Latent mean: {mean.squeeze().tolist()}', flush=True)
    print(f'  Latent std:  {std.squeeze().tolist()}',  flush=True)
    return mean, std


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction visualisation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def save_recon(vae, x_fixed, path, device):
    xhat, _, _ = vae(x_fixed.to(device))
    grid = make_grid(
        torch.cat([x_fixed.cpu(), xhat.cpu()], dim=0).clamp(-1, 1) * 0.5 + 0.5,
        nrow=x_fixed.size(0)
    )
    save_image(grid, path)


# ─────────────────────────────────────────────────────────────────────────────
# Loss summary plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_losses(loss_records, out_dir):
    keys = [k for k in loss_records[0].keys() if k != 'step']
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 3 * len(keys)), squeeze=False)
    steps = [r['step'] for r in loss_records]
    for i, k in enumerate(keys):
        vals = [r[k] for r in loss_records]
        axes[i][0].plot(steps, vals, lw=1)
        axes[i][0].set_ylabel(k)
        axes[i][0].set_xlabel('step')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'vae_loss.png'), dpi=120)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir',     type=str,   default='outputs/cifar10_vae')
    parser.add_argument('--data_dir',    type=str,   default='/tmp/cifar10_vae')
    parser.add_argument('--total_steps', type=int,   default=100_000)
    parser.add_argument('--batch_size',  type=int,   default=128)
    parser.add_argument('--lr',          type=float, default=4.5e-6)   # VQGAN-style small LR
    parser.add_argument('--disc_lr',     type=float, default=4.5e-6)
    parser.add_argument('--kl_weight',   type=float, default=1e-6)
    parser.add_argument('--perc_weight', type=float, default=1.0)
    parser.add_argument('--disc_weight', type=float, default=0.5)      # post-adaptive
    parser.add_argument('--disc_start',  type=int,   default=10_000)   # step to enable disc
    parser.add_argument('--kl_warmup',   type=int,   default=5_000)    # β ramps 0→kl_weight
    parser.add_argument('--ema_decay',   type=float, default=0.9999)
    parser.add_argument('--grad_clip',   type=float, default=1.0)
    parser.add_argument('--z_ch',        type=int,   default=4)
    parser.add_argument('--base_ch',     type=int,   default=128)
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--log_every',   type=int,   default=100)
    parser.add_argument('--save_every',  type=int,   default=20_000)
    parser.add_argument('--recon_every', type=int,   default=5_000)
    parser.add_argument('--resume',      type=str,   default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    tf_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds_train = datasets.CIFAR10(args.data_dir, train=True,  download=True, transform=tf)
    ds_val   = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=tf_val)

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, pin_memory=True)
    loader_val   = DataLoader(ds_val,   batch_size=64, shuffle=True,
                              num_workers=args.num_workers)

    # Fixed validation batch for visualisation
    x_fixed, _ = next(iter(loader_val))
    x_fixed = x_fixed[:16]

    # ── Models ───────────────────────────────────────────────────────────────
    ch_mult = (1, 2, 2)   # 128 → 256 → 256 channels; 32→16→8 spatial
    vae  = KLVAE(z_ch=args.z_ch, base_ch=args.base_ch, ch_mult=ch_mult).to(device)
    disc = PatchDiscriminator().to(device)
    vae_ema = copy.deepcopy(vae).requires_grad_(False)

    # Perceptual loss
    perc_loss_fn = VGGPerceptualLoss(device)

    # Latent shape
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 32, 32, device=device)
        z_dummy = vae.encode(dummy)[0]
    latent_shape = tuple(z_dummy.shape[1:])
    print(f'Latent shape: {latent_shape}')

    n_params_vae  = sum(p.numel() for p in vae.parameters())
    n_params_disc = sum(p.numel() for p in disc.parameters())
    print(f'VAE params: {n_params_vae / 1e6:.1f} M')
    print(f'Disc params: {n_params_disc / 1e6:.1f} M')

    # ── Optimisers ────────────────────────────────────────────────────────────
    optim_vae  = torch.optim.Adam(vae.parameters(),  lr=args.lr,      betas=(0.5, 0.9))
    optim_disc = torch.optim.Adam(disc.parameters(), lr=args.disc_lr, betas=(0.5, 0.9))

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    best_rec   = float('inf')
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        vae.load_state_dict(ckpt['vae'])
        disc.load_state_dict(ckpt['disc'])
        vae_ema.load_state_dict(ckpt['vae_ema'])
        optim_vae.load_state_dict(ckpt['optim_vae'])
        optim_disc.load_state_dict(ckpt['optim_disc'])
        start_step = ckpt.get('step', 0)
        best_rec   = ckpt.get('best_rec', float('inf'))
        print(f'Resumed from {args.resume} at step {start_step}', flush=True)

    # ── CSV logging ───────────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, 'loss.csv')
    csv_fields = ['step', 'loss_total', 'loss_rec', 'loss_kl', 'loss_perc',
                  'loss_adv_g', 'loss_disc', 'adv_weight']
    csv_file   = open(csv_path, 'a', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if start_step == 0:
        csv_writer.writeheader()

    loss_records = []

    # ── Infinite data iterator ────────────────────────────────────────────────
    def inf_loader(loader):
        while True:
            for batch in loader:
                yield batch

    data_iter = inf_loader(loader_train)

    # ── Last decoder layer (for adaptive weight) ──────────────────────────────
    last_dec_w = vae.decoder.conv_out.weight

    # ── Training ──────────────────────────────────────────────────────────────
    step = start_step
    t0   = time.time()

    pbar = tqdm(total=args.total_steps, initial=start_step, desc='VAE training')
    while step < args.total_steps:
        x, _ = next(data_iter)
        x = x.to(device)

        # KL warmup factor
        kl_w = args.kl_weight * min(1.0, step / max(1, args.kl_warmup))

        # ── Forward pass ─────────────────────────────────────────────────────
        xhat, mean, logvar = vae(x)

        loss_rec  = F.l1_loss(xhat, x)
        loss_kl   = vae.kl_loss(mean, logvar)
        loss_perc = perc_loss_fn(xhat, x)
        nll_loss  = loss_rec + args.perc_weight * loss_perc    # reconstruction objective

        # ── Adversarial generator loss (after disc_start) ────────────────────
        use_disc  = step >= args.disc_start
        adv_w     = 0.0
        loss_adv_g = torch.tensor(0.0, device=device)
        if use_disc:
            logits_fake_g = disc(xhat)
            loss_adv_g    = gen_hinge_loss(logits_fake_g)
            try:
                adv_w = adaptive_adv_weight(
                    nll_loss, loss_adv_g, last_dec_w, args.disc_weight).item()
            except RuntimeError:
                adv_w = args.disc_weight

        loss_vae = nll_loss + kl_w * loss_kl + adv_w * loss_adv_g

        optim_vae.zero_grad(set_to_none=True)
        loss_vae.backward()
        nn.utils.clip_grad_norm_(vae.parameters(), args.grad_clip)
        optim_vae.step()

        # ── Discriminator update (after disc_start) ───────────────────────────
        loss_disc = torch.tensor(0.0, device=device)
        if use_disc:
            logits_real = disc(x.detach())
            logits_fake = disc(xhat.detach())
            loss_disc   = disc_hinge_loss(logits_real, logits_fake)

            optim_disc.zero_grad(set_to_none=True)
            loss_disc.backward()
            nn.utils.clip_grad_norm_(disc.parameters(), args.grad_clip)
            optim_disc.step()

        # ── EMA ───────────────────────────────────────────────────────────────
        ema_update(vae, vae_ema, args.ema_decay)

        step += 1
        pbar.update(1)

        # ── Logging ───────────────────────────────────────────────────────────
        if step % args.log_every == 0:
            rec_val = loss_rec.item()
            row = {
                'step':       step,
                'loss_total': loss_vae.item(),
                'loss_rec':   rec_val,
                'loss_kl':    loss_kl.item(),
                'loss_perc':  loss_perc.item(),
                'loss_adv_g': loss_adv_g.item(),
                'loss_disc':  loss_disc.item(),
                'adv_weight': adv_w,
            }
            csv_writer.writerow(row)
            csv_file.flush()
            loss_records.append(row)
            elapsed = (time.time() - t0) / 60
            pbar.set_postfix(rec=f'{rec_val:.4f}', kl=f'{loss_kl.item():.3f}',
                             disc=f'{loss_disc.item():.3f}')

            if rec_val < best_rec:
                best_rec = rec_val
                torch.save({'vae': vae.state_dict(), 'vae_ema': vae_ema.state_dict(),
                            'disc': disc.state_dict(),
                            'optim_vae': optim_vae.state_dict(),
                            'optim_disc': optim_disc.state_dict(),
                            'step': step, 'best_rec': best_rec},
                           os.path.join(args.out_dir, 'vae_best.pt'))

        # ── Reconstruction visualisation ─────────────────────────────────────
        if step % args.recon_every == 0:
            vae.eval()
            save_recon(vae,     x_fixed, os.path.join(args.out_dir, f'recon_{step:07d}.png'), device)
            save_recon(vae_ema, x_fixed, os.path.join(args.out_dir, f'recon_ema_{step:07d}.png'), device)
            vae.train()

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if step % args.save_every == 0:
            torch.save({'vae': vae.state_dict(), 'vae_ema': vae_ema.state_dict(),
                        'disc': disc.state_dict(),
                        'optim_vae': optim_vae.state_dict(),
                        'optim_disc': optim_disc.state_dict(),
                        'step': step, 'best_rec': best_rec},
                       os.path.join(args.out_dir, f'vae_step_{step:07d}.pt'))
            print(f'\n[{step}] Saved checkpoint', flush=True)

    pbar.close()
    csv_file.close()

    # ── Final checkpoint ──────────────────────────────────────────────────────
    torch.save({'vae': vae.state_dict(), 'vae_ema': vae_ema.state_dict(),
                'disc': disc.state_dict(),
                'step': step},
               os.path.join(args.out_dir, 'vae_final.pt'))

    # ── Latent normalisation statistics ──────────────────────────────────────
    # Use the EMA model for stability
    vae_ema.eval()
    # Reload with no augmentation for unbiased stats
    loader_stat = DataLoader(
        datasets.CIFAR10(args.data_dir, train=True, download=False,
                         transform=tf_val),
        batch_size=256, shuffle=False, num_workers=args.num_workers)
    lat_mean, lat_std = compute_latent_stats(vae_ema, loader_stat, device)
    torch.save({'mean': lat_mean, 'std': lat_std, 'latent_shape': latent_shape},
               os.path.join(args.out_dir, 'latent_stats.pt'))
    print(f'Saved latent_stats.pt  →  {args.out_dir}', flush=True)

    # ── Final recon grid ──────────────────────────────────────────────────────
    save_recon(vae_ema, x_fixed, os.path.join(args.out_dir, 'recon_final.png'), device)

    # ── Loss plot ─────────────────────────────────────────────────────────────
    if loss_records:
        plot_losses(loss_records, args.out_dir)

    print(f'Done. Total time: {(time.time()-t0)/3600:.2f} h', flush=True)


if __name__ == '__main__':
    main()
