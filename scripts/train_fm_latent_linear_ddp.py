"""
Linear Flow Matching (I-CFM) on VAE latents — DDP multi-GPU training.

Interpolation path (linear / OT-FM):
  X_t = (1-t) * X0 + t * X1,   t ∈ [0, 1]
  Conditional velocity: u_t(X_t | X0, X1) = X1 - X0

Source:  X0 ~ N(0, I)
Target:  X1 ~ normalised latent  z = (enc(x) - mean) / std

Best practices:
  ✓ Pre-computed latents (avoids re-encoding each step)
  ✓ Per-channel latent normalisation so X1 has zero mean / unit std
  ✓ Random horizontal flip augmentation in latent space
  ✓ EMA model, gradient clipping, LR warmup
  ✓ Linear LR scaling with world_size
  ✓ VAE decoder for pixel-space FID evaluation
  ✓ Checkpoint every eval_every steps, auto-resume

The UNet velocity field is sized for latent space:
  dim = (z_ch, latent_h, latent_w),   e.g. (4, 8, 8) for 4× downsampled 32×32

Launch (DDP):
  torchrun --nproc_per_node=N train_fm_latent_linear_ddp.py [args...]

Requires:
  outputs/cifar10_vae/vae_best.pt   (or --vae_ckpt)
  outputs/cifar10_vae/latent_stats.pt    (or --latent_stats)
"""

import os
import sys
import copy
import time
import csv
import math
import glob
import argparse
from datetime import datetime

sys.path.insert(0, '/nfs/ghome/live/cmarouani/FREE')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torchcfm.models.unet.unet import UNetModelWrapper
from evaluation.metrics import InceptionMetrics

# We import the VAE architecture from the local training script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_vae_cifar10 import KLVAE


# ─────────────────────────────────────────────────────────────────────────────
# DDP helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_dist():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size(), local_rank == 0


def ddp_infiniteloop(loader, sampler):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for x in loader:
            yield x
        epoch += 1


def ema_update(src, tgt, decay):
    for sp, tp in zip(src.parameters(), tgt.parameters()):
        tp.data.mul_(decay).add_(sp.data, alpha=1 - decay)


def warmup_lr(step, warmup):
    return min(step + 1, warmup) / warmup


# ─────────────────────────────────────────────────────────────────────────────
# Latent dataset
# ─────────────────────────────────────────────────────────────────────────────

class LatentDataset(Dataset):
    """Normalised latents with optional horizontal flip augmentation."""

    def __init__(self, latents: torch.Tensor, flip: bool = True):
        self.latents = latents   # (N, C, H, W), already normalised
        self.flip    = flip

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        z = self.latents[idx]
        if self.flip and torch.rand(1).item() < 0.5:
            z = torch.flip(z, dims=[-1])
        return z


# ─────────────────────────────────────────────────────────────────────────────
# VAE encode / decode helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_all(vae, dataloader, device):
    """Encode all images to latents (encoder mean, no noise)."""
    chunks = []
    for x, _ in tqdm(dataloader, desc='Encoding latents', leave=False):
        mean, _ = vae.encode(x.to(device))
        chunks.append(mean.cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def decode_latents(vae, z_norm, lat_mean, lat_std, device, bs=64):
    """Denormalise then decode latents → pixels in [-1, 1]."""
    vae.eval()
    out = []
    lat_mean = lat_mean.to(device)
    lat_std  = lat_std.to(device)
    for i in range(0, z_norm.shape[0], bs):
        zb = z_norm[i:i+bs].to(device)
        zb = zb * lat_std + lat_mean
        imgs = vae.decode(zb).clamp(-1, 1)
        out.append(imgs.cpu())
    return torch.cat(out, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Euler sampler (linear ODE)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def euler_sample_latent(model, n, n_steps, latent_shape, device, bs=256):
    """Euler integration X_t = X_0 + integral(v_theta) from t=0 to 1."""
    model.eval()
    out = []
    dt = 1.0 / n_steps
    for i in range(0, n, bs):
        b = min(bs, n - i)
        x = torch.randn(b, *latent_shape, device=device)
        for s in range(n_steps):
            t = torch.full((b,), s * dt, device=device)
            x = x + model(t, x) * dt
        out.append(x.cpu())
    model.train()
    return torch.cat(out, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_ckpt(path, net_module, ema_net, optim, sched, step, loss_ema):
    torch.save({
        'net':      net_module.state_dict(),
        'ema':      ema_net.state_dict(),
        'optim':    optim.state_dict(),
        'sched':    sched.state_dict(),
        'step':     step,
        'loss_ema': loss_ema,
    }, path)


def find_last_checkpoint(out_dir):
    files = sorted(glob.glob(f'{out_dir}/checkpoints/ckpt_step_*.pt'))
    return files[-1] if files else None


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(step, ema_model, vae, lat_mean, lat_std, inception,
             real_mu, real_sig, real_feats, out_dir, n_samples,
             latent_shape, device, n_steps=35):
    t0 = time.time()
    z_norm  = euler_sample_latent(ema_model, n_samples, n_steps, latent_shape, device)
    samples = decode_latents(vae, z_norm, lat_mean, lat_std, device)

    grid = make_grid(samples[:64].clamp(-1, 1), nrow=8, normalize=True, value_range=(-1, 1))
    save_image(grid, f'{out_dir}/samples/step_{step:07d}.png')

    feats_f, probs_f = inception.get_activations(samples)
    fid              = inception.compute_fid(real_mu, real_sig, feats_f)
    kid_m, kid_s     = inception.compute_kid(real_feats, feats_f)
    is_m,  is_s      = inception.compute_is(probs_f)
    elapsed = time.time() - t0
    print(f'  [eval {step}] FID={fid:.3f}  KID={kid_m:.3f}±{kid_s:.3f}'
          f'  IS={is_m:.2f}±{is_s:.2f}  ({elapsed:.0f}s)', flush=True)
    return fid, kid_m, kid_s, is_m, is_s


# ─────────────────────────────────────────────────────────────────────────────
# Summary plots
# ─────────────────────────────────────────────────────────────────────────────

def make_summary_plots(out_dir):
    try:
        import pandas as pd
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Latent Linear FM — Training Summary', fontsize=13)

        loss_df = pd.read_csv(f'{out_dir}/loss.csv')
        roll_std = loss_df['loss_raw'].rolling(50, min_periods=1).std()
        ax = axes[0, 0]
        ax.plot(loss_df['step'], loss_df['loss_ema'], lw=1.2)
        ax.fill_between(loss_df['step'],
                        loss_df['loss_ema'] - roll_std,
                        loss_df['loss_ema'] + roll_std, alpha=0.2)
        ax.set_title('Training loss'); ax.set_xlabel('step'); ax.grid(True, alpha=0.3)

        m = pd.read_csv(f'{out_dir}/metrics.csv')
        axes[0, 1].plot(m['step'], m['fid'], 'o-', color='C1', lw=1.5)
        axes[0, 1].set_title('FID ↓'); axes[0, 1].set_xlabel('step')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(m['step'], m['kid_mean'], 'o-', color='C2', lw=1.5)
        axes[1, 0].fill_between(m['step'], m['kid_mean'] - m['kid_std'],
                                m['kid_mean'] + m['kid_std'], alpha=0.2, color='C2')
        axes[1, 0].set_title('KID ↓'); axes[1, 0].set_xlabel('step')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(m['step'], m['is_mean'], 'o-', color='C3', lw=1.5)
        axes[1, 1].fill_between(m['step'], m['is_mean'] - m['is_std'],
                                m['is_mean'] + m['is_std'], alpha=0.2, color='C3')
        axes[1, 1].set_title('Inception Score ↑'); axes[1, 1].set_xlabel('step')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{out_dir}/training_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out_dir}/training_summary.png')
    except Exception as e:
        print(f'[warning] Summary plot failed: {e}')


# ─────────────────────────────────────────────────────────────────────────────
# Training report (markdown)
# ─────────────────────────────────────────────────────────────────────────────

def generate_training_report(args, out_dir, t_total_h, world_size, gpu_name, start_time_str):
    try:
        import pandas as pd
        m = pd.read_csv(f'{out_dir}/metrics.csv')
        best_idx = m['fid'].idxmin()
        best_step, best_fid = int(m.loc[best_idx, 'step']), m.loc[best_idx, 'fid']
        last = m.iloc[-1]

        lines = [
            f'# Training Report — Latent Linear FM — {datetime.now().strftime("%Y-%m-%d")}',
            '',
            '## Configuration',
            '',
            f'| Parameter | Value |',
            f'|-----------|-------|',
            f'| Script | `train_fm_latent_linear_ddp.py` |',
            f'| VAE checkpoint | `{args.vae_ckpt}` |',
            f'| Latent stats | `{args.latent_stats}` |',
            f'| Interpolation | $X_t = (1-t)X_0 + tX_1$ (linear) |',
            f'| Source | $X_0 \\sim \\mathcal{{N}}(0,I)$ (latent space) |',
            f'| GPUs | {world_size}× {gpu_name} |',
            f'| Per-GPU batch | {args.batch_size} |',
            f'| Effective batch | {args.batch_size * world_size} |',
            f'| Total steps | {args.total_steps} |',
            f'| LR | {args.lr} × {world_size} = {args.lr * world_size:.2e} |',
            f'| Total time | {t_total_h:.2f} h |',
            '',
            '## FID / KID / IS Progression',
            '',
            '| Step | FID ↓ | KID mean | IS mean |',
            '|------|-------|----------|---------|',
        ]
        for _, row in m.iterrows():
            marker = ' ← **best**' if int(row['step']) == best_step else ''
            lines.append(f"| {int(row['step'])} | {row['fid']:.3f}{marker} | "
                         f"{row['kid_mean']:.3f}±{row['kid_std']:.3f} | "
                         f"{row['is_mean']:.2f}±{row['is_std']:.2f} |")

        lines += [
            '',
            f'**Best FID**: {best_fid:.3f} @ step {best_step}',
            f'**Final FID**: {last["fid"]:.3f} @ step {int(last["step"])}',
            '',
            '## Notes',
            '',
            '- Latent space: 4×8×8 (4× spatial downsampling from 32×32)',
            '- Latents normalised per-channel to zero mean / unit std before FM training',
            '- Velocity field: small UNet on 8×8 spatial latents',
            '- Decoded via trained VAE for FID evaluation',
        ]
        report_path = f'{out_dir}/training_report_{datetime.now().strftime("%Y-%m-%d")}.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f'Report saved: {report_path}')
    except Exception as e:
        print(f'[warning] Report generation failed: {e}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--vae_ckpt',     type=str,   default='outputs/cifar10_vae/vae_best.pt')
    parser.add_argument('--latent_stats', type=str,   default='outputs/cifar10_vae/latent_stats.pt')
    parser.add_argument('--out_dir',      type=str,   default='outputs/cifar10_latent_linear')
    parser.add_argument('--data_dir',     type=str,   default='/tmp/cifar10_latent')
    # Model
    parser.add_argument('--num_channel',  type=int,   default=128)
    # Training
    parser.add_argument('--total_steps',  type=int,   default=100_000)
    parser.add_argument('--batch_size',   type=int,   default=128)
    parser.add_argument('--lr',           type=float, default=2e-4)
    parser.add_argument('--warmup',       type=int,   default=1_000)
    parser.add_argument('--ema_decay',    type=float, default=0.9999)
    parser.add_argument('--grad_clip',    type=float, default=1.0)
    parser.add_argument('--num_workers',  type=int,   default=4)
    # Evaluation
    parser.add_argument('--eval_every',   type=int,   default=10_000)
    parser.add_argument('--fid_samples',  type=int,   default=2_000)
    parser.add_argument('--n_steps',      type=int,   default=35)
    # VAE architecture (must match trained VAE)
    parser.add_argument('--vae_z_ch',     type=int,   default=4)
    parser.add_argument('--vae_base_ch',  type=int,   default=128)
    # Resume
    parser.add_argument('--resume',       type=str,   default='auto',
                        help='"auto" to find last checkpoint, or path to .pt file')
    args = parser.parse_args()

    # ── DDP setup ────────────────────────────────────────────────────────────
    local_rank, world_size, is_main = setup_dist()
    device = torch.device(f'cuda:{local_rank}')

    if is_main:
        os.makedirs(f'{args.out_dir}/checkpoints', exist_ok=True)
        os.makedirs(f'{args.out_dir}/samples',     exist_ok=True)
        gpu_name = torch.cuda.get_device_name(local_rank)
        print(f'=== Latent Linear FM (DDP): world_size={world_size} ===')
        print(f'GPU: {gpu_name}')
        print(f'Per-GPU batch={args.batch_size}, effective batch={args.batch_size * world_size}')
        print(f'LR: {args.lr} × {world_size} = {args.lr * world_size:.2e}')
    else:
        gpu_name = ''

    # ── Load VAE ─────────────────────────────────────────────────────────────
    if is_main:
        print(f'Loading VAE from {args.vae_ckpt}', flush=True)
    vae = KLVAE(z_ch=args.vae_z_ch, base_ch=args.vae_base_ch, ch_mult=(1, 2, 2)).to(device)

    # Support both full checkpoint dict and raw state dict
    vae_ckpt_data = torch.load(args.vae_ckpt, map_location=device)
    if 'vae_ema' in vae_ckpt_data:
        vae.load_state_dict(vae_ckpt_data['vae_ema'])
    elif 'vae' in vae_ckpt_data:
        vae.load_state_dict(vae_ckpt_data['vae'])
    else:
        vae.load_state_dict(vae_ckpt_data)
    vae.eval().requires_grad_(False)

    # ── Latent normalisation stats ────────────────────────────────────────────
    stats = torch.load(args.latent_stats, map_location='cpu')
    lat_mean  = stats['mean']                # (1, z_ch, 1, 1)
    lat_std   = stats['std']                 # (1, z_ch, 1, 1)
    latent_shape = stats['latent_shape']     # e.g. (4, 8, 8)
    if is_main:
        print(f'Latent shape: {latent_shape}')
        print(f'Latent mean (ch): {lat_mean.squeeze().tolist()}')
        print(f'Latent std  (ch): {lat_std.squeeze().tolist()}')

    # ── Pre-compute / cache latents ───────────────────────────────────────────
    cache_path = f'{args.out_dir}/latents_norm.pt'
    if is_main:
        if os.path.exists(cache_path):
            print(f'Loading cached normalised latents: {cache_path}', flush=True)
            latents_norm = torch.load(cache_path, map_location='cpu')
        else:
            tf_enc = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            ds_enc = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=tf_enc)
            loader_enc = DataLoader(ds_enc, batch_size=256, shuffle=False,
                                    num_workers=args.num_workers)
            raw_latents = encode_all(vae, loader_enc, device)
            lat_mean_d = lat_mean.to(device)
            lat_std_d  = lat_std.to(device)
            latents_norm = ((raw_latents.to(device) - lat_mean_d) / lat_std_d).cpu()
            torch.save(latents_norm, cache_path)
            print(f'Saved normalised latents {tuple(latents_norm.shape)} → {cache_path}')
    dist.barrier()

    if not is_main:
        latents_norm = torch.load(cache_path, map_location='cpu')

    if is_main:
        print(f'Latents: {tuple(latents_norm.shape)}  '
              f'mean={latents_norm.mean():.4f}  std={latents_norm.std():.4f}', flush=True)

    # ── Latent DataLoader ─────────────────────────────────────────────────────
    lat_dataset = LatentDataset(latents_norm, flip=True)
    lat_sampler = DistributedSampler(lat_dataset, num_replicas=world_size,
                                     rank=local_rank, shuffle=True)
    lat_loader  = DataLoader(lat_dataset, batch_size=args.batch_size,
                             sampler=lat_sampler, num_workers=args.num_workers,
                             drop_last=True, pin_memory=True)
    lat_iter    = ddp_infiniteloop(lat_loader, lat_sampler)

    # ── FM velocity-field model ───────────────────────────────────────────────
    _h = latent_shape[-1]   # spatial size (e.g. 8)
    net = UNetModelWrapper(
        dim=latent_shape,
        num_res_blocks=2,
        num_channels=args.num_channel,
        channel_mult=(1, 2),
        num_heads=4,
        num_head_channels=64,
        attention_resolutions=f'{_h}',
        dropout=0.1,
    ).to(device)

    ema_net = copy.deepcopy(net).requires_grad_(False)
    net_ddp = DDP(net, device_ids=[local_rank])

    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    if is_main:
        print(f'FM model params: {n_params:.1f} M', flush=True)

    # ── Optimiser + LR scheduler ──────────────────────────────────────────────
    scaled_lr = args.lr * world_size
    optim = torch.optim.Adam(net.parameters(), lr=scaled_lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: warmup_lr(step, args.warmup))

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    loss_ema   = None
    if args.resume == 'auto':
        ckpt_path = find_last_checkpoint(args.out_dir)
    else:
        ckpt_path = args.resume if (args.resume and os.path.exists(args.resume)) else None

    if ckpt_path:
        if is_main:
            print(f'Resuming from {ckpt_path}', flush=True)
        ckpt = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(ckpt['net'])
        ema_net.load_state_dict(ckpt['ema'])
        optim.load_state_dict(ckpt['optim'])
        sched.load_state_dict(ckpt['sched'])
        start_step = ckpt.get('step', 0)
        loss_ema   = ckpt.get('loss_ema', None)
    dist.barrier()

    # ── Metrics / reference stats ─────────────────────────────────────────────
    if is_main:
        print('Loading InceptionV3 & computing reference statistics …', flush=True)
        # Decode training latents to pixel space for reference statistics
        inception = InceptionMetrics(device=device)
        # Sample a reference set: decode a subset of training latents
        # Use real CIFAR-10 images (ground truth, not decoded) for reference
        tf_ref = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        ds_ref  = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=tf_ref)
        ref_loader = DataLoader(ds_ref, batch_size=256, shuffle=False, num_workers=4)
        real_mu, real_sig, real_feats = inception.compute_real_stats(ref_loader)
        print(f'Reference stats: {len(ds_ref)} images.', flush=True)

    # ── CSV logger ────────────────────────────────────────────────────────────
    if is_main:
        csv_path = f'{args.out_dir}/loss.csv'
        csv_file = open(csv_path, 'a', newline='')
        csv_writer = csv.DictWriter(csv_file, fieldnames=['step', 'loss_raw', 'loss_ema'])
        if not os.path.exists(csv_path) or start_step == 0:
            csv_writer.writeheader()

        met_path = f'{args.out_dir}/metrics.csv'
        met_file = open(met_path, 'a', newline='')
        met_writer = csv.DictWriter(met_file,
                                    fieldnames=['step', 'fid', 'kid_mean', 'kid_std',
                                                'is_mean', 'is_std'])
        if not os.path.exists(met_path) or start_step == 0:
            met_writer.writeheader()

    start_time = time.time()
    start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ── Training loop ─────────────────────────────────────────────────────────
    step = start_step
    pbar = tqdm(total=args.total_steps, initial=start_step, disable=not is_main,
                desc='Latent FM')

    while step < args.total_steps:
        x1 = next(lat_iter).to(device)   # (B, z_ch, H, W) normalised latent
        B  = x1.shape[0]

        # Sample source noise and interpolation time
        x0 = torch.randn_like(x1)
        t  = torch.rand(B, device=device)
        tb = t.view(B, 1, 1, 1)

        # Linear interpolation: X_t = (1-t)*X0 + t*X1
        xt = (1.0 - tb) * x0 + tb * x1
        # Constant conditional velocity: u_t = X1 - X0
        ut = x1 - x0

        # Velocity prediction and MSE loss
        vt   = net_ddp(t, xt)
        loss = (vt - ut).pow(2).mean()

        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        optim.step()
        sched.step()

        ema_update(net, ema_net, args.ema_decay)

        step += 1
        pbar.update(1)

        loss_val = loss.item()
        loss_ema = loss_val if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_val

        if is_main and step % 100 == 0:
            csv_writer.writerow({'step': step, 'loss_raw': loss_val, 'loss_ema': loss_ema})
            csv_file.flush()
            pbar.set_postfix(loss=f'{loss_ema:.4f}')

        # ── Evaluation ───────────────────────────────────────────────────────
        if is_main and step % args.eval_every == 0:
            fid, kid_m, kid_s, is_m, is_s = run_eval(
                step, ema_net, vae, lat_mean, lat_std, inception,
                real_mu, real_sig, real_feats,
                args.out_dir, args.fid_samples,
                latent_shape, device, args.n_steps,
            )
            met_writer.writerow({'step': step, 'fid': fid, 'kid_mean': kid_m,
                                 'kid_std': kid_s, 'is_mean': is_m, 'is_std': is_s})
            met_file.flush()

            ckpt_dir = f'{args.out_dir}/checkpoints'
            save_ckpt(f'{ckpt_dir}/ckpt_step_{step:07d}.pt',
                      net, ema_net, optim, sched, step, loss_ema)
            # Keep only last 3 checkpoints
            old = sorted(glob.glob(f'{ckpt_dir}/ckpt_step_*.pt'))[:-3]
            for p in old:
                os.remove(p)

        dist.barrier()

    pbar.close()

    # ── Post-training ─────────────────────────────────────────────────────────
    t_total_h = (time.time() - start_time) / 3600

    if is_main:
        csv_file.close()
        met_file.close()

        # Final eval
        if step % args.eval_every != 0:
            fid, kid_m, kid_s, is_m, is_s = run_eval(
                step, ema_net, vae, lat_mean, lat_std, inception,
                real_mu, real_sig, real_feats,
                args.out_dir, args.fid_samples,
                latent_shape, device, args.n_steps,
            )

        # Save final checkpoint
        save_ckpt(f'{args.out_dir}/checkpoints/ckpt_step_{step:07d}.pt',
                  net, ema_net, optim, sched, step, loss_ema)

        make_summary_plots(args.out_dir)
        generate_training_report(args, args.out_dir, t_total_h,
                                 world_size, gpu_name, start_time_str)
        print(f'\nDone. Total time: {t_total_h:.2f} h', flush=True)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
