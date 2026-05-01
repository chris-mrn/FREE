"""
Compute Fisher-Rao speed on a trained latent FM model using Hutchinson estimator.

The latent FM model operates in VAE latent space (e.g., shape (B, C, H, W) = (B, 4, 8, 8)).
This script:
  1. Loads a trained latent FM checkpoint (trained with train_fm_latent_linear_ddp.py)
  2. Loads reference latents (normalised CIFAR-10)
  3. Estimates v_t^FR = sqrt(E[(div u_t^θ(X_t))²]) via Hutchinson divergence estimator
  4. Saves speed curve and weighting

IMPORTANT: Model architecture (num_channel, channel_mult) MUST match the training run.
For standard latent FM trained with train_fm_latent_linear_ddp.py:
  - num_channel=128 (default)
  - channel_mult=[1, 2] (default)

Usage:
  python compute_latent_fr_speed.py \\
      --ckpt outputs/cifar10_latent_linear/checkpoints/ckpt_step_0200000.pt \\
      --latent_stats outputs/cifar10_vae/latent_stats.pt \\
      --out_dir outputs/latent_speed_fr \\
      --speed_n_t 50 \\
      --speed_B 512 \\
      --speed_epochs 3 \\
      --speed_hutch 5 \\
      --channel_mult 1 2
"""
import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, '/nfs/ghome/live/cmarouani/FREE')

from torchcfm.models.unet.unet import UNetModelWrapper
from path.path import LinearPath
from speed.speed import estimate_speed_grid
from scripts.train_vae_cifar10 import KLVAE
from torchvision import datasets, transforms


def load_latent_stats(latent_stats_path):
    """Load latent mean and std from checkpoint."""
    ckpt = torch.load(latent_stats_path, map_location='cpu')
    lat_mean = ckpt['mean']
    lat_std = ckpt['std']
    return lat_mean, lat_std


def collect_reference_latents(vae, latent_stats_path, n_samples=500, data_dir='/tmp/cifar10_latent', device='cpu'):
    """
    Collect and normalise reference latents from VAE encoder.

    Returns (n_samples, C, H, W) normalised latents.
    """
    os.makedirs(data_dir, exist_ok=True)
    ds = datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, num_workers=4)

    vae.eval()
    latents = []
    with torch.no_grad():
        for imgs, _ in loader:
            if sum(z.shape[0] for z in latents) >= n_samples:
                break
            imgs = imgs.to(device)
            mean, _ = vae.encode(imgs)
            latents.append(mean.cpu())

    z = torch.cat(latents, 0)[:n_samples]

    # Normalise using latent stats
    lat_mean, lat_std = load_latent_stats(latent_stats_path)
    lat_mean = lat_mean.to(z.device)
    lat_std = lat_std.to(z.device)
    z_norm = (z - lat_mean) / lat_std

    return z_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to latent FM checkpoint (ema_step_*.pt)')
    parser.add_argument('--latent_stats', type=str, required=True,
                        help='Path to latent stats (latent_stats.pt)')
    parser.add_argument('--out_dir', type=str, default='outputs/latent_speed_fr')
    parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_latent')

    # Speed estimation hyperparams
    parser.add_argument('--speed_n_t', type=int, default=50,
                        help='Grid points for speed estimation')
    parser.add_argument('--speed_B', type=int, default=512,
                        help='Batch size per speed epoch')
    parser.add_argument('--speed_epochs', type=int, default=3,
                        help='Epochs over reference set for speed estimation')
    parser.add_argument('--speed_hutch', type=int, default=5,
                        help='Hutchinson probes for FR divergence estimator')

    # Model architecture (must match the trained checkpoint)
    parser.add_argument('--num_channel', type=int, default=128,
                        help='Base number of channels (must match training)')
    parser.add_argument('--channel_mult', type=int, nargs='+', default=[1, 2],
                        help='Channel multipliers (must match training). Latent FM uses [1, 2]')
    parser.add_argument('--vae_z_ch', type=int, default=4,
                        help='VAE latent channels (must match training)')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ─── Load checkpoint ───────────────────────────────────────────────────────
    print(f'Loading checkpoint: {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location=device)

    # Build model with same architecture as training
    latent_dim = (args.vae_z_ch, 8, 8)  # Standard for 4× downsampled 32×32 CIFAR-10
    _h = latent_dim[-1]  # spatial size (8)
    model = UNetModelWrapper(
        dim=latent_dim,
        num_res_blocks=2,
        num_channels=args.num_channel,
        channel_mult=tuple(args.channel_mult),  # Must match training!
        num_heads=4,
        num_head_channels=64,
        attention_resolutions=f'{_h}',  # '8' for 8×8 latents
        dropout=0.1,
    ).to(device)
    model.load_state_dict(ckpt['ema'])
    print(f'Loaded EMA model at step {ckpt.get("step", "unknown")}')

    # ─── Load VAE and collect reference latents ────────────────────────────────
    print('Collecting reference latents from VAE...')
    # Load VAE with standard CIFAR-10 configuration
    # VAE uses ch_mult=(1,2,2) which is different from FM's ch_mult=(1,2)
    vae = KLVAE(z_ch=args.vae_z_ch, base_ch=128, ch_mult=(1, 2, 2))

    # Try multiple VAE checkpoint paths
    vae_paths = [
        'outputs/cifar10_vae/vae_final.pt',
        'outputs/cifar10_vae/vae_best.pt',
        'outputs/cifar10_vae/vae_best.pt',
    ]
    vae_loaded = False
    for vae_ckpt_path in vae_paths:
        if os.path.exists(vae_ckpt_path):
            vae_ckpt = torch.load(vae_ckpt_path, map_location=device)
            # Handle different checkpoint formats
            if isinstance(vae_ckpt, dict):
                # Full checkpoint dict: try 'vae_ema', then 'vae'
                if 'vae_ema' in vae_ckpt:
                    state = vae_ckpt['vae_ema']
                elif 'vae' in vae_ckpt:
                    state = vae_ckpt['vae']
                else:
                    # Direct state dict
                    state = vae_ckpt
            else:
                state = vae_ckpt

            vae.load_state_dict(state)
            print(f'Loaded VAE from {vae_ckpt_path}')
            vae_loaded = True
            break

    if not vae_loaded:
        print(f'Warning: VAE checkpoint not found at any of: {vae_paths}')
        print('Using untrained VAE; results may be invalid.')

    vae = vae.to(device).eval()
    z_ref = collect_reference_latents(vae, args.latent_stats, n_samples=500, data_dir=args.data_dir, device=device)
    print(f'Reference latents shape: {z_ref.shape}, range: [{z_ref.min():.2f}, {z_ref.max():.2f}]')

    # ─── Set up path (latent FM uses linear path) ─────────────────────────────
    path = LinearPath()

    # ─── Estimate Fisher-Rao speed ─────────────────────────────────────────────
    print(f'\nEstimating Fisher-Rao speed with {args.speed_hutch} Hutchinson probes...')
    t_grid = np.linspace(0, path.T_MAX, args.speed_n_t + 1)

    v_fr = estimate_speed_grid(
        model, path, t_grid, z_ref.to(device),
        B=args.speed_B,
        n_epochs=args.speed_epochs,
        speed_type='fr',
        device=device,
        n_hutch=args.speed_hutch
    )

    print(f'\nFR speed range: {v_fr.min():.4f} -> {v_fr.max():.4f}')

    # ─── Compute weighting w(t) = L / v_t ─────────────────────────────────────
    def weighting(speeds, t_grid):
        """w(t) = L / v_t."""
        cum = np.zeros(len(t_grid))
        for i in range(1, len(t_grid)):
            cum[i] = np.trapezoid(speeds[:i+1], t_grid[:i+1])
        L = max(cum[-1], 1e-12)
        return np.where(speeds > 1e-12, L / speeds, 0.0)

    w_fr = weighting(v_fr, t_grid)
    print(f'Weighting (raw) range: {w_fr.min():.4f} -> {w_fr.max():.4f}')

    # ─── Save ─────────────────────────────────────────────────────────────────
    np.save(os.path.join(args.out_dir, 'fr_speed.npy'), v_fr)
    np.save(os.path.join(args.out_dir, 'fr_weighting.npy'), w_fr)
    np.save(os.path.join(args.out_dir, 't_grid.npy'), t_grid)

    print(f'\nSaved to {args.out_dir}/:')
    print(f'  fr_speed.npy')
    print(f'  fr_weighting.npy')
    print(f'  t_grid.npy')
    print('\nDone!')


if __name__ == '__main__':
    main()
