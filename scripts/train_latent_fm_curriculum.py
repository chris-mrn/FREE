"""
Continue training latent FM with Fisher-Rao curriculum learning.

This script:
  1. Loads the latest latent FM checkpoint
  2. Computes Fisher-Rao speed using Hutchinson estimator
  3. Continues training with speed-adaptive curriculum learning

The curriculum blends from uniform sampling to speed-weighted sampling,
allowing the model to focus training time on regions where the flow changes rapidly.

Usage:
  python scripts/train_latent_fm_curriculum.py \\
      --ckpt_dir outputs/cifar10_latent_linear/checkpoints \\
      --latent_stats outputs/cifar10_vae/latent_stats.pt \\
      --out_dir outputs/cifar10_latent_linear_curriculum \\
      --speed_n_t 1000 \\
      --speed_epochs 10 \\
      --speed_hutch 5 \\
      --curriculum_start 50000 \\
      --curriculum_blend 25000 \\
      --additional_steps 200000 \\
      --batch_size 256
"""
import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '/nfs/ghome/live/cmarouani/FREE')

from torchcfm.models.unet.unet import UNetModelWrapper
from path.path import LinearPath
from speed.speed import estimate_speed_grid, BlendedSampler, UniformSampler, CdfSampler
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from scripts.train_vae_cifar10 import KLVAE
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from metrics.metrics import InceptionMetrics
import csv


def find_latest_checkpoint(ckpt_dir):
    """Find the latest checkpoint in directory."""
    ckpt_dir = Path(ckpt_dir)
    ckpts = sorted(ckpt_dir.glob('ckpt_step_*.pt'))
    if not ckpts:
        raise ValueError(f"No checkpoints found in {ckpt_dir}")
    return ckpts[-1]


def load_checkpoint(ckpt_path, device='cpu'):
    """Load checkpoint and return model, optimizer state, step."""
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt


def load_latent_stats(latent_stats_path):
    """Load latent mean and std from checkpoint."""
    ckpt = torch.load(latent_stats_path, map_location='cpu')
    lat_mean = ckpt['mean']
    lat_std = ckpt['std']
    return lat_mean, lat_std


def collect_reference_latents(vae, latent_stats_path, n_samples=500, data_dir='/tmp/cifar10_latent', device='cpu'):
    """Collect and normalise reference latents from VAE encoder."""
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


def get_vae_checkpoint_path(base_paths):
    """Try multiple VAE checkpoint paths."""
    for path in base_paths:
        if os.path.exists(path):
            return path
    return None


def cosine_blend(progress):
    """Cosine annealing blend: 0->1 as progress 0->1."""
    return 0.5 * (1.0 - np.cos(np.pi * progress))


def sample_latents(model, path, n_samples, n_steps, device):
    """Sample latents from the model."""
    latent_shape = (4, 8, 8)
    z0 = torch.randn(n_samples, *latent_shape, device=device)

    model.eval()
    with torch.no_grad():
        t_grid = np.linspace(0, path.T_MAX, n_steps)
        z = z0
        for i in range(n_steps - 1):
            t = torch.full((n_samples,), t_grid[i], device=device)
            dt = t_grid[i + 1] - t_grid[i]
            vel = model(t, z)
            z = z + vel * dt

    return z


def decode_latents_to_images(vae, latents, latent_stats_path, device):
    """Decode latents to image space."""
    lat_mean, lat_std = load_latent_stats(latent_stats_path)
    lat_mean = lat_mean.to(device)
    lat_std = lat_std.to(device)

    latents_unnorm = latents * lat_std + lat_mean

    vae.eval()
    with torch.no_grad():
        images = vae.decode(latents_unnorm)

    return images


def compute_fid_nfe_sweep(model, vae, path, latent_stats_path, nfe_list=[10, 30, 35, 50, 100, 200], n_samples=256, device='cpu'):
    """Compute FID scores for different NFE (sampling step counts)."""
    from torchvision import datasets, transforms

    # Compute reference stats on real CIFAR-10
    print('    Computing reference CIFAR-10 stats...')
    ds = datasets.CIFAR10(
        root='/tmp/cifar10_eval', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
    )
    real_loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)
    metrics = InceptionMetrics(device=device, batch_size=256)
    real_mu, real_sig, _ = metrics.compute_real_stats(real_loader)
    results = {}

    for n_steps in nfe_list:
        print(f'    NFE={n_steps}...', end='', flush=True)
        try:
            # Sample and decode
            z_latent = sample_latents(model, path, n_samples, n_steps, device)
            x_gen = decode_latents_to_images(vae, z_latent, latent_stats_path, device)

            # Clip to [-1, 1]
            x_gen = torch.clamp(x_gen, -1, 1)

            # Compute FID
            feats_gen, _ = metrics.get_activations(x_gen)
            fid = metrics.compute_fid(real_mu, real_sig, feats_gen)
            results[n_steps] = fid
            print(f' FID={fid:.4f}')
        except Exception as e:
            print(f' FAILED ({e})')
            results[n_steps] = None

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help='Directory with latent FM checkpoints')
    parser.add_argument('--latent_stats', type=str, required=True,
                        help='Path to latent stats')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory for continued training')
    parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_latent')

    # Speed estimation
    parser.add_argument('--speed_n_t', type=int, default=1000,
                        help='Grid points for FR speed estimation')
    parser.add_argument('--speed_B', type=int, default=512,
                        help='Batch size for speed estimation')
    parser.add_argument('--speed_epochs', type=int, default=10,
                        help='Epochs for speed estimation')
    parser.add_argument('--speed_hutch', type=int, default=5,
                        help='Hutchinson probes for FR divergence')

    # Curriculum learning
    parser.add_argument('--curriculum_start', type=int, default=50000,
                        help='Step to start curriculum blending')
    parser.add_argument('--curriculum_blend', type=int, default=25000,
                        help='Steps to blend uniform -> speed-adaptive')

    # Training
    parser.add_argument('--additional_steps', type=int, default=100000,
                        help='Additional training steps')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ema_decay', type=float, default=0.9999)

    # Model
    parser.add_argument('--num_channel', type=int, default=128)
    parser.add_argument('--channel_mult', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--vae_z_ch', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    # ─── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = find_latest_checkpoint(args.ckpt_dir)
    print(f'Loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device)

    current_step = ckpt.get('step', 0)
    print(f'Current step: {current_step}')

    # Build model
    latent_dim = (args.vae_z_ch, 8, 8)
    _h = latent_dim[-1]
    model = UNetModelWrapper(
        dim=latent_dim,
        num_res_blocks=2,
        num_channels=args.num_channel,
        channel_mult=tuple(args.channel_mult),
        num_heads=4,
        num_head_channels=64,
        attention_resolutions=f'{_h}',
        dropout=0.1,
    ).to(device)
    model.load_state_dict(ckpt['ema'])
    print(f'Loaded EMA model')

    # ─── Load VAE and collect reference latents ────────────────────────────────
    print('Loading VAE and collecting reference latents...')
    vae = KLVAE(z_ch=args.vae_z_ch, base_ch=128, ch_mult=(1, 2, 2))

    vae_paths = [
        'outputs/cifar10_vae/vae_final.pt',
        'outputs/cifar10_vae/vae_best.pt',
        'outputs/cifar10_vae/vae_best.pt',
    ]
    vae_ckpt_path = get_vae_checkpoint_path(vae_paths)
    if vae_ckpt_path:
        vae_ckpt = torch.load(vae_ckpt_path, map_location=device)
        if isinstance(vae_ckpt, dict):
            if 'vae_ema' in vae_ckpt:
                state = vae_ckpt['vae_ema']
            elif 'vae' in vae_ckpt:
                state = vae_ckpt['vae']
            else:
                state = vae_ckpt
        else:
            state = vae_ckpt
        vae.load_state_dict(state)
        print(f'Loaded VAE from {vae_ckpt_path}')
    else:
        print(f'Warning: VAE checkpoint not found')

    vae = vae.to(device).eval()
    z_ref = collect_reference_latents(vae, args.latent_stats, n_samples=500,
                                     data_dir=args.data_dir, device=device)
    print(f'Reference latents shape: {z_ref.shape}')

    # ─── Load or compute Fisher-Rao speed ─────────────────────────────────────
    path = LinearPath()
    speed_dir = os.path.join(args.out_dir, 'speed_profile')
    speed_files = [
        os.path.join(speed_dir, 'fr_speed.npy'),
        os.path.join(speed_dir, 'fr_weighting.npy'),
        os.path.join(speed_dir, 't_grid.npy'),
    ]

    if all(os.path.exists(f) for f in speed_files):
        print(f'\nLoading precomputed Fisher-Rao speed from {speed_dir}')
        v_fr = np.load(speed_files[0])
        w_fr = np.load(speed_files[1])
        t_grid = np.load(speed_files[2])
        print(f'Loaded: v_fr shape {v_fr.shape}, w_fr shape {w_fr.shape}, t_grid shape {t_grid.shape}')
        print(f'FR speed range: {v_fr.min():.4f} -> {v_fr.max():.4f}')
        print(f'Weighting range: {w_fr.min():.4f} -> {w_fr.max():.4f}')
    else:
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

        print(f'FR speed range: {v_fr.min():.4f} -> {v_fr.max():.4f}')

        # Compute weighting
        def weighting(speeds, t_grid):
            cum = np.zeros(len(t_grid))
            for i in range(1, len(t_grid)):
                cum[i] = np.trapezoid(speeds[:i+1], t_grid[:i+1])
            L = max(cum[-1], 1e-12)
            return np.where(speeds > 1e-12, L / speeds, 0.0)

        w_fr = weighting(v_fr, t_grid)
        print(f'Weighting range: {w_fr.min():.4f} -> {w_fr.max():.4f}')

        # Save speed profile
        os.makedirs(speed_dir, exist_ok=True)
        np.save(os.path.join(speed_dir, 'fr_speed.npy'), v_fr)
        np.save(os.path.join(speed_dir, 'fr_weighting.npy'), w_fr)
        np.save(os.path.join(speed_dir, 't_grid.npy'), t_grid)
        print(f'Saved speed profile to {speed_dir}')

    # ─── Set up curriculum samplers ────────────────────────────────────────────
    print(f'\nSetting up curriculum learning...')
    uniform_sampler = UniformSampler(T_MAX=path.T_MAX)
    fr_sampler = CdfSampler(t_grid, v_fr, path.T_MAX, smooth_sigma=0.05)

    curriculum_sampler = BlendedSampler(uniform_sampler, fr_sampler)

    # ─── Set up training ──────────────────────────────────────────────────────
    print(f'\nSetting up training (additional {args.additional_steps} steps)...')
    fm = ConditionalFlowMatcher()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.additional_steps, eta_min=1e-6)

    # Load training data
    ds = datasets.CIFAR10(
        root=args.data_dir, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # ─── Set up metrics logging ───────────────────────────────────────────────
    metrics_csv = os.path.join(args.out_dir, 'metrics.csv')
    nfe_list = [10, 30, 35, 50, 100, 200]
    fieldnames = ['step', 'loss', 'mix'] + [f'fid_nfe{nfe}' for nfe in nfe_list]
    csv_file = open(metrics_csv, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    # ─── Training loop ────────────────────────────────────────────────────────
    model.train()
    step = current_step
    end_step = current_step + args.additional_steps

    print(f'\nTraining from step {step} to {end_step}')
    print(f'Curriculum blend: uniform -> FR-weighted over steps 0 -> {args.curriculum_blend}')
    print(f'Then: pure FR-weighted from step {args.curriculum_blend} onwards')
    print(f'='*70)

    loss_history = []
    eval_steps = [args.curriculum_blend // 2, args.curriculum_blend, args.curriculum_blend + 50000]
    eval_steps = [s for s in eval_steps if s < end_step]
    if end_step not in eval_steps:
        eval_steps.append(end_step - 1)

    while step < end_step:
        for imgs, _ in loader:
            if step >= end_step:
                break

            imgs = imgs.to(device)
            B = imgs.shape[0]

            # Encode to latent space
            with torch.no_grad():
                z1, _ = vae.encode(imgs)
                # Normalise
                lat_mean = load_latent_stats(args.latent_stats)[0].to(device)
                lat_std = load_latent_stats(args.latent_stats)[1].to(device)
                z1 = (z1 - lat_mean) / lat_std

            # Sample z0 (Gaussian noise)
            z0 = torch.randn_like(z1)

            # Curriculum blending: start at step 0
            if step < args.curriculum_blend:
                progress = step / args.curriculum_blend
                mix = cosine_blend(progress)  # Cosine blend from uniform (0) to FR-weighted (1)
            else:
                mix = 1.0  # Pure FR-weighted

            # Sample t using blended sampler
            t = curriculum_sampler.sample(B, device=device, mix=mix)

            # Get conditional flow
            t_expanded = t.to(device=device, dtype=torch.float32)
            t_returned, xt, ut = fm.sample_location_and_conditional_flow(z0, z1, t=t_expanded)

            # Forward pass
            vt = model(t_expanded, xt)

            # Loss
            loss = F.mse_loss(vt, ut)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            loss_history.append(loss.item())

            if step % 100 == 0:
                avg_loss = np.mean(loss_history[-100:])
                print(f'Step {step:7d}/{end_step} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Mix: {mix:.3f}')

            # Checkpoint and evaluate every 50k steps
            if step % 50000 == 0 and step > current_step:
                ckpt_path = os.path.join(args.out_dir, f'ckpt_step_{step:07d}.pt')
                torch.save({
                    'model': model.state_dict(),
                    'ema': model.state_dict(),  # Simplified; use proper EMA in production
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'step': step,
                }, ckpt_path)
                print(f'  → Saved checkpoint: {ckpt_path}')

                # Compute FID sweep
                if step in eval_steps:
                    print(f'  → Computing FID sweep (NFE={nfe_list})...')
                    try:
                        fid_results = compute_fid_nfe_sweep(
                            model, vae, path, args.latent_stats,
                            nfe_list=nfe_list, n_samples=256, device=device
                        )
                        row = {'step': step, 'loss': avg_loss, 'mix': mix}
                        for nfe, fid in fid_results.items():
                            row[f'fid_nfe{nfe}'] = fid
                        csv_writer.writerow(row)
                        csv_file.flush()
                    except Exception as e:
                        print(f'  → FID computation failed: {e}')

            step += 1

    # Final checkpoint
    ckpt_path = os.path.join(args.out_dir, f'ckpt_step_{step:07d}.pt')
    torch.save({
        'model': model.state_dict(),
        'ema': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step,
    }, ckpt_path)
    print(f'\nFinal checkpoint saved: {ckpt_path}')

    # Final FID evaluation
    print(f'Computing final FID sweep (NFE={nfe_list})...')
    try:
        fid_results = compute_fid_nfe_sweep(
            model, vae, path, args.latent_stats,
            nfe_list=nfe_list, n_samples=256, device=device
        )
        row = {'step': step, 'loss': np.mean(loss_history[-100:]), 'mix': 1.0}
        for nfe, fid in fid_results.items():
            row[f'fid_nfe{nfe}'] = fid
        csv_writer.writerow(row)
    except Exception as e:
        print(f'Final FID computation failed: {e}')

    csv_file.close()
    print(f'\nMetrics saved to {metrics_csv}')
    print(f'Training complete! Trained from step {current_step} to {step}')


if __name__ == '__main__':
    main()
