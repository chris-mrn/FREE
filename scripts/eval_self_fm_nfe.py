"""
NFE sweep evaluation for a Self-FM checkpoint (data-to-data interpolant).

Inference: start from real CIFAR-10 training images (X1), integrate
the learned vector field from t=T_MAX to t=T_MIN using Euler with
uniform steps. This matches the training interpolant:
    X_t = (1-t)*X1 + t*X1_tilde + sqrt(t(1-t))*eps
At t=T_MAX≈1, X_t ≈ X1_tilde (target); integrate back to t=T_MIN≈0
to produce a novel sample.

Usage:
  python eval_self_fm_nfe.py \\
      --ckpt_dir  outputs/cifar10_self_fm/checkpoints \\
      --out_dir   outputs/cifar10_self_fm \\
      --nfe_list  5 10 20 35 50 100 200 \\
      --n_samples 10000 \\
      --data_dir  /tmp/cifar10_eval
"""
import os, sys, glob, argparse, math
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader

sys.path.insert(0, '/nfs/ghome/live/cmarouani/FREE')
from torchcfm.models.unet.unet import UNetModelWrapper
from evaluation.metrics import InceptionMetrics

T_MIN = 0.01
T_MAX = 0.99


# ── Self-FM Euler sampler ──────────────────────────────────────────────────────

@torch.no_grad()
def euler_sample_self_fm(model, x1_pool, n_samples, n_steps, device, bs=256):
    """
    Integrate the self-FM ODE from t=T_MAX → t=T_MIN with n_steps Euler steps.
    Starting points are drawn randomly from x1_pool (real CIFAR-10 images).
    Returns (n_samples, 3, 32, 32) in [-1, 1].
    """
    model.eval()
    dt = (T_MAX - T_MIN) / n_steps
    out = []
    for i in range(0, n_samples, bs):
        b = min(bs, n_samples - i)
        idx = torch.randint(0, len(x1_pool), (b,))
        x   = x1_pool[idx].to(device)              # start from real image
        # Integrate backwards: t from T_MAX down to T_MIN
        for s in range(n_steps):
            t_val = T_MAX - s * dt
            t     = torch.full((b,), t_val, device=device)
            v     = model(t, x)
            x     = x - v * dt                     # backward Euler step
        out.append(x.cpu())
    model.train()
    return torch.cat(out, 0)


# ── Checkpoint discovery ───────────────────────────────────────────────────────

def find_checkpoints(ckpt_dir):
    """Return list of (step, path) sorted by step."""
    paths = glob.glob(os.path.join(ckpt_dir, 'ema_step_*.pt'))
    result = []
    for p in paths:
        name = os.path.basename(p)
        try:
            step = int(name.replace('ema_step_', '').replace('.pt', '').replace('_final', ''))
            result.append((step, p))
        except ValueError:
            pass
    return sorted(result)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir',  required=True)
    parser.add_argument('--out_dir',   required=True)
    parser.add_argument('--nfe_list',  type=int, nargs='+', default=[5, 10, 20, 35, 50, 100, 200])
    parser.add_argument('--n_samples', type=int, default=10_000)
    parser.add_argument('--data_dir',  default='/tmp/cifar10_eval')
    parser.add_argument('--num_channel', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--only_steps', type=int, nargs='+', default=None,
                        help='If set, only evaluate checkpoints at these steps')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'eval_samples'), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    if device.type == 'cuda':
        print(f'GPU    : {torch.cuda.get_device_name(0)}')
    print(f'Config : n_samples={args.n_samples}, NFE list={args.nfe_list}')
    print(f'Interpolant: X_t = (1-t)*X1 + t*X1_tilde + sqrt(t(1-t))*eps, t∈[{T_MIN},{T_MAX}]')
    print(f'Sampling: start from CIFAR-10 data images (NOT Gaussian noise)\n')

    # ── Load CIFAR-10 ──────────────────────────────────────────────────────────
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Collect pool of starting images (draw randomly from train set at inference)
    x1_pool = []
    for imgs, _ in loader:
        x1_pool.append(imgs)
        if sum(x.shape[0] for x in x1_pool) >= args.n_samples * 2:
            break
    x1_pool = torch.cat(x1_pool, 0)
    print(f'Starting-point images: {x1_pool.shape}  (real CIFAR-10 train)\n')

    # ── InceptionV3 reference stats ───────────────────────────────────────────
    print('Computing InceptionV3 statistics on real CIFAR-10 (this takes ~2 min) ...')
    inception = InceptionMetrics(device=device, batch_size=args.batch_size)
    real_mu, real_sig, real_feats = inception.compute_real_stats(loader)
    print(f'Real stats done  ({len(ds):,} images)\n')

    # ── Build model ───────────────────────────────────────────────────────────
    net = UNetModelWrapper(
        dim=(3, 32, 32), num_res_blocks=2, num_channels=args.num_channel,
        channel_mult=[1, 2, 2, 2], num_heads=4, num_head_channels=64,
        attention_resolutions='16', dropout=0.1,
    ).to(device)

    # ── Discover checkpoints ──────────────────────────────────────────────────
    ckpts = find_checkpoints(args.ckpt_dir)
    if args.only_steps is not None:
        ckpts = [(s, p) for s, p in ckpts if s in args.only_steps]
    print(f'Found {len(ckpts)} checkpoint(s):')
    for step, path in ckpts:
        print(f'  {os.path.basename(path)}  (step {step:,})')

    # ── CSV output ────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, 'nfe_fid_table.csv')
    rows = []

    # ── Evaluate each checkpoint × each NFE ───────────────────────────────────
    for step, ckpt_path in ckpts:
        print(f'\n{"="*60}')
        print(f'Checkpoint : {os.path.basename(ckpt_path)}  (step {step:,})')
        print(f'{"="*60}')
        ck = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(ck['ema'])
        net.eval()

        for nfe in args.nfe_list:
            import time as _time
            t0 = _time.time()
            print(f'\n  NFE={nfe} — generating {args.n_samples:,} images ...')
            samples = euler_sample_self_fm(
                net, x1_pool, args.n_samples, nfe, device, bs=args.batch_size)

            feats_f, probs_f = inception.get_activations(samples)
            fid   = inception.compute_fid(real_mu, real_sig, feats_f)
            kid_m, kid_s = inception.compute_kid(real_feats, feats_f)
            is_m, is_s   = inception.compute_is(probs_f)
            elapsed = _time.time() - t0

            print(f'  NFE={nfe:>3}: FID={fid:.4f}  KID={kid_m:.3f}±{kid_s:.3f}'
                  f'  IS={is_m:.3f}±{is_s:.3f}  ({elapsed:.0f}s)')

            if nfe == 35:
                grid = make_grid(samples[:64].clamp(-1, 1), nrow=8,
                                 normalize=True, value_range=(-1, 1))
                save_image(grid, os.path.join(
                    args.out_dir, 'eval_samples',
                    f'samples_step{step:07d}_nfe{nfe}.png'))

            rows.append({'step': step, 'nfe': nfe,
                         'fid': round(fid, 4),
                         'kid_mean': round(kid_m, 6), 'kid_std': round(kid_s, 6),
                         'is_mean': round(is_m, 4), 'is_std': round(is_s, 4)})

    # ── Write CSV ─────────────────────────────────────────────────────────────
    import csv
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['step','nfe','fid','kid_mean','kid_std','is_mean','is_std'])
        w.writeheader()
        w.writerows(rows)
    print(f'\nCSV → {csv_path}')

    # ── Print summary table ────────────────────────────────────────────────────
    print('\n\n── Summary: FID by (step, NFE) ──')
    steps = sorted(set(r['step'] for r in rows))
    nfes  = args.nfe_list
    header = 'Step  | ' + ' | '.join(f'NFE={n:>3}' for n in nfes)
    print(header)
    print('-' * len(header))
    for s in steps:
        row_fids = []
        for n in nfes:
            match = [r['fid'] for r in rows if r['step'] == s and r['nfe'] == n]
            row_fids.append(f'{match[0]:7.3f}' if match else '    —  ')
        print(f'{s:6d} | ' + ' | '.join(row_fids))

    print('\nAll done!')


if __name__ == '__main__':
    main()
