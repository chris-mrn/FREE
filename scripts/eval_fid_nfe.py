"""
Multi-NFE FID evaluation for a trained FM model (spherical or standard).

For each checkpoint found in --ckpt_dir, runs Euler sampling with each
NFE value in --nfe_list and computes FID against CIFAR-10 train set.
Merges results with any existing metrics.csv (FID already computed during
training) and prints a combined table + saves nfe_fid_table.csv.

Usage:
  python eval_fid_nfe.py \\
      --ckpt_dir  outputs/cifar10_spherical/fm_standard/checkpoints \\
      --metrics_csv outputs/cifar10_spherical/fm_standard/metrics.csv \\
      --out_dir   outputs/cifar10_spherical/fm_standard \\
      --nfe_list  35 50 100 \\
      --spherical          # omit for standard linear FM
"""
import os, sys, glob, argparse, math
import numpy as np
import csv
import torch
from torchvision import datasets, transforms

sys.path.insert(0, '/nfs/ghome/live/cmarouani/FREE')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torchcfm.models.unet.unet import UNetModelWrapper
from metrics import InceptionMetrics

T_MAX_SPHERICAL = math.pi / 2


# ── Euler samplers ─────────────────────────────────────────────────────────────

@torch.no_grad()
def euler_sample_spherical(model, n, n_steps, device, bs=256):
    model.eval()
    out, dt = [], T_MAX_SPHERICAL / n_steps
    for i in range(0, n, bs):
        b = min(bs, n - i)
        x = torch.randn(b, 3, 32, 32, device=device)
        for s in range(n_steps):
            t = torch.full((b,), s * dt, device=device)
            x = x + model(t, x) * dt
        out.append(x.cpu())
    model.train()
    return torch.cat(out)


@torch.no_grad()
def euler_sample_linear(model, n, n_steps, device, bs=256):
    model.eval()
    out, dt = [], 1.0 / n_steps
    for i in range(0, n, bs):
        b = min(bs, n - i)
        x = torch.randn(b, 3, 32, 32, device=device)
        for s in range(n_steps):
            t = torch.full((b,), s * dt, device=device)
            x = x + model(t, x) * dt
        out.append(x.cpu())
    model.train()
    return torch.cat(out)


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def load_ema(path, device, num_channel=128):
    net = UNetModelWrapper(
        dim=(3, 32, 32), num_res_blocks=2, num_channels=num_channel,
        channel_mult=[1, 2, 2, 2], num_heads=4, num_head_channels=64,
        attention_resolutions='16', dropout=0.0,
    ).to(device)
    ck = torch.load(path, map_location=device)
    key = 'ema' if 'ema' in ck else 'net'
    net.load_state_dict(ck[key])
    net.eval()
    return net


def step_from_path(path):
    """Extract step number from checkpoint filename."""
    name = os.path.basename(path)
    for part in name.replace('_', ' ').replace('.', ' ').split():
        if part.isdigit():
            return int(part)
    return -1


# ── Table printing ─────────────────────────────────────────────────────────────

def print_table(rows, nfe_list, existing_nfe=35):
    """
    rows: dict  step -> {nfe: fid_value}
    """
    all_nfe = sorted({existing_nfe} | set(nfe_list))
    col_w   = 12
    header  = f"{'Step':>8}  " + "  ".join(f"FID@{n:<{col_w-4}}" for n in all_nfe)
    sep     = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for step in sorted(rows):
        vals = rows[step]
        row  = f"{step:>8}  "
        row += "  ".join(
            f"{vals[n]:{col_w}.2f}" if n in vals else f"{'—':>{col_w}}"
            for n in all_nfe
        )
        print(row)
    print(sep)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir',    type=str, required=True)
    parser.add_argument('--out_dir',     type=str, required=True)
    parser.add_argument('--metrics_csv', type=str, default=None,
                        help='Existing metrics.csv to pull FID@35 from')
    parser.add_argument('--nfe_list',    type=int, nargs='+', default=[35, 50, 100])
    parser.add_argument('--n_samples',   type=int, default=10_000)
    parser.add_argument('--num_channel', type=int, default=128)
    parser.add_argument('--data_dir',    type=str, default='/tmp/cifar10_eval')
    parser.add_argument('--spherical',   action='store_true',
                        help='Use spherical Euler sampler (t in [0, pi/2])')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU:    {torch.cuda.get_device_name(0)}')
    sampler = euler_sample_spherical if args.spherical else euler_sample_linear
    print(f'Sampler: {"spherical" if args.spherical else "linear"} Euler')

    # ── Find checkpoints ──────────────────────────────────────────────────────
    ckpt_paths = sorted(glob.glob(os.path.join(args.ckpt_dir, '*.pt')))
    if not ckpt_paths:
        print(f'No checkpoints found in {args.ckpt_dir}'); return
    print(f'Found {len(ckpt_paths)} checkpoint(s):')
    for p in ckpt_paths:
        print(f'  {os.path.basename(p)}  (step {step_from_path(p):,})')

    # ── Load reference FID data from existing metrics.csv ─────────────────────
    # dict: step -> {nfe: fid}
    rows = {}
    existing_nfe = None
    if args.metrics_csv and os.path.exists(args.metrics_csv):
        with open(args.metrics_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = int(row['step'])
                fid  = float(row['fid'])
                rows.setdefault(step, {})[35] = fid   # training always uses 35 steps
        existing_nfe = 35
        print(f'\nLoaded {len(rows)} existing FID@35 entries from {args.metrics_csv}')

    # ── CIFAR-10 real stats ────────────────────────────────────────────────────
    print('\nLoading CIFAR-10 and computing reference statistics...')
    ds = datasets.CIFAR10(
        root=args.data_dir, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
    )
    real_loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)
    inception   = InceptionMetrics(device=device, batch_size=256)
    real_mu, real_sig, real_feats = inception.compute_real_stats(real_loader)
    print(f'Real stats computed over {len(real_feats)} images.')

    # ── Evaluate each checkpoint × each NFE ───────────────────────────────────
    new_rows = []   # for CSV output
    for ckpt_path in ckpt_paths:
        step = step_from_path(ckpt_path)
        print(f'\n{"─"*60}')
        print(f'Checkpoint: {os.path.basename(ckpt_path)}  (step {step:,})')
        model = load_ema(ckpt_path, device, args.num_channel)

        rows.setdefault(step, {})
        for nfe in args.nfe_list:
            if nfe in rows[step]:
                print(f'  NFE={nfe:3d}: FID={rows[step][nfe]:.2f}  (cached)')
                continue
            print(f'  NFE={nfe:3d}: sampling {args.n_samples} images...', end='', flush=True)
            import time; t0 = time.time()
            samples          = sampler(model, args.n_samples, nfe, device)
            feats_f, probs_f = inception.get_activations(samples)
            fid              = inception.compute_fid(real_mu, real_sig, feats_f)
            print(f'  FID={fid:.2f}  ({time.time()-t0:.0f}s)')
            rows[step][nfe] = fid
            new_rows.append({'step': step, 'nfe': nfe, 'fid': fid})

        del model
        torch.cuda.empty_cache()

    # ── Print table ───────────────────────────────────────────────────────────
    print_table(rows, args.nfe_list, existing_nfe=existing_nfe or args.nfe_list[0])

    # ── Save CSV ──────────────────────────────────────────────────────────────
    all_nfe = sorted({existing_nfe or 35} | set(args.nfe_list))
    out_csv = os.path.join(args.out_dir, 'nfe_fid_table.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step'] + [f'fid_nfe{n}' for n in all_nfe])
        for step in sorted(rows):
            w.writerow([step] + [f"{rows[step].get(n, '')}" for n in all_nfe])
    print(f'\nTable saved to: {out_csv}')


if __name__ == '__main__':
    main()
