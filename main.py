#!/usr/bin/env python3
"""
main.py — Unified Flow Matching CLI

Subcommands:
  train     Train a flow matching model
  evaluate  Sweep NFE values and compute FID/KID/IS on checkpoints
  compare   Compare speed profiles across methods or checkpoints
  analyze   Compute reparameterized divergence, speed analysis, or curriculum plots

Examples
--------
# Train linear FM on CIFAR-10
python main.py train --path linear --dataset cifar10 --out_dir outputs/myrun

# Train spherical FM with OT coupling and curriculum, multi-GPU
torchrun --nproc_per_node=4 main.py train \\
    --path spherical --dataset cifar10 --coupling ot \\
    --t_mode curriculum --speed_type ot --ddp \\
    --out_dir outputs/sph_curriculum

# Train 2D flow matching
python main.py train --path linear --dataset 8gaussians --out_dir outputs/2d_test

# Evaluate: sweep NFE on saved checkpoints
python main.py evaluate \\
    --ckpt_dir outputs/cifar10_linear/checkpoints \\
    --out_dir  outputs/cifar10_linear \\
    --nfe_list 10 20 35 50 100

# Compare speed methods
python main.py compare \\
    --speed_dir outputs/cifar10 \\
    --out_dir   outputs/comparison

# Analyze: compute reparameterized divergence
python main.py analyze --mode reparam_div \\
    --ckpt    outputs/cifar10_self_fm/checkpoints/ema_step_0200000.pt \\
    --speed_t outputs/cifar10_self_fm/fr_t_grid_step100000.npy \\
    --speed_v outputs/cifar10_self_fm/fr_speed_step100000.npy \\
    --out_dir outputs/cifar10_self_fm
"""

import argparse
import os
import sys


# ── Argument builders ──────────────────────────────────────────────────────────

def add_train_args(parser):
    # Core axes
    parser.add_argument('--path',     default='linear',
                        choices=['linear', 'spherical'])
    parser.add_argument('--dataset',  default='cifar10',
                        help='cifar10 | 8gaussians | 40gaussians | moons | circles | checkerboard')
    parser.add_argument('--coupling', default='independent',
                        choices=['independent', 'ot'])
    parser.add_argument('--t_mode',   default='uniform',
                        choices=['uniform', 'weighted', 'curriculum'])
    parser.add_argument('--speed_type', default='ot',
                        choices=['ot', 'fr', 'score'],
                        help='Speed measure for weighted/curriculum t-sampling')

    # Speed source (weighted mode)
    parser.add_argument('--speed_dir', default=None,
                        help='Dir with precomputed *_speed*.npy files (weighted mode)')

    # Curriculum
    parser.add_argument('--curriculum_start', type=int, default=100_000)
    parser.add_argument('--curriculum_blend', type=int, default=25_000)
    parser.add_argument('--curriculum_restarts', type=int, default=0)
    parser.add_argument('--curriculum_restart_every', type=int, default=50_000)

    # Speed estimation hyperparams
    parser.add_argument('--speed_n_t',    type=int,   default=100)
    parser.add_argument('--speed_B',      type=int,   default=512)
    parser.add_argument('--speed_epochs', type=int,   default=3)
    parser.add_argument('--speed_hutch',  type=int,   default=4)
    parser.add_argument('--speed_smooth', type=float, default=0.05)

    # Training
    parser.add_argument('--total_steps', type=int,   default=400_001)
    parser.add_argument('--batch_size',  type=int,   default=128)
    parser.add_argument('--lr',          type=float, default=2e-4)
    parser.add_argument('--warmup',      type=int,   default=5_000)
    parser.add_argument('--ema_decay',   type=float, default=0.9999)
    parser.add_argument('--grad_clip',   type=float, default=1.0)
    parser.add_argument('--eval_every',  type=int,   default=20_000)
    parser.add_argument('--fid_samples', type=int,   default=10_000)
    parser.add_argument('--nfe_list',    type=int,   nargs='+', default=[10, 20, 35, 50, 100])
    parser.add_argument('--keep_ckpts', type=int,   default=2)
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--seed',        type=int,   default=42)

    # Model
    parser.add_argument('--num_channel', type=int,   default=128)
    parser.add_argument('--hidden_2d',   type=int,   default=256)
    parser.add_argument('--depth_2d',    type=int,   default=4)

    # I/O
    parser.add_argument('--data_dir', default='/tmp/fm_data')
    parser.add_argument('--out_dir',  required=True)
    parser.add_argument('--resume',   default='auto',
                        help='"auto" to find latest checkpoint, path to specific file, '
                             'or "disabled" to start fresh')
    parser.add_argument('--ddp', action='store_true',
                        help='Enable multi-GPU via torchrun')


def add_evaluate_args(parser):
    parser.add_argument('--ckpt_dir',   required=True,
                        help='Directory containing checkpoint .pt files')
    parser.add_argument('--out_dir',    required=True)
    parser.add_argument('--path',       default='linear', choices=['linear', 'spherical'])
    parser.add_argument('--nfe_list',   type=int, nargs='+', default=[10, 20, 35, 50, 100])
    parser.add_argument('--fid_samples', type=int, default=10_000)
    parser.add_argument('--data_dir',   default='/tmp/fm_data')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_channel', type=int, default=128)
    parser.add_argument('--latent',     action='store_true',
                        help='Evaluate latent FM (requires --vae_ckpt and --latent_stats)')
    parser.add_argument('--vae_ckpt',   default=None)
    parser.add_argument('--latent_stats', default=None)


def add_compare_args(parser):
    parser.add_argument('--speed_dir',  required=True,
                        help='Directory with precomputed speed .npy files')
    parser.add_argument('--out_dir',    required=True)
    parser.add_argument('--path',       default='linear', choices=['linear', 'spherical'])
    parser.add_argument('--ckpt_model', default=None,
                        help='Optional checkpoint for fresh Hutchinson FR inference')
    parser.add_argument('--data_dir',   default='/tmp/fm_data')
    parser.add_argument('--n_hutch',    type=int, default=8)
    parser.add_argument('--n_t',        type=int, default=100)
    parser.add_argument('--num_channel', type=int, default=128)


def add_analyze_args(parser):
    parser.add_argument('--mode',      required=True,
                        choices=['reparam_div', 'speed_analysis', 'curriculum_plot'],
                        help='Analysis task to run')
    parser.add_argument('--ckpt',      default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--out_dir',   required=True)
    parser.add_argument('--speed_t',   default=None,
                        help='t_grid .npy path (for reparam_div / speed_analysis)')
    parser.add_argument('--speed_v',   default=None,
                        help='v_t .npy path (for reparam_div / speed_analysis)')
    parser.add_argument('--data_dir',  default='/tmp/fm_data')
    parser.add_argument('--n_t',       type=int, default=200)
    parser.add_argument('--B',         type=int, default=256)
    parser.add_argument('--n_hutch',   type=int, default=5)
    parser.add_argument('--n_epochs',  type=int, default=3)
    parser.add_argument('--num_channel', type=int, default=128)


# ── Subcommand handlers ────────────────────────────────────────────────────────

def cmd_train(args):
    from training.trainer import run_training
    # 'disabled' maps to empty string so the resume logic in trainer skips auto-find
    if args.resume == 'disabled':
        args.resume = ''
    run_training(args)


def cmd_evaluate(args):
    """NFE sweep evaluation over checkpoints."""
    import glob, csv, math
    import numpy as np
    import torch
    from torchvision import datasets as tv_datasets, transforms
    from path import get_path, euler_sample
    from evaluation.metrics import InceptionMetrics
    from models.unet import UNetModelWrapper

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path   = get_path(args.path)
    os.makedirs(args.out_dir, exist_ok=True)

    # Reference stats
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds      = tv_datasets.CIFAR10(args.data_dir, train=True, download=True, transform=tfm)
    loader  = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False,
                                          num_workers=args.num_workers)
    print('Computing reference statistics...')
    inception = InceptionMetrics(device=device)
    real_mu, real_sig, real_feats = inception.compute_real_stats(loader)

    ckpt_files = sorted(glob.glob(os.path.join(args.ckpt_dir, 'ckpt_step_*.pt')))
    if not ckpt_files:
        print(f'No checkpoints found in {args.ckpt_dir}')
        return

    out_csv = open(os.path.join(args.out_dir, 'nfe_fid_table.csv'), 'w', newline='')
    writer  = csv.writer(out_csv)
    writer.writerow(['ckpt', 'step', 'nfe', 'fid', 'kid_mean', 'is_mean', 'is_std'])

    for ckpt_path in ckpt_files:
        step = int(os.path.basename(ckpt_path).replace('ckpt_step_', '').replace('.pt', ''))
        print(f'\n=== Checkpoint step {step} ===')

        net = UNetModelWrapper(
            dim=(3, 32, 32), num_res_blocks=2, num_channels=args.num_channel,
            channel_mult=[1, 2, 2, 2], num_heads=4, num_head_channels=64,
            attention_resolutions='16', dropout=0.1,
        ).to(device)
        ck    = torch.load(ckpt_path, map_location=device)
        state = ck.get('ema', ck.get('net', ck))
        net.load_state_dict(state)
        net.eval()

        for nfe in args.nfe_list:
            print(f'  NFE={nfe}...', flush=True)
            samples         = euler_sample(net, path, args.fid_samples, nfe, (3, 32, 32), device)
            feats_f, prob_f = inception.get_activations(samples)
            fid             = inception.compute_fid(real_mu, real_sig, feats_f)
            kid_m, _        = inception.compute_kid(real_feats, feats_f)
            is_m, is_s      = inception.compute_is(prob_f)
            print(f'    FID={fid:.3f}  KID={kid_m:.4f}  IS={is_m:.3f}±{is_s:.3f}')
            writer.writerow([os.path.basename(ckpt_path), step, nfe, fid, kid_m, is_m, is_s])
            out_csv.flush()

    out_csv.close()
    print(f'\nResults saved to {args.out_dir}/nfe_fid_table.csv')


def cmd_compare(args):
    """Compare speed profiles across methods."""
    import numpy as np
    import torch
    from evaluation.compare import (
        compute_weighting, smooth_weighting, compute_density, interp_to,
    )
    from evaluation.speed import load_precomputed

    os.makedirs(args.out_dir, exist_ok=True)
    path_name = args.path

    results = {}
    for speed_type in ('ot', 'score', 'fr'):
        try:
            t_grid, v_t = load_precomputed(args.speed_dir, path_name, speed_type)
            w     = compute_weighting(v_t, t_grid)
            w_sm  = smooth_weighting(w, t_grid)
            p     = compute_density(w_sm, t_grid)
            results[speed_type] = dict(t_grid=t_grid, v_t=v_t, w=w, w_sm=w_sm, p=p)
            print(f'Loaded {speed_type}: v∈[{v_t.min():.3f},{v_t.max():.3f}]')
        except FileNotFoundError:
            print(f'  {speed_type}: precomputed files not found, skipping')

    if not results:
        print('No precomputed speed files found. Run cifar10_speed_all.py first.')
        return

    # Generate comparison plot
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'ot': 'steelblue', 'score': 'tomato', 'fr': 'seagreen'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, data in results.items():
        t, v = data['t_grid'], data['v_t']
        axes[0].plot(t, v,         color=colors[name], label=name, lw=1.5)
        axes[1].plot(t, data['w_sm'], color=colors[name], label=name, lw=1.5)
        axes[2].plot(t, data['p'],    color=colors[name], label=name, lw=1.5)

    axes[0].set_title('Speed $v_t$'); axes[0].set_xlabel('$t$')
    axes[1].set_title('Weighting $w(t)$'); axes[1].set_xlabel('$t$')
    axes[2].set_title('Sampling density $p(t)$'); axes[2].set_xlabel('$t$')
    for ax in axes:
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(args.out_dir, 'compare_speed_methods.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved → {out_png}')


def cmd_analyze(args):
    """Route to specific analysis task."""
    if args.mode == 'reparam_div':
        _analyze_reparam_div(args)
    elif args.mode == 'speed_analysis':
        _analyze_speed(args)
    elif args.mode == 'curriculum_plot':
        _analyze_curriculum_plot(args)


def _analyze_reparam_div(args):
    """Compute reparameterized divergence under arc-length schedule."""
    import numpy as np
    import torch
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from torchvision import datasets as tv_datasets, transforms
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from evaluation.energy import build_alpha, hutchinson_div_sq
    from models.unet import UNetModelWrapper

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t_grid = np.load(args.speed_t)
    v_t    = np.load(args.speed_v)
    alpha, t_ext, cdf, w_raw = build_alpha(t_grid, v_t)

    T_MIN, T_MAX = 0.2, 0.8
    s_grid   = np.linspace(0.0, 1.0, args.n_t, dtype=np.float32)
    tau_grid = alpha(s_grid)
    t_uniform = np.linspace(T_MIN + 0.01, T_MAX - 0.01, args.n_t, dtype=np.float32)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    loader = DataLoader(
        tv_datasets.CIFAR10(args.data_dir, train=True, download=True, transform=tfm),
        batch_size=512, shuffle=True, num_workers=4)
    x1_pool = next(iter(loader))[0]

    net = UNetModelWrapper(
        dim=(3, 32, 32), num_res_blocks=2, num_channels=args.num_channel,
        channel_mult=[1, 2, 2, 2], num_heads=4, num_head_channels=64,
        attention_resolutions='16', dropout=0.1,
    ).to(device)
    ck    = torch.load(args.ckpt, map_location=device)
    state = ck.get('ema', ck.get('net', ck))
    net.load_state_dict(state)
    net.eval()

    from evaluation.energy import hutchinson_div_sq

    def sample_xt_self(pool, t_val, B, dev):
        N = len(pool)
        x1       = pool[torch.randint(0, N, (B,))].to(dev)
        x1_tilde = pool[torch.randint(0, N, (B,))].to(dev)
        import math
        sigma_t = float(math.sqrt(t_val * (1.0 - t_val)))
        eps = torch.randn_like(x1)
        return ((1.0 - t_val) * x1 + t_val * x1_tilde + sigma_t * eps).detach()

    div_reparam = np.zeros((args.n_epochs, args.n_t))
    div_uniform = np.zeros((args.n_epochs, args.n_t))

    for ep in range(args.n_epochs):
        x1_pool = next(iter(loader))[0]
        for i, (s_val, tau_val, t_val) in enumerate(tqdm(
                zip(s_grid, tau_grid, t_uniform), total=args.n_t,
                desc=f'ep{ep+1}')):
            xt_r = sample_xt_self(x1_pool, float(tau_val), args.B, device)
            div_reparam[ep, i] = hutchinson_div_sq(net, float(s_val), xt_r, args.n_hutch, device)
            xt_u = sample_xt_self(x1_pool, float(t_val), args.B, device)
            div_uniform[ep, i] = hutchinson_div_sq(net, float(t_val), xt_u, args.n_hutch, device)

    mean_r, std_r = div_reparam.mean(0), div_reparam.std(0)
    mean_u, std_u = div_uniform.mean(0), div_uniform.std(0)

    np.save(os.path.join(args.out_dir, 'div_sq_reparam_mean.npy'), mean_r)
    np.save(os.path.join(args.out_dir, 'div_sq_uniform_mean.npy'), mean_u)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].fill_between(t_uniform, mean_u - std_u, mean_u + std_u, alpha=0.2, color='steelblue')
    axes[0].plot(t_uniform, mean_u, color='steelblue', lw=1.5)
    axes[0].set_title('(A) Raw — no reparam'); axes[0].set_yscale('log')
    axes[1].fill_between(s_grid, mean_r - std_r, mean_r + std_r, alpha=0.2, color='tomato')
    axes[1].plot(s_grid, mean_r, color='tomato', lw=1.5)
    axes[1].set_title('(B) After arc-length reparam'); axes[1].set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'div_sq_reparam.png'), dpi=150)
    plt.close()
    print('Done.')


def _analyze_speed(args):
    """Speed analysis using precomputed or freshly estimated speed profiles."""
    import numpy as np
    from evaluation.compare import compute_weighting, smooth_weighting, compute_density

    if args.speed_t and args.speed_v:
        t_grid = np.load(args.speed_t)
        v_t    = np.load(args.speed_v)
        w     = compute_weighting(v_t, t_grid)
        w_sm  = smooth_weighting(w, t_grid)
        p     = compute_density(w_sm, t_grid)
        print(f'Speed: t∈[{t_grid[0]:.3f},{t_grid[-1]:.3f}]  '
              f'v∈[{v_t.min():.3f},{v_t.max():.3f}]')
        print(f'Weight CV: {w_sm.std()/w_sm.mean():.3f}')
        print(f'Density ratio (max/min): {p.max()/p.min():.2f}x')
    else:
        print('Provide --speed_t and --speed_v for speed analysis.')


def _analyze_curriculum_plot(args):
    """Plot speed/weighting/density from curriculum checkpoint speed files."""
    import glob, numpy as np
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from evaluation.compare import compute_weighting, smooth_weighting, compute_density

    os.makedirs(args.out_dir, exist_ok=True)
    t_files = sorted(glob.glob(os.path.join(args.out_dir, 'speed_t_grid_step*.npy')))
    v_files = sorted(glob.glob(os.path.join(args.out_dir, 'speed_v_t_step*.npy')))

    if not t_files:
        print(f'No speed files found in {args.out_dir}')
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for t_path, v_path in zip(t_files, v_files):
        label = os.path.basename(t_path).replace('speed_t_grid_', '').replace('.npy', '')
        t_grid = np.load(t_path)
        v_t    = np.load(v_path)
        w     = compute_weighting(v_t, t_grid)
        w_sm  = smooth_weighting(w, t_grid)
        p     = compute_density(w_sm, t_grid)
        axes[0].plot(t_grid, v_t,  label=label, lw=1.5)
        axes[1].plot(t_grid, w_sm, label=label, lw=1.5)
        axes[2].plot(t_grid, p,    label=label, lw=1.5)

    axes[0].set_title('Speed $v_t$')
    axes[1].set_title('Weighting $w(t)$')
    axes[2].set_title('Sampling density')
    for ax in axes:
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlabel('$t$')
    plt.tight_layout()
    out_png = os.path.join(args.out_dir, 'curriculum_speed_plot.png')
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f'Plot saved → {out_png}')


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Flow Matching CLI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p_train    = sub.add_parser('train',    help='Train a flow matching model',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_evaluate = sub.add_parser('evaluate', help='Sweep NFE and compute FID/KID/IS',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_compare  = sub.add_parser('compare',  help='Compare speed profiles',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_analyze  = sub.add_parser('analyze',  help='Compute divergence / speed analysis',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_train_args(p_train)
    add_evaluate_args(p_evaluate)
    add_compare_args(p_compare)
    add_analyze_args(p_analyze)

    args = parser.parse_args()

    dispatch = {
        'train':    cmd_train,
        'evaluate': cmd_evaluate,
        'compare':  cmd_compare,
        'analyze':  cmd_analyze,
    }
    dispatch[args.command](args)


if __name__ == '__main__':
    main()
