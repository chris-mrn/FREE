"""
Train a 2D flow-matching model on 8gaussians with a 3-phase FR curriculum.

Phase 1 (0 → phase2_start):        Uniform t-sampling
Phase 2 (phase2_start → total_steps): FR-adaptive t-sampling (direct switch, no blend)
  - At phase2_start, FR speed is estimated via exact divergence (2 backward passes, exact for dim=2)
  - Same CdfSampler is used for the rest of training (Phase 3 is implicit continuation of Phase 2)

Usage:
    PYTHONPATH=. python scripts/train_8gauss_curriculum.py \
        --dataset 8gaussians --total_steps 200001 --phase2_start 50000 \
        --out_dir outputs/8gauss_curriculum

Outputs (out_dir/):
  checkpoints/ckpt_step_XXXXXXX.pt  -- every eval_every steps
  samples/step_XXXXXXX.png          -- scatter plot (data vs generated)
  fr_t_grid.npy / fr_speed.npy      -- FR speed profile estimated at phase2_start
  loss.csv                          -- step, loss_ema, phase
"""
import copy, csv, math, os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.models import MLP2D
from datasets.datasets import sample_2d
from speed.speed import UniformSampler, make_cdf_sampler
from utils.helpers import ema_update, save_ckpt, run_2d_eval
from path.path import LinearPath

# Import exact divergence helpers from the existing 2D script
from scripts.compute_reparam_div_2d import (
    exact_div,
    sample_interpolant_classical,
    sample_interpolant_batch_classical,
)

T_MIN = 0.0
T_MAX = 1.0


def estimate_fr_speed(model, dataset, t_grid, B, n_epochs, device):
    """v_t^FR = sqrt(E[(div u_t)^2]) via exact divergence (valid for dim=2)."""
    model.eval()
    sq_all = np.zeros((n_epochs, len(t_grid)))

    for ep in range(n_epochs):
        for i, t_val in enumerate(tqdm(
                t_grid, desc=f'  FR speed epoch {ep+1}/{n_epochs}', ncols=80, leave=False)):

            t_val_c = float(np.clip(t_val, T_MIN + 1e-3, T_MAX - 1e-3))

            xt, _ = sample_interpolant_classical(dataset, B, t_val_c, device)

            t_tensor = torch.full((xt.shape[0],), t_val_c,
                      device=device, dtype=torch.float32)

            u = model(t_tensor, xt)  # shape: (B, d)

            # divergence (should be per-sample or already batchwise scalar per sample)
            div_est = exact_div(model, t_val_c, xt, device)  # ideally shape: (B,)

            # ---- FIX 1: norm must be per-sample, not global ----
            scaled_norm = t_val_c / (1 - t_val_c) * torch.linalg.norm(u, dim=1)  # (B,)

            # ---- FIX 2: dot product must be per-sample ----
            # xt.T @ u is WRONG for batch data
            scaled_dot_product = (1.0 / t_val_c) * torch.sum(xt * u, dim=1)  # (B,)

            # combine
            err = div_est - scaled_norm - scaled_dot_product  # (B,)

            sq_all[ep, i] = torch.mean(err ** 2).detach().cpu().numpy()

    model.train()
    return np.sqrt(np.maximum(np.mean(sq_all, axis=0), 0.0))



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='8gaussians',
                        choices=['8gaussians', '40gaussians', 'moons', 'circles', 'checkerboard'])
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--total_steps',  type=int,   default=200_001)
    parser.add_argument('--phase2_start', type=int,   default=50_000,
                        help='Step at which to switch from uniform to FR-weighted sampling')
    parser.add_argument('--batch_size',   type=int,   default=256)
    parser.add_argument('--lr',           type=float, default=2e-4)
    parser.add_argument('--ema_decay',    type=float, default=0.9999)
    parser.add_argument('--hidden',       type=int,   default=256)
    parser.add_argument('--depth',        type=int,   default=4)
    parser.add_argument('--speed_n_t',    type=int,   default=100)
    parser.add_argument('--speed_B',      type=int,   default=2_000)
    parser.add_argument('--speed_epochs', type=int,   default=5)
    parser.add_argument('--speed_smooth', type=float, default=0.05)
    parser.add_argument('--eval_every',   type=int,   default=50_000)
    parser.add_argument('--seed',         type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(f'{args.out_dir}/checkpoints', exist_ok=True)
    os.makedirs(f'{args.out_dir}/samples', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    print(f'Dataset: {args.dataset}  total_steps={args.total_steps}  phase2_start={args.phase2_start}')

    path = LinearPath()

    net     = MLP2D(dim=2, hidden=args.hidden, depth=args.depth).to(device)
    ema_net = copy.deepcopy(net)
    optim   = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched   = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda s: 1.0)
    print(f'Params : {sum(p.numel() for p in net.parameters()):,}')

    uniform_sampler = UniformSampler(T_MAX)
    speed_sampler   = None
    phase           = 0
    loss_ema        = None

    csv_path = os.path.join(args.out_dir, 'loss.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'loss_ema', 'phase'])

    print(f'\nTraining {args.total_steps} steps...')
    for step in tqdm(range(1, args.total_steps + 1), ncols=80, desc='train'):

        # ── Phase transition: switch to FR-weighted at phase2_start ────────────
        if step == args.phase2_start and phase == 0:
            tqdm.write(f'\n[step {step}] Estimating FR speed (exact divergence)...')
            t_grid = np.linspace(T_MIN + 0.01, T_MAX - 0.01, args.speed_n_t)
            v_t = estimate_fr_speed(ema_net, args.dataset, t_grid,
                                    args.speed_B, args.speed_epochs, device)
            ratio = v_t.max() / max(v_t.min(), 1e-10)
            tqdm.write(f'  v_t^FR: [{v_t.min():.4f}, {v_t.max():.4f}]  ratio={ratio:.1f}x')
            np.save(os.path.join(args.out_dir, 'fr_t_grid.npy'), t_grid)
            np.save(os.path.join(args.out_dir, 'fr_speed.npy'),  v_t)
            speed_sampler = make_cdf_sampler(t_grid, v_t, T_MAX, smooth_sigma=args.speed_smooth)
            phase = 2
            tqdm.write(f'[step {step}] Switched to FR-weighted sampling (phase 2)')

        # ── Sample t ───────────────────────────────────────────────────────────
        if phase == 0 or speed_sampler is None:
            t_samp = uniform_sampler.sample(args.batch_size, device)
        else:
            t_samp = speed_sampler.sample(args.batch_size, device)

        # ── Training step ──────────────────────────────────────────────────────
        xt, ut = sample_interpolant_batch_classical(args.dataset, args.batch_size, t_samp, device)
        optim.zero_grad()
        loss = ((net(t_samp, xt) - ut) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optim.step()
        sched.step()
        ema_update(net, ema_net, args.ema_decay)
        loss_ema = loss.item() if loss_ema is None else 0.99 * loss_ema + 0.01 * loss.item()

        if step % 500 == 0:
            csv_writer.writerow([step, f'{loss_ema:.6f}', phase])

        # ── Eval + checkpoint ──────────────────────────────────────────────────
        if step % args.eval_every == 0 or step == args.total_steps:
            ckpt_path = os.path.join(args.out_dir, 'checkpoints', f'ckpt_step_{step:07d}.pt')
            curriculum_state = {'phase': phase, 'phase2_start': args.phase2_start}
            save_ckpt(ckpt_path, net, ema_net, optim, sched, step, loss_ema, curriculum_state)
            run_2d_eval(step, ema_net, path, args.dataset, args.out_dir, n_samples=2000, device=device)
            tqdm.write(f'[step {step}] loss_ema={loss_ema:.4f}  phase={phase}  saved ckpt')

    csv_file.close()
    print(f'\nDone. Final loss_ema={loss_ema:.4f}  phase={phase}')
    print(f'Outputs: {args.out_dir}/')


if __name__ == '__main__':
    main()
