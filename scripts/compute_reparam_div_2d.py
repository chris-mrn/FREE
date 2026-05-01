"""
Compute energy repartition E[(div u_s^theta(X_{alpha(s)}))^2] for 2D flow-matching experiments.

Supports two modes:
  - self:  Self-FM interpolant (data↔data) used previously.
  - classical: Classical FM interpolant (noise→data).

Unlike the CIFAR-10 version, this script:
  - Trains a 2D MLP with arc-length curriculum inline (fast, ~minutes)
  - Uses EXACT divergence (2 backward passes per point, no Hutchinson needed for dim=2)

Workflow:
  1. Train MLP on chosen interpolant with arc-length curriculum
  2. At --curriculum_start, estimate FR speed via exact divergence
  3. Build arc-length reparametrisation alpha : [0,1] -> [T_MIN, T_MAX]
  4. Sweep D_raw(t) = E[(div u_t^theta(X_t))^2] and D(s) = E[(div u_s^theta(X_{alpha(s)}))^2]
  5. Compare CV(D_raw) vs CV(D) -- is energy equalised?

Usage:
    python scripts/compute_reparam_div_2d.py --mode classical --dataset 8gaussians --out_dir outputs/fm_2d/8gaussians
"""
import copy, math, os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.datasets import sample_2d
from models   import MLP2D
from speed    import UniformSampler, make_cdf_sampler, BlendedSampler
from utils    import ema_update, cosine_blend

T_MIN = 0.2
T_MAX = 0.8


# ── Self-FM interpolant ────────────────────────────────────────────────────────

def sample_interpolant(dataset, B, t_val, device):
    """
    Sample (X_t, u_t) from the 2D self-FM interpolant at a fixed scalar t_val.
    X_t = (1-t)*X1 + t*X1_tilde + sqrt(t(1-t))*eps
    u_t = X1_tilde - X1 + dsigma_dt * eps
    """
    x1       = sample_2d(dataset, B).to(device)
    x1_tilde = sample_2d(dataset, B).to(device)
    sigma_t  = math.sqrt(float(t_val) * (1.0 - float(t_val)))
    dsigma   = (1.0 - 2.0 * float(t_val)) / (2.0 * sigma_t + 1e-8)
    eps      = torch.randn_like(x1)
    xt       = (1.0 - t_val) * x1 + t_val * x1_tilde + sigma_t * eps
    ut       = x1_tilde - x1 + dsigma * eps
    return xt.detach(), ut.detach()


def sample_interpolant_batch(dataset, B, t_samp, device):
    """
    Vectorised version: t_samp is (B,), returns (xt, ut) of shape (B, 2).
    Used in the training loop where each sample has a different t.
    """
    x1       = sample_2d(dataset, B).to(device)
    x1_tilde = sample_2d(dataset, B).to(device)
    eps      = torch.randn_like(x1)
    t        = t_samp.clamp(T_MIN + 1e-4, T_MAX - 1e-4)
    sigma_t  = torch.sqrt(t * (1.0 - t)).unsqueeze(-1)            # (B, 1)
    dsigma   = ((1.0 - 2.0 * t) / (2.0 * sigma_t.squeeze(-1) + 1e-8)).unsqueeze(-1)
    tb       = t.unsqueeze(-1)
    xt       = (1.0 - tb) * x1 + tb * x1_tilde + sigma_t * eps
    ut       = x1_tilde - x1 + dsigma * eps
    return xt.detach(), ut.detach()


# ── Classical FM interpolant (noise -> data) ───────────────────────────────────

def sample_interpolant_classical(dataset, B, t_val, device):
    """
    Sample (X_t, u_t) from the classical FM interpolant (noise -> data) at scalar t_val.
    x0 ~ N(0,I), x1 ~ p_data
    X_t = (1-t)*x0 + t*x1 + sqrt(t(1-t))*eps
    u_t = x1 - x0 + dsigma_dt * eps
    """
    x0 = torch.randn(B, 2, device=device)
    x1 = sample_2d(dataset, B).to(device)
    sigma_t  = math.sqrt(float(t_val) * (1.0 - float(t_val)))
    dsigma   = (1.0 - 2.0 * float(t_val)) / (2.0 * sigma_t + 1e-8)
    eps      = torch.randn_like(x1)
    xt       = (1.0 - t_val) * x0 + t_val * x1 + sigma_t * eps
    ut       = x1 - x0 + dsigma * eps
    return xt.detach(), ut.detach()


def sample_interpolant_batch_classical(dataset, B, t_samp, device):
    """
    Vectorised version of the classical FM interpolant.
    """
    x1 = sample_2d(dataset, B).to(device)
    x0 = torch.randn_like(x1)
    eps = torch.randn_like(x1)
    t = t_samp.clamp(T_MIN + 1e-4, T_MAX - 1e-4)
    sigma_t = torch.sqrt(t * (1.0 - t)).unsqueeze(-1)
    dsigma = ((1.0 - 2.0 * t) / (2.0 * sigma_t.squeeze(-1) + 1e-8)).unsqueeze(-1)
    tb = t.unsqueeze(-1)
    xt = (1.0 - tb) * x0 + tb * x1 + sigma_t * eps
    ut = x1 - x0 + dsigma * eps
    return xt.detach(), ut.detach()


# ── Exact divergence for dim=2 ─────────────────────────────────────────────────

def exact_div(model, t_val, xt, device):
    """
    Exact E[(div u_t^theta(X_t))^2] for a 2D model.
    div u = du1/dx1 + du2/dx2, computed with 2 backward passes.
    No Hutchinson approximation needed since dim=2.
    """
    B = xt.shape[0]
    t = torch.full((B,), float(t_val), device=device)
    x = xt.detach().requires_grad_(True)
    u = model(t, x)                                                     # (B, 2)
    g0 = torch.autograd.grad(u[:, 0].sum(), x,
                             retain_graph=True,  create_graph=False)[0]  # (B, 2)
    g1 = torch.autograd.grad(u[:, 1].sum(), x,
                             retain_graph=False, create_graph=False)[0]  # (B, 2)
    div = g0[:, 0] + g1[:, 1]                                           # (B,)
    return div


# ── FR speed estimation ────────────────────────────────────────────────────────

def estimate_fr_speed(model, dataset, t_grid, B, n_epochs, device):
    """v_t^FR = sqrt(E[(div u_t^theta)^2]) estimated on t_grid (exact for dim=2)."""
    model.eval()
    sq_all = np.zeros((n_epochs, len(t_grid)))
    for ep in range(n_epochs):
        for i, t_val in enumerate(tqdm(
                t_grid, desc=f'  FR speed ep{ep+1}/{n_epochs}', ncols=80, leave=False)):
            t_val = float(np.clip(t_val, T_MIN + 1e-3, T_MAX - 1e-3))
            xt, _ = sample_interpolant(dataset, B, t_val, device)
            sq_all[ep, i] = exact_div_sq(model, t_val, xt, device)
    model.train()
    return np.sqrt(np.maximum(np.median(sq_all, axis=0), 0.0))


# ── Arc-length reparametrisation ───────────────────────────────────────────────

def build_alpha(t_grid, v_t):
    """CDF^{-1} of p(t) ∝ 1/v_t. Returns alpha: array -> array."""
    w   = 1.0 / np.maximum(v_t, 1e-10)
    dt  = np.diff(t_grid)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (w[:-1] + w[1:]) * dt)])
    cum /= cum[-1]
    t_ext = np.concatenate([[T_MIN], t_grid, [T_MAX]])
    c_ext = np.concatenate([[0.0],   cum,    [1.0]])
    return lambda s: np.interp(s, c_ext, t_ext).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='8gaussians',
                        choices=['8gaussians', '40gaussians',
                                 'moons', 'circles', 'checkerboard'])
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--mode', choices=['self', 'classical'], default='self',
                        help='Interpolant mode: self (data\u21E6data) or classical (noise\u2192data)')

    # Training
    parser.add_argument('--total_steps',      type=int,   default=50_000)
    parser.add_argument('--batch_size',       type=int,   default=512)
    parser.add_argument('--lr',               type=float, default=1e-3)
    parser.add_argument('--ema_decay',        type=float, default=0.999)
    parser.add_argument('--hidden',           type=int,   default=256)
    parser.add_argument('--depth',            type=int,   default=4)

    # Curriculum
    parser.add_argument('--curriculum_start', type=int,   default=20_000)
    parser.add_argument('--curriculum_blend', type=int,   default=5_000)

    # Speed estimation
    parser.add_argument('--speed_n_t',        type=int,   default=100)
    parser.add_argument('--speed_B',          type=int,   default=2_000)
    parser.add_argument('--speed_epochs',     type=int,   default=5)

    # Energy sweep
    parser.add_argument('--div_n_t',          type=int,   default=200)
    parser.add_argument('--div_B',            type=int,   default=2_000)
    parser.add_argument('--div_epochs',       type=int,   default=5)

    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    if device.type == 'cuda':
        print(f'GPU    : {torch.cuda.get_device_name(0)}')
    print(f'Dataset: {args.dataset}  total_steps={args.total_steps}')
    print(f'Curriculum start={args.curriculum_start}  blend={args.curriculum_blend}')

    # Choose interpolant mode (default: self)
    if args.mode == 'classical':
        print('Mode  : classical FM (noise -> data)')
        # Rebind sampling functions to the classical variants
        sample_interpolant = sample_interpolant_classical
        sample_interpolant_batch = sample_interpolant_batch_classical
    else:
        print('Mode  : self FM (data \u2194 data)')
        # sample_interpolant / sample_interpolant_batch keep their default definitions

    # ── Model ─────────────────────────────────────────────────────────────────
    net     = MLP2D(dim=2, hidden=args.hidden, depth=args.depth).to(device)
    ema_net = copy.deepcopy(net)
    optim   = torch.optim.Adam(net.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in net.parameters())
    print(f'Params : {n_params:,}')

    uniform_sampler = UniformSampler(T_MAX)
    speed_sampler   = None
    phase, speed_step = 0, None
    t_grid_speed, v_t_speed = None, None
    loss_ema = None

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f'\nTraining {args.total_steps} steps...')
    for step in tqdm(range(1, args.total_steps + 1), ncols=80, desc='train'):

        # Curriculum: estimate speed and start blending
        if step == args.curriculum_start and phase == 0:
            tqdm.write(f'\n[step {step}] Estimating FR speed...')
            t_grid_speed = np.linspace(T_MIN + 0.01, T_MAX - 0.01, args.speed_n_t)
            v_t_speed    = estimate_fr_speed(
                ema_net, args.dataset, t_grid_speed,
                args.speed_B, args.speed_epochs, device)
            ratio = v_t_speed.max() / max(v_t_speed.min(), 1e-10)
            tqdm.write(f'  v_t^FR: [{v_t_speed.min():.4f}, {v_t_speed.max():.4f}]  '
                       f'ratio={ratio:.1f}x')
            speed_sampler = make_cdf_sampler(t_grid_speed, v_t_speed, T_MAX,
                                             smooth_sigma=0.05)
            np.save(os.path.join(args.out_dir, 'fr_t_grid.npy'), t_grid_speed)
            np.save(os.path.join(args.out_dir, 'fr_speed.npy'),  v_t_speed)
            phase, speed_step = 1, step

        # End of blend
        if phase == 1 and step >= speed_step + args.curriculum_blend:
            phase = 2
            tqdm.write(f'[step {step}] Phase 2: pure arc-length sampling')

        # Sample t
        if phase == 0 or speed_sampler is None:
            t_samp = uniform_sampler.sample(args.batch_size, device)
        elif phase == 1:
            mix    = cosine_blend(step, speed_step, args.curriculum_blend)
            t_samp = BlendedSampler(uniform_sampler, speed_sampler).sample(
                args.batch_size, device, mix)
        else:
            t_samp = speed_sampler.sample(args.batch_size, device)

        # Training step
        xt, ut = sample_interpolant_batch(args.dataset, args.batch_size, t_samp, device)
        optim.zero_grad()
        loss = ((net(t_samp, xt) - ut) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optim.step()
        ema_update(net, ema_net, args.ema_decay)
        loss_ema = loss.item() if loss_ema is None else 0.99 * loss_ema + 0.01 * loss.item()

    print(f'Training done. Final loss EMA: {loss_ema:.4f}  phase={phase}')
    torch.save({'net': net.state_dict(), 'ema': ema_net.state_dict()},
               os.path.join(args.out_dir, 'checkpoint_final.pt'))

    # If curriculum never triggered (total_steps < curriculum_start), estimate now
    if v_t_speed is None:
        print('Estimating FR speed post-training...')
        t_grid_speed = np.linspace(T_MIN + 0.01, T_MAX - 0.01, args.speed_n_t)
        v_t_speed    = estimate_fr_speed(
            ema_net, args.dataset, t_grid_speed,
            args.speed_B, args.speed_epochs, device)
        np.save(os.path.join(args.out_dir, 'fr_t_grid.npy'), t_grid_speed)
        np.save(os.path.join(args.out_dir, 'fr_speed.npy'),  v_t_speed)

    alpha    = build_alpha(t_grid_speed, v_t_speed)
    s_grid   = np.linspace(0.0, 1.0, args.div_n_t, dtype=np.float32)
    tau_grid = alpha(s_grid)
    t_unif   = np.linspace(T_MIN + 0.01, T_MAX - 0.01, args.div_n_t, dtype=np.float32)
    print(f'\nalpha: s∈[0,1] → tau∈[{tau_grid.min():.4f}, {tau_grid.max():.4f}]')

    # ── Energy repartition sweep ──────────────────────────────────────────────
    print(f'Energy sweep: n_t={args.div_n_t}, B={args.div_B}, epochs={args.div_epochs}')
    ema_net.eval()
    div_rep = np.zeros((args.div_epochs, args.div_n_t))
    div_uni = np.zeros((args.div_epochs, args.div_n_t))

    for ep in range(args.div_epochs):
        for i, (s_val, tau_val, t_val) in enumerate(tqdm(
                zip(s_grid, tau_grid, t_unif), total=args.div_n_t,
                desc=f'  div ep{ep+1}/{args.div_epochs}', ncols=80, leave=False)):

            # Reparametrised: sample X at tau=alpha(s), call model at s
            xt_rep, _ = sample_interpolant(args.dataset, args.div_B, float(tau_val), device)
            div_rep[ep, i] = exact_div_sq(ema_net, float(s_val), xt_rep, device)

            # Raw: sample and call at same uniform t
            xt_uni, _ = sample_interpolant(args.dataset, args.div_B, float(t_val), device)
            div_uni[ep, i] = exact_div_sq(ema_net, float(t_val), xt_uni, device)

    rep_mean, rep_std = div_rep.mean(0), div_rep.std(0)
    uni_mean, uni_std = div_uni.mean(0), div_uni.std(0)

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(os.path.join(args.out_dir, 'div_sq_s_grid.npy'),       s_grid)
    np.save(os.path.join(args.out_dir, 'div_sq_tau_grid.npy'),     tau_grid)
    np.save(os.path.join(args.out_dir, 'div_sq_reparam_mean.npy'), rep_mean)
    np.save(os.path.join(args.out_dir, 'div_sq_reparam_std.npy'),  rep_std)
    np.save(os.path.join(args.out_dir, 'div_sq_uniform_t.npy'),    t_unif)
    np.save(os.path.join(args.out_dir, 'div_sq_uniform_mean.npy'), uni_mean)
    np.save(os.path.join(args.out_dir, 'div_sq_uniform_std.npy'),  uni_std)
    print('Arrays saved.')

    # ── Summary ───────────────────────────────────────────────────────────────
    cv_raw = uni_mean.std() / uni_mean.mean()
    cv_rep = rep_mean.std() / rep_mean.mean()
    change = (1.0 - cv_rep / max(cv_raw, 1e-10)) * 100
    print(f'\n── Summary ({args.dataset}) ──')
    print(f'Raw (uniform t):')
    print(f'  mean={uni_mean.mean():.4f}  std={uni_mean.std():.4f}  CV={cv_raw:.4f}')
    print(f'Reparametrised (arc-length s):')
    print(f'  mean={rep_mean.mean():.4f}  std={rep_mean.std():.4f}  CV={cv_rep:.4f}')
    print(f'CV change: {cv_raw:.4f} → {cv_rep:.4f}  ({change:+.1f}%)')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(f'Energy repartition — 2D {args.mode}-FM  ({args.dataset},  '
                 f'{args.total_steps:,} steps)', fontsize=12)

    # (A) FR speed
    ax = axes[0]
    ax.plot(t_grid_speed, v_t_speed, color='steelblue', lw=1.5)
    ax.set_xlabel('$t$');  ax.set_ylabel('$v_t^{FR}$')
    ax.set_title('(A) FR speed profile')
    ax.grid(True, alpha=0.3)

    # (B) Raw D(t)
    ax = axes[1]
    ax.fill_between(t_unif, uni_mean - uni_std, uni_mean + uni_std,
                    alpha=0.2, color='steelblue')
    ax.plot(t_unif, uni_mean, color='steelblue', lw=1.5)
    ax.set_xlabel('$t$ (uniform)');  ax.set_ylabel(r'$E[(\mathrm{div}\,u_t^\theta)^2]$')
    ax.set_title(f'(B) Raw   CV={cv_raw:.3f}')
    ax.grid(True, alpha=0.3)

    # (C) Reparametrised D(s)
    ax = axes[2]
    ax.fill_between(s_grid, rep_mean - rep_std, rep_mean + rep_std,
                    alpha=0.2, color='tomato')
    ax.plot(s_grid, rep_mean, color='tomato', lw=1.5)
    ax.axhline(rep_mean.mean(), color='k', lw=1, ls='--',
               label=f'mean={rep_mean.mean():.3f}')
    ax.set_xlabel('$s$ (arc-length)');  ax.set_ylabel(r'$E[(\mathrm{div}\,u_s^\theta)^2]$')
    ax.set_title(f'(C) Reparametrised   CV={cv_rep:.3f}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (D) alpha(s)
    ax = axes[3]
    ax.plot(s_grid, tau_grid, color='seagreen', lw=1.5)
    ax.plot([0, 1], [T_MIN, T_MAX], 'k--', lw=1, alpha=0.5, label='identity')
    ax.set_xlabel('$s$');  ax.set_ylabel(r'$\tau = \alpha(s)$')
    ax.set_title(r'(D) Reparametrisation $\alpha$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(args.out_dir, 'div_sq_reparam_2d.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved → {out_png}')
    print('Done.')


if __name__ == '__main__':
    main()
