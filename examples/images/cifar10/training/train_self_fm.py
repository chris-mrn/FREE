"""
Self-interpolant Flow Matching on CIFAR-10.

Interpolant (stochastic):
    X_t = (1-t)*X1 + t*X1_tilde + sqrt(t*(1-t)) * eps,   eps ~ N(0,I)

where X1 and X1_tilde are two independent samples from p_data (CIFAR-10).
The source AND target are both data; there is no Gaussian noise source.

Conditional velocity (closed form):
    u_t = X1_tilde - X1 + dsigma_dt * eps
    dsigma_dt = (1 - 2t) / (2 * sqrt(t*(1-t)))

t is clipped to [T_MIN, T_MAX] to avoid the singularity in dsigma_dt.

Training schedule:
  Steps 0 … speed_step   : uniform  t ~ U(T_MIN, T_MAX)
  Step  speed_step        : compute FR speed from EMA, save plot
  Steps speed_step … +blend_steps: cosine blend U → p_FR
  Steps onwards           : pure p_FR,   p(t) ∝ 1/v_t^FR  (arc-length schedule)
"""
import os, sys, argparse, copy, time, math
sys.path.insert(0, '/nfs/ghome/live/cmarouani/FREE')

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm
from torchcfm.models.unet.unet import UNetModelWrapper

T_MIN = 0.01   # avoid singularity in dsigma_dt near 0 and 1
T_MAX = 0.99

# ── Utilities ──────────────────────────────────────────────────────────────────

def ema_update(source, target, decay):
    src = source.state_dict()
    tgt = target.state_dict()
    for k in src:
        tgt[k].data.mul_(decay).add_(src[k].data, alpha=1 - decay)


def infiniteloop(loader):
    while True:
        for x, _ in loader:
            yield x


def warmup_lr(step, warmup):
    return min(step + 1, warmup) / warmup


# ── Self-interpolant helpers ───────────────────────────────────────────────────

def self_interpolant(x1, x1_tilde, t):
    """
    x1, x1_tilde : (B, C, H, W)
    t            : (B,) in [T_MIN, T_MAX]
    Returns xt, ut  both (B, C, H, W)
    """
    t_b     = t.view(-1, 1, 1, 1)
    sigma_t = torch.sqrt(t_b * (1.0 - t_b))          # (B,1,1,1)
    eps     = torch.randn_like(x1)
    xt      = (1.0 - t_b) * x1 + t_b * x1_tilde + sigma_t * eps
    dsigma  = (1.0 - 2.0 * t_b) / (2.0 * sigma_t)   # (B,1,1,1)
    ut      = x1_tilde - x1 + dsigma * eps
    return xt, ut


# ── FR speed estimation ────────────────────────────────────────────────────────

def _fr_speed_one_epoch(model, x_ref, t_grid, n_hutch, chunk_size, B_per_t, device):
    """
    One sweep of Hutchinson FR-speed estimation.
    Marginal X_t sampled from the self-interpolant with TWO independent
    draws from x_ref (no Gaussian source).
    Returns array (n_t,) of v_t^FR estimates.
    """
    n_t   = len(t_grid)
    N_ref = len(x_ref)
    speeds_sq = np.zeros(n_t, dtype=np.float64)

    for start in range(0, n_t, chunk_size):
        end     = min(start + chunk_size, n_t)
        t_chunk = t_grid[start:end]
        C       = len(t_chunk)
        total_B = C * B_per_t

        # Two independent data samples as X1 and X1_tilde
        idx0 = torch.randint(0, N_ref, (total_B,), device=device)
        idx1 = torch.randint(0, N_ref, (total_B,), device=device)
        x_src  = x_ref[idx0]   # X_1
        x_tgt  = x_ref[idx1]   # X̃_1
        eps    = torch.randn_like(x_src)

        t_exp  = torch.tensor(
            np.repeat(t_chunk, B_per_t), dtype=torch.float32, device=device
        )
        t_b    = t_exp.view(-1, 1, 1, 1)
        sigma_t = torch.sqrt(t_b * (1.0 - t_b)).clamp(min=1e-5)
        xt     = ((1.0 - t_b) * x_src + t_b * x_tgt + sigma_t * eps
                  ).detach().requires_grad_(True)

        v = model(t_exp, xt)   # (total_B, C, H, W)

        divs = []
        for probe in range(n_hutch):
            z = torch.randint(0, 2, xt.shape, device=device).float() * 2 - 1
            retain = (probe < n_hutch - 1)
            grad = torch.autograd.grad(
                (v * z).sum(), xt,
                create_graph=False, retain_graph=retain
            )[0]
            divs.append((grad * z).sum(dim=(1, 2, 3)).detach())

        div_mean  = torch.stack(divs, 0).mean(0)
        div2      = (div_mean ** 2).reshape(C, B_per_t).mean(1)
        speeds_sq[start:end] = div2.cpu().numpy()
        del xt, v, divs, div_mean, div2

    return np.sqrt(speeds_sq)


def compute_fr_speed_curve(model, x_ref, t_grid, n_epochs=5, n_hutch=5,
                            chunk_size=128, B_per_t=2, device='cuda', smooth_sigma=3.0):
    model.eval()
    all_speeds = []
    print(f'  [FR speed] n_epochs={n_epochs}, n_hutch={n_hutch}, '
          f'B_per_t={B_per_t}, chunk_size={chunk_size}, n_t={len(t_grid)}', flush=True)
    t0 = time.time()

    for epoch in range(n_epochs):
        speeds = _fr_speed_one_epoch(
            model, x_ref, t_grid, n_hutch, chunk_size, B_per_t, device
        )
        all_speeds.append(speeds)
        print(f'  [FR speed] epoch {epoch+1}/{n_epochs}  '
              f'range=[{speeds.min():.3f}, {speeds.max():.3f}]', flush=True)

    fr_speeds_raw = np.median(np.stack(all_speeds, axis=0), axis=0)
    fr_speeds     = gaussian_filter1d(fr_speeds_raw, sigma=smooth_sigma)
    fr_speeds     = np.clip(fr_speeds, 1e-6, None)

    elapsed = time.time() - t0
    print(f'  [FR speed] done in {elapsed:.1f}s  '
          f'smoothed range=[{fr_speeds.min():.3f}, {fr_speeds.max():.3f}]', flush=True)
    model.train()
    return fr_speeds_raw, fr_speeds


# ── Inverse-CDF sampler ────────────────────────────────────────────────────────

class InverseCDFSampler:
    def __init__(self, t_grid, weights, name='sampler'):
        self.name = name
        dt         = np.diff(t_grid)
        increments = 0.5 * (weights[:-1] + weights[1:]) * dt
        cdf        = np.concatenate([[0.0], np.cumsum(increments)])
        cdf       /= cdf[-1]
        self._t   = t_grid.astype(np.float64)
        self._cdf = cdf.astype(np.float64)

    def sample(self, n, device):
        u      = np.random.rand(n)
        t_vals = np.interp(u, self._cdf, self._t).astype(np.float32)
        return torch.tensor(t_vals, device=device)

    @staticmethod
    def from_fr_speed(t_grid, fr_speeds, name='fr_sampler'):
        """Arc-length schedule: p(t) ∝ 1/v_t^FR."""
        weights = 1.0 / np.clip(fr_speeds, 1e-6, None)
        return InverseCDFSampler(t_grid, weights, name=name)


class UniformSampler:
    def __init__(self, t_min=T_MIN, t_max=T_MAX):
        self.t_min = t_min
        self.t_max = t_max

    def sample(self, n, device):
        return torch.rand(n, device=device) * (self.t_max - self.t_min) + self.t_min


# ── Cosine blend ───────────────────────────────────────────────────────────────

def cosine_mix(step, start, duration):
    progress = min(1.0, max(0.0, (step - start) / duration))
    return 0.5 * (1.0 - math.cos(math.pi * progress))


def sample_t_blended(n, device, sampler_a, sampler_b, mix):
    mask = torch.rand(n) < mix
    n_b  = int(mask.sum().item())
    n_a  = n - n_b
    parts = []
    if n_a > 0:
        parts.append(sampler_a.sample(n_a, device))
    if n_b > 0:
        parts.append(sampler_b.sample(n_b, device))
    t = torch.cat(parts)
    return t[torch.randperm(n, device=device)]


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(path, net, ema_net, optim, sched, step, loss_ema, state):
    torch.save({
        'net':      net.state_dict(),
        'ema':      ema_net.state_dict(),
        'optim':    optim.state_dict(),
        'sched':    sched.state_dict(),
        'step':     step,
        'loss_ema': loss_ema,
        'state':    state,
    }, path)


def find_last_checkpoint(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None
    files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.pt'))
    return os.path.join(ckpt_dir, files[-1]) if files else None


# ── Speed plotting ─────────────────────────────────────────────────────────────

def save_speed_plot(out_dir, step, t_grid, fr_raw, fr_smooth):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        w = 1.0 / np.clip(fr_smooth, 1e-6, None)
        w /= w.mean()

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle(f'Self-FM (CIFAR-10) — FR speed at step {step}', fontsize=12)

        axes[0].plot(t_grid, fr_raw,    lw=1.5, alpha=0.5, color='steelblue', label='raw')
        axes[0].plot(t_grid, fr_smooth, lw=2,   color='steelblue',            label='smoothed')
        axes[0].set(xlabel='t', ylabel='v_t^FR', title='Fisher-Rao Speed')
        axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

        axes[1].plot(t_grid, w, lw=2, color='darkorange')
        axes[1].set(xlabel='t', ylabel='1/v_t (normalised)', title='Arc-length weight p(t) ∝ 1/v_t')
        axes[1].grid(True, alpha=0.3)

        # CDF of the arc-length schedule (shows where training focuses)
        dt  = np.diff(t_grid)
        inc = 0.5 * (w[:-1] + w[1:]) * dt
        cdf = np.concatenate([[0.0], np.cumsum(inc)])
        cdf /= cdf[-1]
        axes[2].plot(t_grid, cdf, lw=2, color='seagreen')
        axes[2].plot(t_grid, (t_grid - t_grid[0]) / (t_grid[-1] - t_grid[0]),
                     'k--', lw=1, alpha=0.4, label='uniform')
        axes[2].set(xlabel='t', ylabel='CDF', title='Sampling CDF (arc-length)')
        axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        fname = os.path.join(out_dir, f'fr_speed_step{step}.png')
        plt.savefig(fname, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f'  [plot] Saved: {fname}', flush=True)
    except Exception as e:
        print(f'  [plot] Failed: {e}', flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument('--total_steps',  type=int,   default=200_001)
    parser.add_argument('--batch_size',   type=int,   default=128)
    parser.add_argument('--lr',           type=float, default=2e-4)
    parser.add_argument('--grad_clip',    type=float, default=1.0)
    parser.add_argument('--warmup',       type=int,   default=5_000)
    parser.add_argument('--ema_decay',    type=float, default=0.9999)
    parser.add_argument('--num_channel',  type=int,   default=128)
    parser.add_argument('--save_step',    type=int,   default=50_000)
    parser.add_argument('--num_workers',  type=int,   default=4)
    parser.add_argument('--data_dir',     type=str,   default='/tmp/cifar10_self_fm')
    parser.add_argument('--out_dir',      type=str,   default='/tmp/self_fm_out')
    parser.add_argument('--resume',       type=str,   default='auto')
    # Curriculum
    parser.add_argument('--speed_step',   type=int,   default=100_000,
                        help='Step at which FR speed is computed and curriculum begins')
    parser.add_argument('--blend_steps',  type=int,   default=25_000,
                        help='Steps to cosine-blend from uniform to arc-length schedule')
    # FR speed estimation
    parser.add_argument('--fr_n_t',       type=int,   default=1_000)
    parser.add_argument('--fr_n_epochs',  type=int,   default=5)
    parser.add_argument('--fr_n_hutch',   type=int,   default=5)
    parser.add_argument('--fr_B_per_t',   type=int,   default=2)
    parser.add_argument('--fr_chunk',     type=int,   default=128)
    parser.add_argument('--fr_smooth',    type=float, default=3.0)
    parser.add_argument('--fr_n_ref',     type=int,   default=2_000)
    # Misc
    parser.add_argument('--timing_only',  action='store_true')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU:    {torch.cuda.get_device_name(0)}')
    print(f'Config: steps={args.total_steps}, bs={args.batch_size}, lr={args.lr}')
    print(f'Interpolant: X_t=(1-t)*X1 + t*X1_tilde + sqrt(t(1-t))*eps, '
          f't in [{T_MIN}, {T_MAX}]')
    print(f'Curriculum: speed_step={args.speed_step}, blend={args.blend_steps}')

    # ── Dataset ───────────────────────────────────────────────────────────────
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_ref = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = datasets.CIFAR10(root=args.data_dir, train=True,
                          download=True, transform=transform_train)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True)
    looper = infiniteloop(loader)

    ds_ref = datasets.CIFAR10(root=args.data_dir, train=True,
                               download=False, transform=transform_ref)
    ref_loader = torch.utils.data.DataLoader(
        ds_ref, batch_size=args.fr_n_ref, shuffle=True)
    x_ref, _ = next(iter(ref_loader))
    x_ref = x_ref.to(device)
    print(f'Reference images: {x_ref.shape}')

    # ── Model ──────────────────────────────────────────────────────────────────
    net = UNetModelWrapper(
        dim=(3, 32, 32), num_res_blocks=2, num_channels=args.num_channel,
        channel_mult=[1, 2, 2, 2], num_heads=4, num_head_channels=64,
        attention_resolutions='16', dropout=0.1,
    ).to(device)
    ema_net = copy.deepcopy(net)
    print(f'Model params: {sum(p.numel() for p in net.parameters())/1e6:.2f}M')

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: warmup_lr(s, args.warmup))

    # ── Curriculum state ───────────────────────────────────────────────────────
    state = {
        'phase':    0,      # 0=uniform, 1=blending, 2=pure arc-length
        'fr_t':     None,
        'fr_speed': None,
    }
    sampler_uniform = UniformSampler(T_MIN, T_MAX)
    sampler_fr      = None

    # ── Resume ─────────────────────────────────────────────────────────────────
    start_step = 1
    loss_ema   = None

    resume_path = None
    if args.resume == 'auto':
        resume_path = find_last_checkpoint(ckpt_dir)
    elif args.resume and os.path.isfile(args.resume):
        resume_path = args.resume

    if resume_path:
        print(f'Resuming from: {resume_path}')
        ck = torch.load(resume_path, map_location=device)
        net.load_state_dict(ck['net'])
        ema_net.load_state_dict(ck['ema'])
        optim.load_state_dict(ck['optim'])
        sched.load_state_dict(ck['sched'])
        start_step = ck['step'] + 1
        loss_ema   = ck.get('loss_ema', None)
        if 'state' in ck:
            state = ck['state']
            if state['fr_speed'] is not None:
                t_g = np.array(state['fr_t'])
                v_g = np.array(state['fr_speed'])
                sampler_fr = InverseCDFSampler.from_fr_speed(t_g, v_g)
        print(f'Resumed at step={start_step}, phase={state["phase"]}')

    # ── Training loop ──────────────────────────────────────────────────────────
    t_grid_fr = np.linspace(T_MIN + 0.01, T_MAX - 0.01, args.fr_n_t)

    pbar = tqdm(range(start_step, args.total_steps + 1), dynamic_ncols=True,
                initial=start_step - 1, total=args.total_steps)
    t_train_start = time.time()

    for step in pbar:

        # ── Curriculum event ──────────────────────────────────────────────────
        if step == args.speed_step and state['phase'] == 0:
            print(f'\n[Curriculum] Computing FR speed at step {step}...', flush=True)
            t0_fr = time.time()
            fr_raw, fr_smooth = compute_fr_speed_curve(
                ema_net, x_ref, t_grid_fr,
                n_epochs=args.fr_n_epochs, n_hutch=args.fr_n_hutch,
                chunk_size=args.fr_chunk, B_per_t=args.fr_B_per_t,
                device=device, smooth_sigma=args.fr_smooth,
            )
            print(f'[Curriculum] FR speed in {time.time()-t0_fr:.1f}s', flush=True)
            save_speed_plot(args.out_dir, step, t_grid_fr, fr_raw, fr_smooth)
            np.save(os.path.join(args.out_dir, f'fr_t_grid_step{step}.npy'), t_grid_fr)
            np.save(os.path.join(args.out_dir, f'fr_speed_raw_step{step}.npy'), fr_raw)
            np.save(os.path.join(args.out_dir, f'fr_speed_step{step}.npy'), fr_smooth)
            sampler_fr = InverseCDFSampler.from_fr_speed(t_grid_fr, fr_smooth)
            state.update({
                'phase':    1,
                'fr_t':     t_grid_fr.tolist(),
                'fr_speed': fr_smooth.tolist(),
            })
            print(f'[Curriculum] Phase 0→1: blending Uniform → arc-length '
                  f'over {args.blend_steps} steps', flush=True)

        if (state['phase'] == 1
                and step >= args.speed_step + args.blend_steps):
            state['phase'] = 2
            print(f'[Curriculum] Phase 2: pure arc-length from step {step}', flush=True)

        # ── Sample t ──────────────────────────────────────────────────────────
        phase = state['phase']
        B     = args.batch_size

        if phase == 0:
            t = sampler_uniform.sample(B, device)
        elif phase == 1:
            mix = cosine_mix(step, args.speed_step, args.blend_steps)
            t   = sample_t_blended(B, device, sampler_uniform, sampler_fr, mix)
        else:
            t = sampler_fr.sample(B, device)

        # ── Self-interpolant training step ────────────────────────────────────
        x1       = next(looper).to(device)
        x1_tilde = next(looper).to(device)   # independent replica from data

        xt, ut = self_interpolant(x1, x1_tilde, t)

        optim.zero_grad()
        vt   = net(t, xt)
        loss = ((vt - ut) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        optim.step()
        sched.step()
        ema_update(net, ema_net, args.ema_decay)

        loss_val = loss.item()
        loss_ema = loss_val if loss_ema is None else 0.99 * loss_ema + 0.01 * loss_val
        pbar.set_postfix(loss=f'{loss_ema:.4f}', phase=phase)

        # ── Timing estimate ───────────────────────────────────────────────────
        if step == start_step + 199:
            elapsed = time.time() - t_train_start
            rate    = 200 / elapsed
            print(f'\n[Timing] 200 steps in {elapsed:.1f}s '
                  f'({rate:.1f} steps/s, {rate*60:.0f} steps/min)')
            print(f'[Timing] Estimated training time: '
                  f'{args.total_steps / rate / 3600:.1f}h')
            if args.timing_only:
                print('--timing_only: exiting.')
                return

        # ── Checkpoint ────────────────────────────────────────────────────────
        if args.save_step > 0 and step % args.save_step == 0:
            ckpt_path = os.path.join(ckpt_dir, f'ema_step_{step:07d}.pt')
            save_checkpoint(ckpt_path, net, ema_net, optim, sched,
                            step, loss_ema, state)
            print(f'\n[ckpt] Saved: {ckpt_path}', flush=True)

    total = time.time() - t_train_start
    print(f'\nTraining done in {total/3600:.2f}h ({total/60:.1f}min)')
    ckpt_path = os.path.join(ckpt_dir, f'ema_step_{args.total_steps:07d}_final.pt')
    save_checkpoint(ckpt_path, net, ema_net, optim, sched,
                    args.total_steps, loss_ema, state)
    print(f'[ckpt] Final saved: {ckpt_path}')


if __name__ == '__main__':
    main()
