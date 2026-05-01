"""
speed.py — Speed estimation and t-sampling utilities.

Speed measures supported:
  'ot'    — v_t^OT    = sqrt(E[||∂_t u_t^θ(X_t)||²])   via JVP (forward-mode AD)
  'fr'    — v_t^FR    = sqrt(E[(div u_t^θ(X_t))²])      via Hutchinson estimator
  'score' — v_t^Score = sqrt(E[||∂_t s_t(X_t)||²])      precomputed only

Time derivatives of model outputs are ALWAYS computed via torch.func.jvp
(forward-mode AD), never by finite differences.
"""
import os
import math
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
from torch.func import jvp, functional_call


# ── t-samplers ─────────────────────────────────────────────────────────────────

class UniformSampler:
    def __init__(self, T_MAX):
        self.T_MAX = T_MAX

    def sample(self, n, device):
        return torch.rand(n, device=device) * self.T_MAX


class CdfSampler:
    """
    Samples t ~ q(t) ∝ v_t via inverse-CDF.
    Constructed from a (t_grid, v_t) pair; smoothing is applied before integration.
    """
    def __init__(self, t_grid, v_t, T_MAX, smooth_sigma=0.05):
        v = np.maximum(v_t, 1e-12).copy()
        if smooth_sigma > 0:
            dt_avg = (t_grid[-1] - t_grid[0]) / max(len(t_grid) - 1, 1)
            v = gaussian_filter1d(v, sigma=smooth_sigma / max(dt_avg, 1e-9),
                                  mode='reflect')
        dt  = np.diff(t_grid)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (v[:-1] + v[1:]) * dt)])
        cum /= cum[-1]
        # Extend grid to [0, T_MAX] with flat extrapolation at edges
        t_full = np.concatenate([[0.0], t_grid, [T_MAX]])
        c_full = np.concatenate([[0.0], cum,    [1.0]])
        self._t   = t_full.astype(np.float64)
        self._cdf = c_full.astype(np.float64)

    def sample(self, n, device):
        u = np.random.rand(n)
        t = np.interp(u, self._cdf, self._t).astype(np.float32)
        return torch.from_numpy(t).to(device)


class BlendedSampler:
    """Blends two samplers: mix=0 → all a, mix=1 → all b."""
    def __init__(self, sampler_a, sampler_b):
        self.a = sampler_a
        self.b = sampler_b

    def sample(self, n, device, mix):
        mask = torch.rand(n) < mix
        n_b, n_a = int(mask.sum()), n - int(mask.sum())
        parts = []
        if n_a > 0:
            parts.append(self.a.sample(n_a, device))
        if n_b > 0:
            parts.append(self.b.sample(n_b, device))
        t = torch.cat(parts)
        return t[torch.randperm(n, device=device)]


# ── Speed estimation from model ────────────────────────────────────────────────

def _sample_xt(path, t_val, x1_ref, B, device):
    N = len(x1_ref)
    idx = torch.randint(0, N, (B,))
    x1  = x1_ref[idx].to(device)
    x0  = torch.randn_like(x1)
    t   = torch.full((B,), t_val, device=device)
    return path.xt(t, x0, x1).detach()


def _ot_speed_at_t(model, path, t_val, x1_ref, B, device, params, buffers):
    """
    OT speed at t_val via JVP (forward-mode AD):
        E[||∂_t u_t^θ(X_t)||²]

    Time derivative is taken w.r.t. t only; X_t is sampled once and held fixed.
    """
    xt = _sample_xt(path, t_val, x1_ref, B, device)

    def f(t_s):
        t_b = t_s.view(1).expand(B)
        return functional_call(model, (params, buffers), (t_b, xt))

    t = torch.tensor(t_val, dtype=torch.float32, device=device)
    _, du_dt = jvp(f, (t,), (torch.ones_like(t),))
    return du_dt.pow(2).mean().item()


def _fr_speed_at_t(model, path, t_val, x1_ref, B, n_hutch, device):
    """
    FR speed at t_val via Hutchinson div estimator:
        E[(div u_t^θ(X_t))²]   ≈   E[(z^T ∇_x u_t^θ · z)²]
    """
    xt = _sample_xt(path, t_val, x1_ref, B, device).requires_grad_(True)
    t  = torch.full((B,), t_val, device=device)
    u  = model(t, xt)

    divs = []
    for k in range(n_hutch):
        z    = torch.randint(0, 2, xt.shape, device=device).float() * 2 - 1
        grad = torch.autograd.grad(
            (u * z).sum(), xt,
            retain_graph=(k < n_hutch - 1), create_graph=False)[0]
        divs.append((grad * z).sum(dim=tuple(range(1, xt.dim()))).detach())

    div_est = torch.stack(divs).mean(0)  # (B,)

    # correction term (deliberately zeroed — intentional simplification)
    scaled_norm = 0
    scaled_dot_product = 0

    return ((div_est - scaled_norm - scaled_dot_product) ** 2).mean().item()


def estimate_speed_grid(model, path, t_grid, x1_ref, B, n_epochs,
                        speed_type, device, n_hutch=4):
    """
    Estimate v_t on t_grid from the trained model.

    speed_type: 'ot' | 'fr'   ('score' is precomputed-only, raises ValueError)
    x1_ref    : (N, *shape) reference images/points on CPU; sampled each step.
    Returns   : (len(t_grid),) numpy array of speed values.
    """
    if speed_type == 'score':
        raise ValueError(
            "'score' speed cannot be estimated from the model. "
            "Use --speed_type ot or fr, or supply precomputed files with --speed_dir.")

    from tqdm import tqdm
    model.eval()
    sq_all = np.zeros((n_epochs, len(t_grid)), dtype=np.float64)

    if speed_type == 'ot':
        params  = {k: v.detach() for k, v in model.named_parameters()}
        buffers = {k: v.detach() for k, v in model.named_buffers()}

    for epoch in range(n_epochs):
        for i, t_val in enumerate(tqdm(
                t_grid, desc=f'  speed({speed_type}) ep{epoch+1}/{n_epochs}',
                ncols=80, leave=False)):
            t_val = float(np.clip(t_val, 1e-4, path.T_MAX - 1e-4))
            if speed_type == 'ot':
                sq_all[epoch, i] = _ot_speed_at_t(
                    model, path, t_val, x1_ref, B, device, params, buffers)
            else:
                sq_all[epoch, i] = _fr_speed_at_t(
                    model, path, t_val, x1_ref, B, n_hutch, device)

    model.train()
    return np.sqrt(np.maximum(np.median(sq_all, axis=0), 0.0))


# ── Precomputed loading ────────────────────────────────────────────────────────

# Maps (path_name, speed_type) → (t_grid_file, speed_file)
_PRECOMPUTED = {
    ('linear',    'ot'):    ('t_grid.npy',     'ot_weighting.npy'),
    ('linear',    'score'): ('t_grid.npy',     'score_weighting.npy'),
    ('linear',    'fr'):    ('t_grid.npy',     'fr_weighting.npy'),
    ('spherical', 'ot'):    ('t_grid_sph.npy', 'ot_speed_sph.npy'),
    ('spherical', 'score'): ('t_grid_sph.npy', 'score_speed_sph.npy'),
    ('spherical', 'fr'):    ('t_grid_sph.npy', 'fr_speed_sph.npy'),
}


def load_precomputed(speed_dir, path_name, speed_type):
    """
    Load (t_grid, v_t) from precomputed .npy files in speed_dir.
    Raises FileNotFoundError with a helpful message if files are missing.
    """
    key = (path_name, speed_type)
    if key not in _PRECOMPUTED:
        raise ValueError(f'No precomputed entry for path={path_name}, type={speed_type}')
    t_file, v_file = _PRECOMPUTED[key]
    t_path = os.path.join(speed_dir, t_file)
    v_path = os.path.join(speed_dir, v_file)
    missing = [p for p in (t_path, v_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f'Precomputed speed files not found:\n' +
            '\n'.join(f'  {p}' for p in missing) +
            '\nRun the matching speed analysis script first, or use '
            '--t_mode curriculum to compute speed online from the model.')
    t_grid = np.load(t_path)
    v_t    = np.load(v_path)
    print(f'Loaded precomputed {speed_type} speed ({path_name}): '
          f't∈[{t_grid[0]:.3f},{t_grid[-1]:.3f}]  '
          f'v∈[{v_t.min():.3f},{v_t.max():.3f}]')
    return t_grid, v_t


def make_cdf_sampler(t_grid, v_t, T_MAX, smooth_sigma=0.05):
    """Convenience wrapper: build a CdfSampler from (t_grid, v_t)."""
    return CdfSampler(t_grid, v_t, T_MAX, smooth_sigma=smooth_sigma)
