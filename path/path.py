"""
path.py — Interpolation path definitions for flow matching.

LinearPath:    x_t = (1-t)*x0 + t*x1,           t in [0, 1]
               u_t = x1 - x0
SphericalPath: x_t = cos(t)*x0 + sin(t)*x1,     t in [0, pi/2]
               u_t = -sin(t)*x0 + cos(t)*x1

Both classes expose the same interface so the training loop is path-agnostic.
"""
import math
import torch


def _b(t, x):
    """Reshape t (B,) to broadcast with x (B, d1, d2, ...) of any rank."""
    return t.view(-1, *([1] * (x.dim() - 1)))


class LinearPath:
    T_MAX = 1.0
    name  = 'linear'

    def xt(self, t, x0, x1):
        tb = _b(t, x0)
        return (1.0 - tb) * x0 + tb * x1

    def ut(self, t, x0, x1):
        return x1 - x0


class SphericalPath:
    T_MAX = math.pi / 2
    name  = 'spherical'

    def xt(self, t, x0, x1):
        tb = _b(t, x0)
        return torch.cos(tb) * x0 + torch.sin(tb) * x1

    def ut(self, t, x0, x1):
        tb = _b(t, x0)
        return -torch.sin(tb) * x0 + torch.cos(tb) * x1


def get_path(name):
    if name == 'linear':
        return LinearPath()
    if name == 'spherical':
        return SphericalPath()
    raise ValueError(f'Unknown path "{name}". Choose: linear | spherical')


@torch.no_grad()
def euler_sample(model, path, n_samples, n_steps, x0_shape, device, bs=256):
    """Euler sampler with uniform time steps in [0, T_MAX]."""
    t_sched = [i / n_steps * path.T_MAX for i in range(n_steps + 1)]
    return euler_sample_schedule(model, n_samples, t_sched, x0_shape, device, bs)


@torch.no_grad()
def euler_sample_schedule(model, n_samples, t_schedule, x0_shape, device, bs=256):
    """
    Euler sampler with a custom time schedule.

    t_schedule: sequence of n+1 floats, t_schedule[0]=0, t_schedule[-1]=T_MAX.
    x0_shape  : shape of a single sample, e.g. (3, 32, 32) or (2,).
    """
    model.eval()
    ts = torch.tensor(t_schedule, dtype=torch.float32, device=device)
    out = []
    for i in range(0, n_samples, bs):
        b = min(bs, n_samples - i)
        x = torch.randn(b, *x0_shape, device=device)
        for k in range(len(ts) - 1):
            tv   = torch.full((b,), ts[k].item(), device=device)
            dt_k = (ts[k + 1] - ts[k]).item()
            x    = x + model(tv, x) * dt_k
        out.append(x.cpu())
    model.train()
    return torch.cat(out, 0)
