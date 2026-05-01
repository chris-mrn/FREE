"""
energy.py — Reparameterized divergence computation.

Core functions for computing E[(div u_t^theta(X_{alpha(t)}))^2] under
the arc-length reparametrisation alpha induced by the Fisher-Rao speed.
"""
import numpy as np
import torch


def build_alpha(t_grid, v_t, T_MIN=0.2, T_MAX=0.8):
    """
    Build alpha = F^{-1} where F(t) = int_0^t v^FR(s) ds / int_0^T v^FR(s) ds.

    This is the arc-length parametrisation: equal increments of s correspond
    to equal Fisher-Rao distance travelled.

    Returns:
        alpha    : callable (s_arr: np.ndarray) -> np.ndarray of t values
        t_ext    : extended t grid including T_MIN and T_MAX
        c_ext    : CDF values at t_ext
        w        : speed weights v_t (positive)
    """
    v = np.maximum(v_t, 1e-10).copy()
    w = v
    dt = np.diff(t_grid)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (w[:-1] + w[1:]) * dt)])
    cum /= cum[-1]
    t_ext = np.concatenate([[T_MIN], t_grid, [T_MAX]])
    c_ext = np.concatenate([[0.0],   cum,    [1.0]])

    def alpha(s_arr):
        return np.interp(s_arr, c_ext, t_ext).astype(np.float32)

    return alpha, t_ext, c_ext, w


def hutchinson_div_sq(model, t_val, xt, n_hutch, device):
    """
    Estimate E[(div u_t^theta(X_t))^2] via Hutchinson with n_hutch probes.

    div u ≈ z^T (∂u/∂x) z  for z ~ Rademacher.

    Args:
        model  : velocity network, callable as model(t_batch, x)
        t_val  : scalar float time value
        xt     : (B, *shape) sampled interpolant
        n_hutch: number of Rademacher probes
        device : torch device

    Returns:
        scalar float: mean over batch of (div_estimate)^2
    """
    B = xt.shape[0]
    t = torch.full((B,), t_val, device=device)
    div_estimates = []

    for _ in range(n_hutch):
        x_req = xt.detach().requires_grad_(True)
        u     = model(t, x_req)
        z     = (torch.randint(0, 2, xt.shape, device=device).float() * 2 - 1)
        grad  = torch.autograd.grad(
            (u * z).sum(), x_req,
            retain_graph=False, create_graph=False
        )[0]
        div_k = (grad * z).sum(dim=tuple(range(1, xt.dim())))
        div_estimates.append(div_k.detach())

    div_avg = torch.stack(div_estimates).mean(0)
    return (div_avg ** 2).mean().item()
