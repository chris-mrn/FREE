"""
compare.py — Speed comparison utilities.

Core functions for comparing speed profiles across methods (OT / Score / FR)
and between batch closed-form vs. model Hutchinson estimates.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def compute_weighting(speeds: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """w(t) = (∫₀¹ v_s ds) / v_t — sampling weight proportional to speed."""
    dt     = np.diff(t_grid)
    mean_v = float(np.sum(0.5 * (speeds[:-1] + speeds[1:]) * dt))
    return mean_v / np.maximum(speeds, 1e-12)


def smooth_weighting(w: np.ndarray, t_grid: np.ndarray,
                     sigma_t: float = 0.05) -> np.ndarray:
    """Gaussian-smooth w(t) with sigma in t-units (not grid steps)."""
    if sigma_t <= 0:
        return w.copy()
    dt = (t_grid[-1] - t_grid[0]) / max(len(t_grid) - 1, 1)
    return gaussian_filter1d(w, sigma=sigma_t / dt, mode='reflect')


def compute_density(w: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """p(t) = w(t) / ∫ w(s) ds — normalised sampling density."""
    dt = np.diff(t_grid)
    Z  = np.sum(0.5 * (w[:-1] + w[1:]) * dt)
    return w / max(Z, 1e-12)


def interp_to(t_src: np.ndarray, v_src: np.ndarray,
              t_dst: np.ndarray) -> np.ndarray:
    """Interpolate (t_src, v_src) onto t_dst."""
    return np.interp(t_dst, t_src, v_src)
