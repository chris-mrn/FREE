"""Tests for evaluation/speed module."""
import numpy as np
import torch
from evaluation.speed import UniformSampler, CdfSampler, make_cdf_sampler


def test_uniform_sampler():
    s = UniformSampler(1.0)
    t = s.sample(64, 'cpu')
    assert t.shape == (64,)
    assert t.min() >= 0.0
    assert t.max() <= 1.0


def test_cdf_sampler():
    t_grid = np.linspace(0.0, 1.0, 50)
    v_t    = np.ones(50)
    s      = make_cdf_sampler(t_grid, v_t, T_MAX=1.0)
    t      = s.sample(128, 'cpu')
    assert t.shape == (128,)
    assert t.min() >= 0.0
    assert t.max() <= 1.0


def test_cdf_sampler_nonuniform():
    t_grid = np.linspace(0.01, 0.99, 100)
    v_t    = np.exp(-4 * (t_grid - 0.5) ** 2)
    s      = make_cdf_sampler(t_grid, v_t, T_MAX=1.0)
    t      = s.sample(1000, 'cpu')
    # Should concentrate samples near t=0.5
    assert t.shape == (1000,)
    assert abs(t.float().mean().item() - 0.5) < 0.05
