"""Curriculum state management and online speed estimation."""
import numpy as np
import torch

from evaluation.speed import (
    BlendedSampler, estimate_speed_grid, make_cdf_sampler,
)
from utils.helpers import cosine_blend


class CurriculumState:
    """
    Serialisable curriculum state for speed-adaptive t-sampling.

    Phases:
        0 — uniform sampling
        1 — blending (cosine ramp from uniform → speed-adaptive)
        2 — pure speed-adaptive
    """
    def __init__(self):
        self.phase = 0
        self.restart_count = 0
        self.last_speed_step = None
        self.t_grid = None
        self.v_t = None

    def to_dict(self):
        return {
            'phase':            self.phase,
            'restart_count':    self.restart_count,
            'last_speed_step':  self.last_speed_step,
            't_grid':           self.t_grid,
            'v_t':              self.v_t,
        }

    @classmethod
    def from_dict(cls, d):
        state = cls()
        state.phase            = d['phase']
        state.restart_count    = d['restart_count']
        state.last_speed_step  = d['last_speed_step']
        state.t_grid           = d['t_grid']
        state.v_t              = d['v_t']
        return state

    def should_start_curriculum(self, step, args):
        return self.phase == 0 and step == args.curriculum_start

    def should_restart(self, step, args):
        return (self.phase == 2
                and args.curriculum_restarts > 0
                and self.restart_count < args.curriculum_restarts
                and self.last_speed_step is not None
                and step == self.last_speed_step + args.curriculum_restart_every)

    def should_end_blend(self, step, args):
        return (self.phase == 1
                and self.last_speed_step is not None
                and step >= self.last_speed_step + args.curriculum_blend)

    def sample_t(self, B, step, device, uniform_sampler, speed_sampler, args):
        """Sample t given current phase, returns (B,) tensor on device."""
        if speed_sampler is None or self.phase == 0:
            return uniform_sampler.sample(B, device)
        if self.phase == 1:
            mix = cosine_blend(step, self.last_speed_step, args.curriculum_blend)
            return BlendedSampler(uniform_sampler, speed_sampler).sample(B, device, mix)
        return speed_sampler.sample(B, device)


def estimate_and_broadcast_speed(step_label, ema_net, path, x1_ref,
                                  args, curriculum, image_ds,
                                  is_main, use_ddp, device):
    """
    Estimate speed on rank 0, broadcast to all ranks, update curriculum in-place.
    Returns a new CdfSampler.
    """
    import torch.distributed as dist
    from data.datasets import sample_2d

    if use_ddp:
        dist.barrier()
    n_t = args.speed_n_t

    if is_main:
        print(f'\n[speed] Estimating {args.speed_type} speed at step {step_label}...',
              flush=True)
        ref  = x1_ref if image_ds else sample_2d(args.dataset, 2000).to(device)
        t_g  = np.linspace(0.01, path.T_MAX - 0.01, n_t)
        v_t_ = estimate_speed_grid(
            ema_net, path, t_g, ref,
            B=args.speed_B, n_epochs=args.speed_epochs,
            speed_type=args.speed_type, device=device,
            n_hutch=args.speed_hutch)
        print(f'  v_t range: [{v_t_.min():.4f}, {v_t_.max():.4f}]')
        import os
        np.save(f'{args.out_dir}/speed_t_grid_step{step_label}.npy', t_g)
        np.save(f'{args.out_dir}/speed_v_t_step{step_label}.npy', v_t_)
        speed_buf = torch.tensor(
            np.stack([t_g, v_t_]), dtype=torch.float32, device=device)
    else:
        speed_buf = torch.zeros(2, n_t, dtype=torch.float32, device=device)

    if use_ddp:
        dist.broadcast(speed_buf, src=0)

    t_g_local = speed_buf[0].cpu().numpy()
    v_t_local = speed_buf[1].cpu().numpy()
    sampler   = make_cdf_sampler(t_g_local, v_t_local, path.T_MAX,
                                  smooth_sigma=args.speed_smooth)
    curriculum.t_grid          = t_g_local.tolist()
    curriculum.v_t             = v_t_local.tolist()
    curriculum.last_speed_step = step_label

    if use_ddp:
        dist.barrier()
    return sampler
