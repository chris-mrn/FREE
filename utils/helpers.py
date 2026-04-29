"""
helpers.py — General training utilities (EMA, LR schedule, checkpoints,
             data loops, image/2D eval, DDP setup).
"""
import copy, glob, math, os, time
import numpy as np
import torch
import torch.nn as nn


# ── EMA + LR schedule ─────────────────────────────────────────────────────────

def ema_update(src, tgt, decay):
    for sp, tp in zip(src.parameters(), tgt.parameters()):
        tp.data.mul_(decay).add_(sp.data, alpha=1.0 - decay)


def warmup_lr(step, warmup):
    return min(step + 1, warmup) / warmup


def cosine_blend(step, start, duration):
    progress = min(1.0, max(0.0, (step - start) / max(duration, 1)))
    return 0.5 * (1.0 - math.cos(math.pi * progress))


# ── Checkpoints ───────────────────────────────────────────────────────────────

def find_last_ckpt(out_dir):
    files = sorted(glob.glob(f'{out_dir}/checkpoints/ckpt_step_*.pt'))
    return files[-1] if files else None


def save_ckpt(path, net_module, ema_net, optim, sched,
              step, loss_ema, curriculum_state):
    torch.save({
        'net':        net_module.state_dict(),
        'ema':        ema_net.state_dict(),
        'optim':      optim.state_dict(),
        'sched':      sched.state_dict(),
        'step':       step,
        'loss_ema':   loss_ema,
        'curriculum': curriculum_state,
    }, path)


# ── Data loops ────────────────────────────────────────────────────────────────

def infinite_image_loop(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for x, _ in loader:
            yield x
        epoch += 1


def infinite_2d_loop(dataset_name, batch_size, device):
    from datasets import sample_2d
    while True:
        yield sample_2d(dataset_name, batch_size).to(device)


# ── Eval ─────────────────────────────────────────────────────────────────────

def run_image_eval(step, ema_model, path, inception, real_mu, real_sig,
                   real_feats, out_dir, n_samples, device, n_steps=35):
    from torchvision.utils import make_grid, save_image
    from path import euler_sample
    t0 = time.time()
    x0_shape = (3, 32, 32)
    samples  = euler_sample(ema_model, path, n_samples, n_steps, x0_shape, device)
    grid     = make_grid(samples[:64].clamp(-1, 1), nrow=8,
                         normalize=True, value_range=(-1, 1))
    save_image(grid, f'{out_dir}/samples/step_{step:07d}.png')
    feats_f, probs_f = inception.get_activations(samples)
    fid            = inception.compute_fid(real_mu, real_sig, feats_f)
    kid_m, kid_s   = inception.compute_kid(real_feats, feats_f)
    is_m,  is_s    = inception.compute_is(probs_f)
    print(f'  [eval {step}] FID={fid:.2f}  KID={kid_m:.3f}±{kid_s:.3f}'
          f'  IS={is_m:.2f}±{is_s:.2f}  ({time.time()-t0:.0f}s)')
    return fid, kid_m, kid_s, is_m, is_s


def run_2d_eval(step, ema_model, path, dataset_name, out_dir, n_samples, device):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from path import euler_sample
    from datasets import sample_2d
    x0_shape = (2,)
    samples  = euler_sample(ema_model, path, n_samples, 200, x0_shape, device)
    ref      = sample_2d(dataset_name, n_samples)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f'Step {step}')
    axes[0].scatter(ref[:, 0], ref[:, 1], s=1, alpha=0.3, label='data')
    axes[0].set_title('Data'); axes[0].set_aspect('equal')
    axes[1].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.3, label='model')
    axes[1].set_title('Generated'); axes[1].set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/samples/step_{step:07d}.png', dpi=120)
    plt.close()


# ── DDP ───────────────────────────────────────────────────────────────────────

def setup_ddp():
    import torch.distributed as dist
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size(), (local_rank == 0)
