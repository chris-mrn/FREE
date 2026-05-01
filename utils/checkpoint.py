"""Checkpoint save/load utilities."""
import glob
import torch


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
