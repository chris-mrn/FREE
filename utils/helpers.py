"""Core training utilities: EMA, LR schedule, blend, DDP setup."""
import math
import os


def ema_update(src, tgt, decay):
    for sp, tp in zip(src.parameters(), tgt.parameters()):
        tp.data.mul_(decay).add_(sp.data, alpha=1.0 - decay)


def warmup_lr(step, warmup):
    return min(step + 1, warmup) / warmup


def cosine_blend(step, start, duration):
    progress = min(1.0, max(0.0, (step - start) / max(duration, 1)))
    return 0.5 * (1.0 - math.cos(math.pi * progress))


def setup_ddp():
    import torch
    import torch.distributed as dist
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size(), (local_rank == 0)
