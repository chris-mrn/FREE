"""Flow matching loss computation and OT coupling."""
import torch
import torch.nn as nn

from utils.helpers import ema_update


def maybe_ot_pair(x0, x1, ot_sampler):
    if ot_sampler is None:
        return x0, x1
    shape = x0.shape
    x0_flat = x0.view(len(x0), -1)
    x1_flat = x1.view(len(x1), -1)
    x0_paired, x1_paired = ot_sampler.sample_plan(x0_flat, x1_flat, replace=False)
    return x0_paired.view(shape), x1_paired.view(shape)


def training_step(net, net_module, ema_net, path, looper, ot_sampler,
                  optim, lr_sched, t, args, image_ds, device):
    """Single training step: forward, backward, EMA update. Returns loss value."""
    x1 = next(looper)
    if image_ds:
        x1 = x1.to(device)
    x0 = torch.randn_like(x1)

    x0, x1 = maybe_ot_pair(x0, x1, ot_sampler)

    xt = path.xt(t, x0, x1)
    ut = path.ut(t, x0, x1)

    optim.zero_grad()
    loss = ((net(t, xt) - ut) ** 2).mean()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
    optim.step()
    lr_sched.step()
    ema_update(net_module, ema_net, args.ema_decay)

    return loss.item()
