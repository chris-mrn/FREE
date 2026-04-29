"""
models.py — Model architectures for flow matching experiments.

Image datasets   : UNetModelWrapper (from torchcfm.models.unet)
2D datasets      : MLP2D (time-conditioned MLP)
"""
import torch
import torch.nn as nn


# ── 2D MLP ─────────────────────────────────────────────────────────────────────

class MLP2D(nn.Module):
    """Time-conditioned MLP for 2D flow matching. Interface: (t, x) -> x."""
    def __init__(self, dim=2, hidden=256, depth=4):
        super().__init__()
        layers = [nn.Linear(dim + 1, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, t, x):
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.net(torch.cat([t.unsqueeze(-1), x], dim=-1))


# ── Build model from args ──────────────────────────────────────────────────────

def build_model(args, device):
    """
    Build and return the flow-matching model on `device`.

    Selects UNetModelWrapper for image datasets (cifar10, etc.) and
    MLP2D for 2D toy datasets.
    """
    from datasets import is_image
    if is_image(args.dataset):
        from torchcfm.models.unet.unet import UNetModelWrapper
        net = UNetModelWrapper(
            dim=(3, 32, 32), num_res_blocks=2, num_channels=args.num_channel,
            channel_mult=[1, 2, 2, 2], num_heads=4, num_head_channels=64,
            attention_resolutions='16', dropout=0.1,
        ).to(device)
    else:
        net = MLP2D(dim=2, hidden=args.hidden_2d, depth=args.depth_2d).to(device)
    return net
