"""Time-conditioned MLP for 2D flow matching experiments."""
import torch
import torch.nn as nn


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
