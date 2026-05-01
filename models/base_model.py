"""Abstract base class for all flow matching velocity field models."""
import torch.nn as nn
from abc import abstractmethod


class FlowModel(nn.Module):
    """
    Interface contract for flow matching velocity fields.

    All models expose: forward(t: Tensor, x: Tensor) -> Tensor
    where t has shape (batch,) and x has shape (batch, *dim).
    """
    @abstractmethod
    def forward(self, t, x):
        ...
