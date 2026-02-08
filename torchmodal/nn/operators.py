"""
torchmodal.nn.operators
~~~~~~~~~~~~~~~~~~~~~~~

Differentiable aggregation operators as ``nn.Module`` wrappers.

These modules wrap the functional API in :mod:`torchmodal.functional`,
adding learnable or configurable temperature parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from torchmodal import functional as F

__all__ = [
    "Softmin",
    "Softmax",
    "ConvPool",
]


class Softmin(nn.Module):
    r"""Differentiable soft minimum module.

    Sound lower bound on :func:`torch.min`:
    ``softmin(x) <= min(x)`` for ``x_i \in [0, 1]``.

    Args:
        tau: Initial temperature. Default 0.1.
        learnable: If ``True``, ``tau`` is a learnable parameter.
        dim: Dimension to aggregate over. Default -1.
    """

    def __init__(
        self,
        tau: float = 0.1,
        learnable: bool = False,
        dim: int = -1,
    ) -> None:
        super().__init__()
        if learnable:
            self.tau = nn.Parameter(torch.tensor(tau))
        else:
            self.register_buffer("tau", torch.tensor(tau))
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return F.softmin(x, tau=self.tau.item(), dim=self.dim)

    def extra_repr(self) -> str:
        return f"tau={self.tau.item():.4f}, dim={self.dim}"


class Softmax(nn.Module):
    r"""Differentiable soft maximum module.

    Sound upper bound on :func:`torch.max`:
    ``softmax(x) >= max(x)`` for ``x_i \in [0, 1]``.

    Args:
        tau: Initial temperature. Default 0.1.
        learnable: If ``True``, ``tau`` is a learnable parameter.
        dim: Dimension to aggregate over. Default -1.
    """

    def __init__(
        self,
        tau: float = 0.1,
        learnable: bool = False,
        dim: int = -1,
    ) -> None:
        super().__init__()
        if learnable:
            self.tau = nn.Parameter(torch.tensor(tau))
        else:
            self.register_buffer("tau", torch.tensor(tau))
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return F.softmax(x, tau=self.tau.item(), dim=self.dim)

    def extra_repr(self) -> str:
        return f"tau={self.tau.item():.4f}, dim={self.dim}"


class ConvPool(nn.Module):
    r"""Convex pooling module.

    Args:
        tau: Initial temperature. Default 0.1.
        learnable: If ``True``, ``tau`` is a learnable parameter.
        dim: Dimension to pool over. Default -1.
    """

    def __init__(
        self,
        tau: float = 0.1,
        learnable: bool = False,
        dim: int = -1,
    ) -> None:
        super().__init__()
        if learnable:
            self.tau = nn.Parameter(torch.tensor(tau))
        else:
            self.register_buffer("tau", torch.tensor(tau))
        self.dim = dim

    def forward(self, x: Tensor, z: Tensor | None = None) -> Tensor:
        if z is None:
            z = x
        return F.conv_pool(x, z, tau=self.tau.item(), dim=self.dim)

    def extra_repr(self) -> str:
        return f"tau={self.tau.item():.4f}, dim={self.dim}"
