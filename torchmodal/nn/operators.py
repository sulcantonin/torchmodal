"""
torchmodal.nn.operators
~~~~~~~~~~~~~~~~~~~~~~~

Differentiable aggregation operators as ``nn.Module`` wrappers.

These modules wrap the functional API in :mod:`torchmodal.functional`,
adding learnable or configurable temperature parameters.

.. note::
   Named ``SmoothMin`` / ``SmoothMax`` (not ``Softmin`` / ``Softmax``)
   to avoid confusion with the standard probability-normalization
   ``torch.softmax``.  Legacy aliases ``Softmin`` and ``Softmax`` are
   provided for backward compatibility.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from torch import Tensor

from torchmodal import functional as F

__all__ = [
    "SmoothMin",
    "SmoothMax",
    "ConvPool",
    # Legacy aliases (deprecated)
    "Softmin",
    "Softmax",
]


class SmoothMin(nn.Module):
    r"""Differentiable smooth minimum module (log-sum-exp lower bound).

    Sound lower bound on :func:`torch.min`:
    ``smooth_min(x) <= min(x)`` for ``x_i \in [0, 1]``.

    Args:
        tau: Initial temperature. Default 0.1.
        learnable: If ``True``, ``tau`` is an ``nn.Parameter`` and receives
            gradient through the aggregation. Nothing keeps it positive;
            clamp it after each optimiser step.
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
        return F.smooth_min(x, tau=self.tau, dim=self.dim)

    def extra_repr(self) -> str:
        return f"tau={self.tau.item():.4f}, dim={self.dim}"


class SmoothMax(nn.Module):
    r"""Differentiable smooth maximum module (log-sum-exp upper bound).

    Sound upper bound on :func:`torch.max`:
    ``smooth_max(x) >= max(x)`` for ``x_i \in [0, 1]``.

    Args:
        tau: Initial temperature. Default 0.1.
        learnable: If ``True``, ``tau`` is an ``nn.Parameter`` and receives
            gradient through the aggregation. Nothing keeps it positive;
            clamp it after each optimiser step.
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
        return F.smooth_max(x, tau=self.tau, dim=self.dim)

    def extra_repr(self) -> str:
        return f"tau={self.tau.item():.4f}, dim={self.dim}"


class ConvPool(nn.Module):
    r"""Convex pooling module.

    Args:
        tau: Initial temperature. Default 0.1.
        learnable: If ``True``, ``tau`` is an ``nn.Parameter`` and receives
            gradient through the pooling weights. Nothing keeps it positive;
            clamp it after each optimiser step.
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
        return F.conv_pool(x, z, tau=self.tau, dim=self.dim)

    def extra_repr(self) -> str:
        return f"tau={self.tau.item():.4f}, dim={self.dim}"


# Legacy aliases ---------------------------------------------------------

def Softmin(*args, **kwargs) -> SmoothMin:  # noqa: N802
    """Deprecated alias for :class:`SmoothMin`."""
    warnings.warn(
        "torchmodal.nn.Softmin is deprecated, use SmoothMin",
        DeprecationWarning,
        stacklevel=2,
    )
    return SmoothMin(*args, **kwargs)


def Softmax(*args, **kwargs) -> SmoothMax:  # noqa: N802
    """Deprecated alias for :class:`SmoothMax`."""
    warnings.warn(
        "torchmodal.nn.Softmax is deprecated, use SmoothMax",
        DeprecationWarning,
        stacklevel=2,
    )
    return SmoothMax(*args, **kwargs)
