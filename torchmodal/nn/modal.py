"""
torchmodal.nn.modal
~~~~~~~~~~~~~~~~~~~

Core modal operator neurons: **Necessity (□)** and **Possibility (♢)**.

These are the central building blocks of the MLNN framework, implementing
differentiable Kripke semantics (Section 3.2.1 of the paper).

The Necessity neuron acts as a "weakest link" detector — aggregating
truth values across accessible worlds via differentiable implication.

The Possibility neuron acts as an "evidence scout" — seeking any
accessible world where the proposition holds.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from torchmodal import functional as F

__all__ = [
    "Necessity",
    "Possibility",
]


class Necessity(nn.Module):
    r"""Necessity (Box / □) neuron.

    Implements the differentiable universal quantification over accessible
    worlds:

    .. math::
        L_{\Box\phi,w} = \text{softmin}_\tau \bigl\{
            (1 - \tilde{A}_{w,w'}) + L_{\phi,w'} \bigr\}_{w' \in W}

    .. math::
        U_{\Box\phi,w} = \text{conv\_pool}_\tau \bigl\{
            (1 - \tilde{A}_{w,w'}) + U_{\phi,w'} \bigr\}_{w' \in W}

    Args:
        tau: Temperature for soft aggregation. Default 0.1.
        learnable_tau: If ``True``, temperature is learnable. Default False.

    Example::

        >>> box = torchmodal.nn.Necessity(tau=0.1)
        >>> # prop_bounds: (|W|, 2) truth bounds for proposition ϕ
        >>> # A: (|W|, |W|) accessibility matrix
        >>> box_phi = box(prop_bounds, A)
    """

    def __init__(
        self, tau: float = 0.1, learnable_tau: bool = False
    ) -> None:
        super().__init__()
        if learnable_tau:
            self.tau = nn.Parameter(torch.tensor(tau))
        else:
            self.register_buffer("tau", torch.tensor(tau))

    def forward(
        self, prop_bounds: Tensor, accessibility: Tensor
    ) -> Tensor:
        """
        Args:
            prop_bounds: ``(|W|, 2)`` or ``(|W|,)`` truth bounds for ϕ.
            accessibility: ``(|W|, |W|)`` accessibility matrix in [0, 1].

        Returns:
            ``(|W|, 2)`` or ``(|W|,)`` truth bounds for □ϕ.
        """
        return F.necessity(prop_bounds, accessibility, tau=self.tau.item())

    def extra_repr(self) -> str:
        return f"tau={self.tau.item():.4f}"


class Possibility(nn.Module):
    r"""Possibility (Diamond / ♢) neuron.

    Implements the differentiable existential quantification over accessible
    worlds:

    .. math::
        L_{\Diamond\phi,w} = \text{conv\_pool}_\tau \bigl\{
            \tilde{A}_{w,w'} + L_{\phi,w'} - 1 \bigr\}_{w' \in W}

    .. math::
        U_{\Diamond\phi,w} = \text{softmax}_\tau \bigl\{
            \tilde{A}_{w,w'} + U_{\phi,w'} - 1 \bigr\}_{w' \in W}

    Satisfies modal duality: ``♢ϕ ≡ ¬□¬ϕ``.

    Args:
        tau: Temperature for soft aggregation. Default 0.1.
        learnable_tau: If ``True``, temperature is learnable. Default False.

    Example::

        >>> diamond = torchmodal.nn.Possibility(tau=0.1)
        >>> dia_phi = diamond(prop_bounds, A)
    """

    def __init__(
        self, tau: float = 0.1, learnable_tau: bool = False
    ) -> None:
        super().__init__()
        if learnable_tau:
            self.tau = nn.Parameter(torch.tensor(tau))
        else:
            self.register_buffer("tau", torch.tensor(tau))

    def forward(
        self, prop_bounds: Tensor, accessibility: Tensor
    ) -> Tensor:
        """
        Args:
            prop_bounds: ``(|W|, 2)`` or ``(|W|,)`` truth bounds for ϕ.
            accessibility: ``(|W|, |W|)`` accessibility matrix in [0, 1].

        Returns:
            ``(|W|, 2)`` or ``(|W|,)`` truth bounds for ♢ϕ.
        """
        return F.possibility(prop_bounds, accessibility, tau=self.tau.item())

    def extra_repr(self) -> str:
        return f"tau={self.tau.item():.4f}"
