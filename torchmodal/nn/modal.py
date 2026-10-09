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

from typing import Optional

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
        L_{\Box\phi,w} = \operatorname{smooth\_min}_\tau \bigl\{
            (1 - \tilde{A}_{w,w'}) + L_{\phi,w'} \bigr\}_{w' \in W}

    .. math::
        U_{\Box\phi,w} = \operatorname{conv\_pool}_\tau \bigl\{
            (1 - \tilde{A}_{w,w'}) + U_{\phi,w'} \bigr\}_{w' \in W}

    With ``top_k=k`` each endpoint aggregates only the *k smallest* of its
    own implication terms (``(1 - Ã) + L`` for the lower bound,
    ``(1 - Ã) + U`` for the upper). The selection is made on the aggregated
    terms, not on ``Ã`` alone, so the true minimum is always kept: the
    bounds stay sound, the smooth lower bound is within ``tau * log(k)``
    of the crisp minimum, and the result does not depend on ``|W|``. This
    is where top-k masking belongs — it used to live on the accessibility
    modules, which was unsound (see :func:`torchmodal.functional.necessity`).

    Args:
        tau: Temperature for soft aggregation. Default 0.1.
        learnable_tau: If ``True``, temperature is an ``nn.Parameter`` that
            receives gradient through the aggregation (the operator is
            differentiable in ``tau``). Nothing keeps it positive: clamp it
            after each optimiser step, or use :meth:`set_tau` with a schedule
            instead. Default False.
        top_k: If set, aggregate only the ``top_k`` smallest implication
            terms per world and endpoint. Default ``None`` (full row).

    For temperature annealing during training, update the temperature via
    :meth:`set_tau` rather than assigning to ``.tau`` (buffers/parameters
    cannot be assigned a plain float).

    Example::

        >>> box = torchmodal.nn.Necessity(tau=0.1)
        >>> # prop_bounds: (|W|, 2) truth bounds for proposition ϕ
        >>> # A: (|W|, |W|) accessibility matrix
        >>> box_phi = box(prop_bounds, A)
        >>> box.set_tau(0.05)  # annealing
        >>> box_k = torchmodal.nn.Necessity(tau=0.1, top_k=8)  # k-neighbourhoods
    """

    def __init__(
        self,
        tau: float = 0.1,
        learnable_tau: bool = False,
        top_k: Optional[int] = None,
    ) -> None:
        super().__init__()
        if learnable_tau:
            self.tau = nn.Parameter(torch.tensor(tau))
        else:
            self.register_buffer("tau", torch.tensor(tau))
        if top_k is not None and top_k < 1:
            raise ValueError(f"top_k must be a positive integer or None, got {top_k}")
        self.top_k = top_k

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
        return F.necessity(
            prop_bounds, accessibility, tau=self.tau, top_k=self.top_k
        )

    def set_tau(self, tau: float) -> None:
        """Set temperature from a float (e.g. for annealing).

        Works for both the buffer and the ``learnable_tau`` parameter; the
        copy runs under ``torch.no_grad`` because an in-place write to a leaf
        that requires grad is otherwise refused by autograd.
        """
        t = torch.as_tensor(tau, device=self.tau.device, dtype=self.tau.dtype)
        with torch.no_grad():
            self.tau.copy_(t)

    def extra_repr(self) -> str:
        return f"tau={self.tau.item():.4f}, top_k={self.top_k}"


class Possibility(nn.Module):
    r"""Possibility (Diamond / ♢) neuron.

    Implements the differentiable existential quantification over accessible
    worlds:

    .. math::
        L_{\Diamond\phi,w} = \operatorname{conv\_pool}_\tau \bigl\{
            \tilde{A}_{w,w'} + L_{\phi,w'} - 1 \bigr\}_{w' \in W}

    .. math::
        U_{\Diamond\phi,w} = \operatorname{smooth\_max}_\tau \bigl\{
            \tilde{A}_{w,w'} + U_{\phi,w'} - 1 \bigr\}_{w' \in W}

    Satisfies modal duality: ``♢ϕ ≡ ¬□¬ϕ``.

    With ``top_k=k`` each endpoint aggregates only the *k largest* of its
    own conjunction terms (``Ã + L - 1`` for the lower bound, ``Ã + U - 1``
    for the upper), so the true maximum is always kept, the bounds stay
    sound, and the smooth upper bound is within ``tau * log(k)`` of the
    crisp maximum. See :class:`Necessity`.

    Args:
        tau: Temperature for soft aggregation. Default 0.1.
        learnable_tau: If ``True``, temperature is an ``nn.Parameter`` that
            receives gradient through the aggregation. See :class:`Necessity`
            for the caveat on keeping it positive. Default False.
        top_k: If set, aggregate only the ``top_k`` largest conjunction
            terms per world and endpoint. Default ``None`` (full row).

    For temperature annealing during training, update the temperature via
    :meth:`set_tau` rather than assigning to ``.tau``.

    Example::

        >>> diamond = torchmodal.nn.Possibility(tau=0.1)
        >>> dia_phi = diamond(prop_bounds, A)
    """

    def __init__(
        self,
        tau: float = 0.1,
        learnable_tau: bool = False,
        top_k: Optional[int] = None,
    ) -> None:
        super().__init__()
        if learnable_tau:
            self.tau = nn.Parameter(torch.tensor(tau))
        else:
            self.register_buffer("tau", torch.tensor(tau))
        if top_k is not None and top_k < 1:
            raise ValueError(f"top_k must be a positive integer or None, got {top_k}")
        self.top_k = top_k

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
        return F.possibility(
            prop_bounds, accessibility, tau=self.tau, top_k=self.top_k
        )

    def set_tau(self, tau: float) -> None:
        """Set temperature from a float (e.g. for annealing).

        Works for both the buffer and the ``learnable_tau`` parameter; the
        copy runs under ``torch.no_grad`` because an in-place write to a leaf
        that requires grad is otherwise refused by autograd.
        """
        t = torch.as_tensor(tau, device=self.tau.device, dtype=self.tau.dtype)
        with torch.no_grad():
            self.tau.copy_(t)

    def extra_repr(self) -> str:
        return f"tau={self.tau.item():.4f}, top_k={self.top_k}"
