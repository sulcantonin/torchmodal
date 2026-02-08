"""
torchmodal.nn.connectives
~~~~~~~~~~~~~~~~~~~~~~~~~

Propositional logic connectives as ``nn.Module`` wrappers.

These implement Łukasiewicz fuzzy logic operators over real-valued
truth bounds in [0, 1], following the LNN framework (Riegel et al., 2020)
as extended by MLNN (Sulc, 2026).

Each connective operates on truth bounds ``[L, U] ⊆ [0, 1]`` and
preserves the bound invariant ``L <= U``.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from torchmodal import functional as F

__all__ = [
    "Negation",
    "Conjunction",
    "Disjunction",
    "Implication",
]


class Negation(nn.Module):
    r"""Fuzzy negation: :math:`\neg x = 1 - x`.

    For bounds, swaps and negates: ``[L', U'] = [1-U, 1-L]``.
    """

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() >= 1 and x.shape[-1] == 2:
            # Bounds tensor: swap L and U after negation
            neg = F.negation(x)
            return neg.flip(-1)
        return F.negation(x)


class Conjunction(nn.Module):
    r"""Łukasiewicz conjunction (fuzzy AND).

    For bounds:
    - ``L_{a∧b} = max(0, L_a + L_b - 1)``
    - ``U_{a∧b} = min(U_a, U_b)``
    """

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        if a.dim() >= 1 and a.shape[-1] == 2:
            import torch

            L = F.conjunction(a[..., 0], b[..., 0])
            U = torch.min(a[..., 1], b[..., 1])
            return torch.stack([L, U], dim=-1)
        return F.conjunction(a, b)


class Disjunction(nn.Module):
    r"""Łukasiewicz disjunction (fuzzy OR).

    For bounds:
    - ``L_{a∨b} = max(L_a, L_b)``
    - ``U_{a∨b} = min(1, U_a + U_b)``
    """

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        if a.dim() >= 1 and a.shape[-1] == 2:
            import torch

            L = torch.max(a[..., 0], b[..., 0])
            U = F.disjunction(a[..., 1], b[..., 1])
            return torch.stack([L, U], dim=-1)
        return F.disjunction(a, b)


class Implication(nn.Module):
    r"""Łukasiewicz implication: :math:`a \to b = \min(1, 1 - a + b)`.

    For bounds:
    - ``L_{a→b} = max(0, 1 - U_a + L_b)``  (strongest constraint)
    - ``U_{a→b} = min(1, 1 - L_a + U_b)``
    """

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        if a.dim() >= 1 and a.shape[-1] == 2:
            import torch

            L = torch.clamp(1.0 - a[..., 1] + b[..., 0], min=0.0, max=1.0)
            U = torch.clamp(1.0 - a[..., 0] + b[..., 1], min=0.0, max=1.0)
            return torch.stack([L, U], dim=-1)
        return F.implication(a, b)
