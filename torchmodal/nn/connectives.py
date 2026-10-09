"""
torchmodal.nn.connectives
~~~~~~~~~~~~~~~~~~~~~~~~~

Propositional logic connectives as ``nn.Module`` wrappers.

These implement Łukasiewicz fuzzy logic operators over real-valued
truth bounds in [0, 1], following the LNN framework (Riegel et al., 2020)
as extended by MLNN (Sulc, 2026).

Each connective operates on truth bounds ``[L, U] ⊆ [0, 1]`` and
preserves the bound invariant ``L <= U``.

**Bounds or point values?** Each module accepts either a ``(..., 2)`` bound
tensor or a point-valued tensor of any shape. By default the two are told
apart by the trailing dimension, which is ambiguous for a *point-valued*
tensor whose last axis happens to have extent 2 — two worlds, two time
steps — and such a tensor was silently treated as one ``[L, U]`` pair per
row (``Negation()(tensor([0.2, 0.9]))`` returned ``[0.1, 0.8]`` instead of
``[0.8, 0.1]``). Pass ``bounds=True`` or ``bounds=False`` at construction to
make the reading explicit; the functional API in
:mod:`torchmodal.functional` has no such ambiguity, since it resolves the
shape against the accessibility relation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from torchmodal import functional as F

__all__ = [
    "Negation",
    "Conjunction",
    "Disjunction",
    "Implication",
]


class _Connective(nn.Module):
    """Shared bounds/point-value dispatch for the connective modules.

    Args:
        bounds: ``True`` to always read inputs as ``(..., 2)`` bounds,
            ``False`` to always read them as point values, ``None``
            (default) to infer from the trailing dimension — see the
            module docstring for why that inference can be wrong.
    """

    def __init__(self, bounds: Optional[bool] = None) -> None:
        super().__init__()
        self.bounds = bounds

    def _is_bounds(self, x: Tensor) -> bool:
        if self.bounds is not None:
            return self.bounds
        return bool(x.dim() >= 1 and x.shape[-1] == 2)

    def extra_repr(self) -> str:
        return f"bounds={self.bounds}"


class Negation(_Connective):
    r"""Fuzzy negation: :math:`\neg x = 1 - x`.

    For bounds, swaps and negates: ``[L', U'] = [1-U, 1-L]``.
    """

    def forward(self, x: Tensor) -> Tensor:
        if self._is_bounds(x):
            # Bounds tensor: swap L and U after negation
            neg = F.negation(x)
            return neg.flip(-1)
        return F.negation(x)


class Conjunction(_Connective):
    r"""Łukasiewicz conjunction (fuzzy AND).

    For bounds:
    - ``L_{a∧b} = max(0, L_a + L_b - 1)``
    - ``U_{a∧b} = min(U_a, U_b)``
    """

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        if self._is_bounds(a):
            L = F.conjunction(a[..., 0], b[..., 0])
            U = torch.min(a[..., 1], b[..., 1])
            return torch.stack([L, U], dim=-1)
        return F.conjunction(a, b)


class Disjunction(_Connective):
    r"""Łukasiewicz disjunction (fuzzy OR).

    For bounds:
    - ``L_{a∨b} = max(L_a, L_b)``
    - ``U_{a∨b} = min(1, U_a + U_b)``
    """

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        if self._is_bounds(a):
            L = torch.max(a[..., 0], b[..., 0])
            U = F.disjunction(a[..., 1], b[..., 1])
            return torch.stack([L, U], dim=-1)
        return F.disjunction(a, b)


class Implication(_Connective):
    r"""Łukasiewicz implication: :math:`a \to b = \min(1, 1 - a + b)`.

    For bounds:
    - ``L_{a→b} = max(0, 1 - U_a + L_b)``  (strongest constraint)
    - ``U_{a→b} = min(1, 1 - L_a + U_b)``
    """

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        if self._is_bounds(a):
            L = torch.clamp(1.0 - a[..., 1] + b[..., 0], min=0.0, max=1.0)
            U = torch.clamp(1.0 - a[..., 0] + b[..., 1], min=0.0, max=1.0)
            return torch.stack([L, U], dim=-1)
        return F.implication(a, b)
