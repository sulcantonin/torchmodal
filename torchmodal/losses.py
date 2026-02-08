"""
torchmodal.losses
~~~~~~~~~~~~~~~~~

Loss functions for MLNN training.

The combined loss drives learning by balancing task performance against
logical consistency:

.. math::
    \\mathcal{L}_{\\text{total}} =
        \\mathcal{L}_{\\text{task}} + \\beta \\mathcal{L}_{\\text{contra}}
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from torchmodal import functional as F

__all__ = [
    "ContradictionLoss",
    "ModalLoss",
    "SparsityLoss",
    "CrystallizationLoss",
    "AxiomRegularization",
]


class ContradictionLoss(nn.Module):
    r"""Logical contradiction loss.

    Penalizes states where the lower bound exceeds the upper bound,
    indicating a logical inconsistency in the Kripke model:

    .. math::
        \mathcal{L}_{\text{contra}} =
            \sum_{w \in W} \sum_\phi \max(0,\; L_{\phi,w} - U_{\phi,w})

    Can use either ``sum`` or ``mean`` reduction, and optionally
    applies squared penalty for smoother gradients.

    Args:
        reduction: ``'sum'``, ``'mean'``, or ``'none'``. Default ``'mean'``.
        squared: If ``True``, use squared contradiction ``max(0, L-U)²``.
            Provides smoother gradients near zero. Default ``False``.
    """

    def __init__(
        self,
        reduction: str = "mean",
        squared: bool = False,
    ) -> None:
        super().__init__()
        assert reduction in ("sum", "mean", "none")
        self.reduction = reduction
        self.squared = squared

    def forward(self, bounds: Tensor) -> Tensor:
        """
        Args:
            bounds: Tensor of shape ``(..., 2)`` with ``[L, U]`` bounds.

        Returns:
            Contradiction loss (scalar or per-element).
        """
        L = bounds[..., 0]
        U = bounds[..., 1]
        contra = torch.relu(L - U)

        if self.squared:
            contra = contra ** 2

        if self.reduction == "sum":
            return contra.sum()
        elif self.reduction == "mean":
            return contra.mean()
        return contra


class ModalLoss(nn.Module):
    r"""Combined modal training loss.

    .. math::
        \mathcal{L}_{\text{total}} =
            \mathcal{L}_{\text{task}} + \beta \mathcal{L}_{\text{contra}}

    This is the standard MLNN training objective (Equation 3 of the paper).
    The task loss drives performance, while the contradiction loss ensures
    logical consistency. The hyperparameter β controls the trade-off.

    Args:
        beta: Weight for the contradiction loss. Default 0.1.
        squared: Use squared contradiction penalty. Default ``False``.

    Example::

        >>> criterion = ModalLoss(beta=0.3)
        >>> task_loss = nn.functional.cross_entropy(logits, targets)
        >>> bounds = model.all_bounds()  # dict of (|W|, 2) tensors
        >>> loss = criterion(task_loss, bounds)
    """

    def __init__(self, beta: float = 0.1, squared: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.contra_loss = ContradictionLoss(
            reduction="mean", squared=squared
        )

    def forward(
        self,
        task_loss: Tensor,
        bounds: dict[str, Tensor] | Tensor,
    ) -> Tensor:
        """
        Args:
            task_loss: Task-specific loss (e.g., cross-entropy).
            bounds: Either a dict mapping proposition names to bounds
                ``(|W|, 2)``, or a single bounds tensor ``(..., 2)``.

        Returns:
            Combined scalar loss.
        """
        if isinstance(bounds, dict):
            contra = torch.tensor(0.0, device=task_loss.device)
            for b in bounds.values():
                contra = contra + self.contra_loss(b)
            if len(bounds) > 0:
                contra = contra / len(bounds)
        else:
            contra = self.contra_loss(bounds)

        return task_loss + self.beta * contra

    def extra_repr(self) -> str:
        return f"beta={self.beta}"


class SparsityLoss(nn.Module):
    r"""L1 sparsity regularization on the accessibility matrix.

    Encourages the model to discover the minimal trust structure:

    .. math::
        \mathcal{L}_{\text{sparse}} = \lambda \|A_\theta\|_1

    Typically applied to off-diagonal elements only (self-trust
    is expected).

    Args:
        lambda_sparse: Regularization strength. Default 0.05.
        exclude_diagonal: Exclude diagonal from penalty. Default ``True``.
    """

    def __init__(
        self,
        lambda_sparse: float = 0.05,
        exclude_diagonal: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_sparse = lambda_sparse
        self.exclude_diagonal = exclude_diagonal

    def forward(self, accessibility: Tensor) -> Tensor:
        """
        Args:
            accessibility: Accessibility matrix ``(|W|, |W|)`` in [0, 1].

        Returns:
            Scalar sparsity loss.
        """
        if self.exclude_diagonal:
            mask = 1.0 - torch.eye(
                accessibility.shape[0],
                device=accessibility.device,
            )
            vals = accessibility * mask
        else:
            vals = accessibility
        return self.lambda_sparse * vals.abs().mean()

    def extra_repr(self) -> str:
        return (
            f"lambda_sparse={self.lambda_sparse}, "
            f"exclude_diagonal={self.exclude_diagonal}"
        )


class CrystallizationLoss(nn.Module):
    r"""Entropy minimization loss for forcing crisp truth assignments.

    Used in satisfiability mode (e.g., Sudoku) to push truth values
    toward 0 or 1:

    .. math::
        \mathcal{L}_{\text{crystal}} =
            -\sum_{w,p} p \log p + (1-p) \log(1-p)

    Often combined with temperature annealing for a "phase transition"
    effect.

    Args:
        reduction: ``'sum'`` or ``'mean'``. Default ``'mean'``.
        eps: Small constant for numerical stability. Default 1e-8.
    """

    def __init__(
        self, reduction: str = "mean", eps: float = 1e-8
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, values: Tensor) -> Tensor:
        """
        Args:
            values: Truth values in (0, 1) of any shape.

        Returns:
            Entropy loss (scalar).
        """
        p = torch.clamp(values, self.eps, 1.0 - self.eps)
        entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())

        if self.reduction == "sum":
            return entropy.sum()
        return entropy.mean()


class AxiomRegularization(nn.Module):
    r"""Regularization losses for enforcing modal logic axiom systems.

    Provides differentiable penalties that encourage the learned
    accessibility relation to satisfy structural properties:

    - **Axiom T** (Reflexivity): ``A[i,i] = 1`` for all *i*.
      System T: □ϕ → ϕ (knowledge is veridical).
    - **Axiom 4** (Transitivity): ``A @ A ≤ A``.
      System S4: □ϕ → □□ϕ (positive introspection).
    - **Axiom B** (Symmetry): ``A ≈ Aᵀ``.
      System B: ϕ → □♢ϕ (Brouwerian axiom).

    These can be combined to enforce specific modal logic systems:
    - **System T** = K + Reflexivity
    - **System S4** = K + Reflexivity + Transitivity
    - **System S5** = K + Reflexivity + Transitivity + Symmetry
    - **System B** = K + Reflexivity + Symmetry

    Args:
        reflexivity: Weight for reflexivity penalty. Default 0.0.
        transitivity: Weight for transitivity penalty. Default 0.0.
        symmetry: Weight for symmetry penalty. Default 0.0.

    Example::

        >>> # Enforce System S4 (reflexive + transitive)
        >>> reg = AxiomRegularization(reflexivity=1.0, transitivity=0.5)
        >>> A = model.get_accessibility()
        >>> loss = reg(A)
    """

    def __init__(
        self,
        reflexivity: float = 0.0,
        transitivity: float = 0.0,
        symmetry: float = 0.0,
    ) -> None:
        super().__init__()
        self.reflexivity = reflexivity
        self.transitivity = transitivity
        self.symmetry = symmetry

    def forward(self, accessibility: Tensor) -> Tensor:
        """
        Args:
            accessibility: Accessibility matrix ``(|W|, |W|)`` in [0, 1].

        Returns:
            Scalar regularization loss.
        """
        loss = torch.tensor(0.0, device=accessibility.device)

        if self.reflexivity > 0:
            # Axiom T: diagonal should be 1.0
            diag = torch.diagonal(accessibility)
            loss = loss + self.reflexivity * torch.mean((1.0 - diag) ** 2)

        if self.transitivity > 0:
            # Axiom 4: A @ A should be <= A (elementwise)
            A_sq = torch.mm(accessibility, accessibility)
            # Clamp to [0,1] range for comparison
            A_sq = torch.clamp(A_sq, 0.0, 1.0)
            violation = torch.relu(A_sq - accessibility)
            loss = loss + self.transitivity * torch.mean(violation ** 2)

        if self.symmetry > 0:
            # Axiom B: A should equal A^T
            diff = accessibility - accessibility.t()
            loss = loss + self.symmetry * torch.mean(diff ** 2)

        return loss

    def extra_repr(self) -> str:
        return (
            f"reflexivity={self.reflexivity}, "
            f"transitivity={self.transitivity}, "
            f"symmetry={self.symmetry}"
        )
