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

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "ContradictionLoss",
    "ModalLoss",
    "SparsityLoss",
    "CrystallizationLoss",
    "AxiomRegularization",
    "SemanticLoss",
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
    - **Axiom D** (Seriality): every world has a successor,
      ``max_j A[i,j] = 1``.
      System D: □ϕ → ♢ϕ (consistency — what is necessary is possible).
    - **Axiom 5** (Euclidean): ``A[i,j] ∧ A[i,k] → A[j,k]``.
      System S5: ♢ϕ → □♢ϕ (negative introspection).

    These can be combined to enforce specific modal logic systems:
    - **System T** = K + Reflexivity
    - **System S4** = K + Reflexivity + Transitivity
    - **System S5** = K + Reflexivity + Transitivity + Symmetry
    - **System B** = K + Reflexivity + Symmetry

    .. warning::
       **Seriality and the identity relation.** The obvious reading of "every
       world has a successor" is ``max_j A[i,j] = 1``, and the identity matrix
       satisfies it perfectly while relating nothing to anything else. Since
       :class:`~torchmodal.nn.LearnableAccessibility` is reflexive by default,
       the identity is exactly where a fit can comfortably settle — so a user
       who asks for "no dead ends" can get a relation that coordinates
       nothing. Pass ``serial_hollow=True`` (the default) to require a
       successor *other than the world itself*, which is what people mean.

    .. note::
       The seriality penalty uses a **hard** ``max``, not
       :func:`~torchmodal.functional.smooth_max`. The smooth surrogate is an
       upper bound on the max, so ``relu(1 - smooth_max(...))`` *understates*
       the violation and scores a non-serial relation as satisfied unless it
       is debiased by ``tau * log n``. Using the exact max avoids the trap;
       the gradient reaches the maximal entry of each row, which is enough to
       drive it toward 1.

    Args:
        reflexivity: Weight for reflexivity penalty. Default 0.0.
        transitivity: Weight for transitivity penalty. Default 0.0.
        symmetry: Weight for symmetry penalty. Default 0.0.
        seriality: Weight for the Axiom D penalty. Default 0.0.
        euclidean: Weight for the Axiom 5 penalty. Default 0.0.
        serial_hollow: When ``True`` (default), seriality ignores self-loops
            so the identity does not satisfy it. Only meaningful when
            ``seriality > 0``.

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
        seriality: float = 0.0,
        euclidean: float = 0.0,
        serial_hollow: bool = True,
    ) -> None:
        super().__init__()
        self.reflexivity = reflexivity
        self.transitivity = transitivity
        self.symmetry = symmetry
        self.seriality = seriality
        self.euclidean = euclidean
        self.serial_hollow = serial_hollow

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

        if self.seriality > 0:
            # Axiom D: every row needs at least one successor.
            A_ser = accessibility
            if self.serial_hollow:
                # Mask the diagonal so a self-loop cannot satisfy the axiom —
                # otherwise the identity scores perfectly while relating
                # nothing to anything else.
                eye = torch.eye(
                    accessibility.shape[-1], device=accessibility.device
                )
                A_ser = accessibility * (1.0 - eye)
            best = A_ser.max(dim=-1).values
            loss = loss + self.seriality * torch.mean(
                torch.relu(1.0 - best) ** 2
            )

        if self.euclidean > 0:
            # Axiom 5: A[i,j] and A[i,k] imply A[j,k].
            # The antecedent uses Godel (min) rather than Lukasiewicz: the
            # Lukasiewicz conjunction drives the antecedent to 0 whenever
            # A[i,j] + A[i,k] <= 1, which makes the constraint vacuously
            # satisfied on exactly the sparse relations where it should bite.
            ante = torch.minimum(
                accessibility.unsqueeze(-1), accessibility.unsqueeze(-2)
            )  # (i, j, k) = min(A[i,j], A[i,k])
            cons = accessibility.unsqueeze(-3)  # (i, j, k) -> A[j,k]
            loss = loss + self.euclidean * torch.mean(
                torch.relu(ante - cons) ** 2
            )

        return loss

    def extra_repr(self) -> str:
        return (
            f"reflexivity={self.reflexivity}, "
            f"transitivity={self.transitivity}, "
            f"symmetry={self.symmetry}, "
            f"seriality={self.seriality}, "
            f"euclidean={self.euclidean}"
        )


class SemanticLoss(nn.Module):
    r"""Semantic constraint loss (Xu et al., 2018) — NeSy baseline.

    Implements the *Semantic Loss* from "A Semantic Loss Function for
    Deep Learning with Symbolic Knowledge" (Xu et al., ICML 2018).
    This is provided as a **baseline** for comparing MLNN's modal
    contradiction loss against non-modal neurosymbolic approaches.

    For a propositional constraint ``C`` over a set of Boolean
    variables with predicted probabilities ``p``, the semantic loss is:

    .. math::
        \mathcal{L}_{\text{semantic}} =
            -\log \sum_{\mathbf{x} \models C}
            \prod_i p_i^{x_i}(1 - p_i)^{1 - x_i}

    This computes the negative log-probability of the constraint being
    satisfied under the current predictions.

    **Relationship to MLNN's ContradictionLoss**:

    Both losses penalize logical inconsistency, but they differ in key
    ways:

    - ``SemanticLoss`` operates over propositional constraints on a
      *single* world — it cannot natively express modal (cross-world)
      constraints like □ϕ or ♢ϕ.
    - ``ContradictionLoss`` operates over truth *bounds* and can
      propagate constraints across worlds via the accessibility relation.
    - ``SemanticLoss`` requires enumerating satisfying assignments
      (exponential in the worst case), while ``ContradictionLoss``
      is always polynomial.

    For **mutual exclusivity** (exactly-one-of-K), the semantic loss
    has a closed-form solution (see :meth:`forward_mutual_exclusive`).

    Args:
        reduction: ``'sum'``, ``'mean'``, or ``'none'``. Default ``'mean'``.

    Example::

        >>> # Mutual exclusivity: exactly one of 9 digits per cell
        >>> sem_loss = SemanticLoss()
        >>> probs = torch.softmax(logits, dim=-1)  # (81, 9)
        >>> loss = sem_loss.forward_mutual_exclusive(probs)

    References:
        Xu et al., "A Semantic Loss Function for Deep Learning with
        Symbolic Knowledge", ICML 2018.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        assert reduction in ("sum", "mean", "none")
        self.reduction = reduction

    def forward_mutual_exclusive(self, probs: Tensor) -> Tensor:
        """Semantic loss for mutual-exclusivity constraints.

        Exactly one variable in each group should be true.  This has
        a closed-form solution that avoids assignment enumeration:

        .. math::
            \\mathcal{L} = -\\log \\sum_k p_k \\prod_{j \\neq k} (1 - p_j)

        Args:
            probs: Predicted probabilities ``(batch, K)`` in [0, 1],
                where K is the number of mutually exclusive classes.

        Returns:
            Semantic loss (scalar or per-element).
        """
        eps = 1e-8
        probs = probs.clamp(eps, 1.0 - eps)

        log_probs = probs.log()
        log_not_probs = (1.0 - probs).log()

        # log Σ_k exp(log p_k + Σ_{j≠k} log(1-p_j))
        # = log Σ_k exp(log p_k - log(1-p_k) + Σ_j log(1-p_j))
        sum_log_not = log_not_probs.sum(dim=-1, keepdim=True)
        per_class = log_probs - log_not_probs + sum_log_not
        log_sat = torch.logsumexp(per_class, dim=-1)

        loss = -log_sat

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        return loss

    def forward(
        self,
        probs: Tensor,
        constraint_type: str = "mutual_exclusive",
    ) -> Tensor:
        """Compute semantic loss for a named constraint type.

        Args:
            probs: Predicted probabilities.
            constraint_type: Currently supports ``"mutual_exclusive"``.

        Returns:
            Semantic loss.
        """
        if constraint_type == "mutual_exclusive":
            return self.forward_mutual_exclusive(probs)
        raise ValueError(
            f"Unknown constraint_type '{constraint_type}'. "
            f"Supported: 'mutual_exclusive'"
        )

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"
