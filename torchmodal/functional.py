"""
torchmodal.functional
~~~~~~~~~~~~~~~~~~~~~

Functional API for differentiable modal logic operators.

Provides stateless functions for soft logic aggregations, propositional
connectives, and modal operators following the MLNN framework
(Sulc, 2026) with Kripke semantics.

All functions operate on tensors of truth bounds in [0, 1].
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    # Differentiable aggregations
    "softmin",
    "softmax",
    "conv_pool",
    # Propositional connectives
    "negation",
    "conjunction",
    "disjunction",
    "implication",
    # Modal operators
    "necessity",
    "possibility",
    # Contradiction
    "contradiction",
]

# ---------------------------------------------------------------------------
# Differentiable aggregations (Section 3.2.1 of the paper)
# ---------------------------------------------------------------------------


def softmin(x: Tensor, tau: float = 0.1, dim: int = -1) -> Tensor:
    r"""Differentiable soft minimum.

    .. math::
        \text{softmin}_\tau(\mathbf{x}) =
            -\tau \log \sum_i \exp(-x_i / \tau)

    This is a *sound lower bound* on :func:`torch.min`:
    ``softmin(x) <= min(x)`` for all ``x_i \in [0, 1]``.

    As :math:`\tau \to 0`, converges to :func:`torch.min`.

    Args:
        x: Input tensor of truth values in [0, 1].
        tau: Temperature controlling approximation sharpness. Default 0.1.
        dim: Dimension along which to aggregate. Default -1.

    Returns:
        Tensor with ``dim`` reduced.
    """
    return -tau * torch.logsumexp(-x / tau, dim=dim)


def softmax(x: Tensor, tau: float = 0.1, dim: int = -1) -> Tensor:
    r"""Differentiable soft maximum.

    .. math::
        \text{softmax}_\tau(\mathbf{x}) =
            \tau \log \sum_i \exp(x_i / \tau)

    This is a *sound upper bound* on :func:`torch.max`:
    ``softmax(x) >= max(x)`` for all ``x_i \in [0, 1]``.

    As :math:`\tau \to 0`, converges to :func:`torch.max`.

    Args:
        x: Input tensor of truth values in [0, 1].
        tau: Temperature controlling approximation sharpness. Default 0.1.
        dim: Dimension along which to aggregate. Default -1.

    Returns:
        Tensor with ``dim`` reduced.
    """
    return tau * torch.logsumexp(x / tau, dim=dim)


def conv_pool(
    x: Tensor, z: Tensor, tau: float = 0.1, dim: int = -1
) -> Tensor:
    r"""Convex pooling operator.

    .. math::
        \text{conv\_pool}_\tau(\mathbf{x}, \mathbf{z}) =
            \sum_i w_i x_i, \quad w_i = \text{softmax}(z_i / \tau)

    When ``z = x``, provides a *lower bound* on ``max(x)``.
    When ``z = -x``, provides an *upper bound* on ``min(x)``.

    Args:
        x: Values to pool.
        z: Logits controlling the convex weights.
        tau: Temperature. Default 0.1.
        dim: Dimension along which to pool. Default -1.

    Returns:
        Tensor with ``dim`` reduced.
    """
    weights = torch.softmax(z / tau, dim=dim)
    return (weights * x).sum(dim=dim)


# ---------------------------------------------------------------------------
# Propositional connectives (weighted, real-valued logic)
# ---------------------------------------------------------------------------


def negation(x: Tensor) -> Tensor:
    r"""Fuzzy negation: :math:`\neg x = 1 - x`.

    Args:
        x: Truth values in [0, 1]. Can be bounds ``(L, U)`` — apply to each.

    Returns:
        Negated truth values.
    """
    return 1.0 - x


def conjunction(a: Tensor, b: Tensor) -> Tensor:
    r"""Łukasiewicz conjunction (fuzzy AND).

    .. math::
        a \wedge b = \max(0,\; a + b - 1)

    For bounds: ``L_{a∧b} = max(0, L_a + L_b - 1)``,
    ``U_{a∧b} = min(U_a, U_b)``.

    Args:
        a: First operand truth values in [0, 1].
        b: Second operand truth values in [0, 1].

    Returns:
        Conjunction truth values.
    """
    return torch.clamp(a + b - 1.0, min=0.0)


def disjunction(a: Tensor, b: Tensor) -> Tensor:
    r"""Łukasiewicz disjunction (fuzzy OR).

    .. math::
        a \vee b = \min(1,\; a + b)

    Args:
        a: First operand truth values in [0, 1].
        b: Second operand truth values in [0, 1].

    Returns:
        Disjunction truth values.
    """
    return torch.clamp(a + b, max=1.0)


def implication(a: Tensor, b: Tensor) -> Tensor:
    r"""Łukasiewicz implication.

    .. math::
        a \to b = \min(1,\; 1 - a + b)

    Equivalent to ``disjunction(negation(a), b)``.

    Args:
        a: Antecedent truth values in [0, 1].
        b: Consequent truth values in [0, 1].

    Returns:
        Implication truth values.
    """
    return torch.clamp(1.0 - a + b, max=1.0)


# ---------------------------------------------------------------------------
# Modal operators (Section 3.2.1)
# ---------------------------------------------------------------------------


def necessity(
    prop_bounds: Tensor,
    accessibility: Tensor,
    tau: float = 0.1,
) -> Tensor:
    r"""Necessity (Box / □) operator — differentiable Kripke semantics.

    Computes truth bounds for □ϕ across all worlds using the weighted
    accessibility matrix. For each world *w*:

    .. math::
        L_{\Box\phi,w} = \text{softmin}_\tau \bigl\{
            (1 - \tilde{A}_{w,w'}) + L_{\phi,w'} \bigr\}_{w' \in W}

    .. math::
        U_{\Box\phi,w} = \text{conv\_pool}_\tau \bigl\{
            (1 - \tilde{A}_{w,w'}) + U_{\phi,w'} \bigr\}_{w' \in W}

    The operator acts as a "weakest link" detector: if a world is highly
    accessible (Ã ≈ 1) but ϕ is false there, the score collapses.

    Args:
        prop_bounds: Truth bounds of shape ``(|W|, 2)`` where columns are
            ``[L, U]``, or ``(|W|,)`` for point-valued truth values (treated
            as both L and U).
        accessibility: Accessibility matrix of shape ``(|W|, |W|)``, values
            in [0, 1].
        tau: Temperature. Default 0.1.

    Returns:
        Tensor of shape ``(|W|, 2)`` or ``(|W|,)`` with necessity bounds.
    """
    point_valued = prop_bounds.dim() == 1
    if point_valued:
        prop_bounds = prop_bounds.unsqueeze(-1).expand(-1, 2)

    L_phi = prop_bounds[:, 0]  # (|W|,)
    U_phi = prop_bounds[:, 1]  # (|W|,)

    # (|W|, |W|): implication terms per source-target world pair
    impl_L = (1.0 - accessibility) + L_phi.unsqueeze(0)  # broadcast target
    impl_U = (1.0 - accessibility) + U_phi.unsqueeze(0)

    # Lower bound: softmin over target worlds (dim=1)
    L_box = softmin(impl_L, tau=tau, dim=1)

    # Upper bound: conv_pool with the implication as both value and logit
    U_box = conv_pool(impl_U, impl_U, tau=tau, dim=1)

    result = torch.stack([L_box, U_box], dim=-1)
    result = torch.clamp(result, 0.0, 1.0)

    if point_valued:
        return result[:, 0]
    return result


def possibility(
    prop_bounds: Tensor,
    accessibility: Tensor,
    tau: float = 0.1,
) -> Tensor:
    r"""Possibility (Diamond / ♢) operator — differentiable Kripke semantics.

    Computes truth bounds for ♢ϕ across all worlds. For each world *w*:

    .. math::
        L_{\Diamond\phi,w} = \text{conv\_pool}_\tau \bigl\{
            \tilde{A}_{w,w'} + L_{\phi,w'} - 1 \bigr\}_{w' \in W}

    .. math::
        U_{\Diamond\phi,w} = \text{softmax}_\tau \bigl\{
            \tilde{A}_{w,w'} + U_{\phi,w'} - 1 \bigr\}_{w' \in W}

    The operator acts as an "evidence scout": it activates if it finds any
    world that is both accessible and where ϕ is true.

    Args:
        prop_bounds: Truth bounds of shape ``(|W|, 2)`` or ``(|W|,)``.
        accessibility: Accessibility matrix ``(|W|, |W|)`` in [0, 1].
        tau: Temperature. Default 0.1.

    Returns:
        Tensor of shape ``(|W|, 2)`` or ``(|W|,)`` with possibility bounds.
    """
    point_valued = prop_bounds.dim() == 1
    if point_valued:
        prop_bounds = prop_bounds.unsqueeze(-1).expand(-1, 2)

    L_phi = prop_bounds[:, 0]
    U_phi = prop_bounds[:, 1]

    # conjunction terms
    conj_L = accessibility + L_phi.unsqueeze(0) - 1.0
    conj_U = accessibility + U_phi.unsqueeze(0) - 1.0

    # Lower bound: conv_pool (sound lower bound)
    L_dia = conv_pool(conj_L, conj_L, tau=tau, dim=1)

    # Upper bound: softmax (weighted existential)
    U_dia = softmax(conj_U, tau=tau, dim=1)

    result = torch.stack([L_dia, U_dia], dim=-1)
    result = torch.clamp(result, 0.0, 1.0)

    if point_valued:
        return result[:, 1]  # for point values return upper (existential)
    return result


# ---------------------------------------------------------------------------
# Contradiction measure
# ---------------------------------------------------------------------------


def contradiction(bounds: Tensor) -> Tensor:
    r"""Compute contradiction for a set of bounds.

    .. math::
        \mathcal{L}_{\text{contra}} =
            \sum_{w, \phi} \max(0,\; L_{\phi,w} - U_{\phi,w})

    A contradiction arises when the lower bound exceeds the upper bound,
    indicating a logical inconsistency.

    Args:
        bounds: Tensor of shape ``(..., 2)`` where the last dimension
            holds ``[L, U]``.

    Returns:
        Scalar contradiction loss.
    """
    L = bounds[..., 0]
    U = bounds[..., 1]
    return torch.relu(L - U).sum()
