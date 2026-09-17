"""
torchmodal.functional
~~~~~~~~~~~~~~~~~~~~~

Functional API for differentiable modal logic operators.

Provides stateless functions for soft logic aggregations, propositional
connectives, and modal operators following the MLNN framework
(Sulc, 2026) with Kripke semantics.

All functions operate on tensors of truth bounds in [0, 1].

.. note::
   The aggregation operators are named ``smooth_min`` / ``smooth_max``
   (not ``softmin`` / ``softmax``) to avoid confusion with the standard
   probability-normalization ``torch.softmax``.  These operators are
   *log-sum-exp* aggregations that serve as sound *bounds* on the true
   min/max, not probability distributions.  Legacy aliases ``softmin``
   and ``softmax`` are provided for backward compatibility.
"""

from __future__ import annotations

import math
import warnings

import torch
from torch import Tensor

__all__ = [
    # Differentiable aggregations
    "smooth_min",
    "smooth_max",
    "conv_pool",
    # Legacy aliases (deprecated)
    "softmin",
    "softmax",
    # Propositional connectives
    "negation",
    "conjunction",
    "disjunction",
    "implication",
    # Modal operators
    "necessity",
    "possibility",
    "until",
    "until_graph",
    # Dynamic epistemic logic (model update)
    "announce",
    "necessity_after",
    "group_announce",
    # Precision control
    "auto_tau",
    # Diagnostics
    "box_width_entropy",
    # Contradiction
    "contradiction",
]

# ---------------------------------------------------------------------------
# Differentiable aggregations (Section 3.2.1 of the paper)
#
# Named smooth_min / smooth_max to avoid collision with the standard
# torch.softmax (probability normalization), which is used internally
# by conv_pool.  These are log-sum-exp aggregations providing sound
# bounds on true min/max.
# ---------------------------------------------------------------------------


def smooth_min(x: Tensor, tau: float = 0.1, dim: int = -1) -> Tensor:
    r"""Differentiable smooth minimum (log-sum-exp lower bound).

    .. math::
        \operatorname{smooth\_min}_\tau(\mathbf{x}) =
            -\tau \log \sum_i \exp(-x_i / \tau)

    This is a *sound lower bound* on :func:`torch.min`:
    ``smooth_min(x) <= min(x)`` for all ``x_i \in [0, 1]``.

    As :math:`\tau \to 0`, converges to :func:`torch.min`.

    .. note::
       Not to be confused with ``torch.softmax`` (probability normalization).
       This function computes a *scalar aggregation* via the log-sum-exp
       identity, not a probability distribution.

    Args:
        x: Input tensor of truth values in [0, 1].
        tau: Temperature controlling approximation sharpness. Default 0.1.
        dim: Dimension along which to aggregate. Default -1.

    Returns:
        Tensor with ``dim`` reduced.
    """
    return -tau * torch.logsumexp(-x / tau, dim=dim)


def smooth_max(x: Tensor, tau: float = 0.1, dim: int = -1) -> Tensor:
    r"""Differentiable smooth maximum (log-sum-exp upper bound).

    .. math::
        \operatorname{smooth\_max}_\tau(\mathbf{x}) =
            \tau \log \sum_i \exp(x_i / \tau)

    This is a *sound upper bound* on :func:`torch.max`:
    ``smooth_max(x) >= max(x)`` for all ``x_i \in [0, 1]``.

    As :math:`\tau \to 0`, converges to :func:`torch.max`.

    .. note::
       Not to be confused with ``torch.softmax`` (probability normalization).
       This function computes a *scalar aggregation* via the log-sum-exp
       identity, not a probability distribution.

    Args:
        x: Input tensor of truth values in [0, 1].
        tau: Temperature controlling approximation sharpness. Default 0.1.
        dim: Dimension along which to aggregate. Default -1.

    Returns:
        Tensor with ``dim`` reduced.
    """
    return tau * torch.logsumexp(x / tau, dim=dim)


# Legacy aliases --------------------------------------------------------

def softmin(x: Tensor, tau: float = 0.1, dim: int = -1) -> Tensor:
    """Deprecated alias for :func:`smooth_min`."""
    warnings.warn(
        "torchmodal.functional.softmin is deprecated, "
        "use smooth_min to avoid confusion with torch.softmax",
        DeprecationWarning,
        stacklevel=2,
    )
    return smooth_min(x, tau=tau, dim=dim)


def softmax(x: Tensor, tau: float = 0.1, dim: int = -1) -> Tensor:
    """Deprecated alias for :func:`smooth_max`."""
    warnings.warn(
        "torchmodal.functional.softmax is deprecated, "
        "use smooth_max to avoid confusion with torch.softmax",
        DeprecationWarning,
        stacklevel=2,
    )
    return smooth_max(x, tau=tau, dim=dim)


def conv_pool(
    x: Tensor, z: Tensor, tau: float = 0.1, dim: int = -1
) -> Tensor:
    r"""Convex pooling operator (attention-weighted average).

    Computes a convex combination of ``x`` using attention weights
    derived from ``z``:

    .. math::
        \operatorname{conv\_pool}_\tau(\mathbf{x}, \mathbf{z}) =
            \sum_i w_i\, x_i, \quad
            w_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}

    The weights ``w`` are a standard probability-normalized softmax
    (``torch.softmax``) applied to the *logits* ``z / tau``.

    **Bound properties** (for ``x_i \in [0, 1]``):

    - ``z = x``  → the largest values receive the highest weight,
      providing a differentiable *lower bound* on ``max(x)``.
    - ``z = -x`` → the smallest values receive the highest weight,
      providing a differentiable *upper bound* on ``min(x)``.

    These two modes are used in the Necessity (□) and Possibility (♢)
    operators to construct *sound* upper/lower bounds that complement
    the ``smooth_min`` / ``smooth_max`` bounds.

    **Exact width.** In the ``z = -x`` mode the gap to the matching lower
    bound is not an estimate but an identity:

    .. math::
        \operatorname{conv\_pool}_\tau(\mathbf{x}, -\mathbf{x})
            - \operatorname{smooth\_min}_\tau(\mathbf{x})
        = \tau\, H\bigl(\operatorname{softmax}(-\mathbf{x}/\tau)\bigr)
        \;\le\; \tau \log n,

    with equality in the upper bound iff every :math:`x_i` ties. Verified
    to 8.9e-16 in float64 over 20k random draws. See
    :func:`box_width_entropy`, which returns this quantity.

    .. warning::
       **This operator is not monotone in** ``x`` **when** ``z = -x``. Its
       derivative is

       .. math::
           \frac{\partial f}{\partial x_k}
           = w_k \left(1 - \frac{x_k - f}{\tau}\right),

       which is **negative** whenever :math:`x_k - f > \tau`: raising a
       term that is already far above the pooled value *lowers* the
       result, because it loses weight faster than it gains value. For
       example at :math:`\tau = 0.1`, going from ``x = [0, 1]`` to
       ``x = [0, 2]`` decreases the pool (4.54e-5 → 4.1e-9), and at
       ``x = [0.3, 0.9]`` the gradients are ``[+1.0123, -0.0123]``.

       This is harmless for soundness — the enclosure holds regardless —
       but it **invalidates the tempting argument** *"the box neuron is
       monotone in* ``A``\\ *, therefore the bound is sound"*. That
       argument is not available. The correct route is monotonicity of
       the **hard** ``min`` together with the one-sided enclosure
       ``smooth_min <= min <= conv_pool``.

    Args:
        x: Values to pool, shape ``(..., N)``.
        z: Logits controlling the convex weights, same shape as ``x``.
            Use ``z = x`` for a lower bound on max, ``z = -x`` for an
            upper bound on min.
        tau: Temperature. Lower values sharpen the weighting toward the
            extreme element. Default 0.1.
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


class _UnsetTau(float):
    """Sentinel for a ``tau`` the caller did not pass.

    Subclasses :class:`float` and carries the historical default value, so
    the signature still type-checks as ``float``, ``inspect.signature``
    still reports ``0.1``, and any code that reads the value is unchanged.
    Only identity (``tau is not _UNSET_TAU``) distinguishes "not passed"
    from an explicit ``tau=0.1``, which is what the deprecation warning
    keys on.
    """

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "0.1"


_UNSET_TAU = _UnsetTau(0.1)


def _as_bounds(
    prop_bounds: Tensor, accessibility: Tensor
) -> tuple[Tensor, bool]:
    """Normalise a proposition argument to ``(..., |W|, 2)`` bounds.

    Point-valued input is detected by *rank relative to the relation*, not by
    an absolute rank: a proposition carries one fewer dimension than its
    accessibility matrix. That rule is exact for both the unbatched case
    (``(|W|,)`` against ``(|W|, |W|)``) and the batched one (``(B, |W|)``
    against ``(B, |W|, |W|)``), and stays unambiguous when ``|W| == 2``,
    where comparing the trailing extent against 2 would not.

    Args:
        prop_bounds: ``(..., |W|, 2)`` bounds or ``(..., |W|)`` point values.
        accessibility: ``(..., |W|, |W|)`` relation.

    Returns:
        ``(bounds, point_valued)``.
    """
    point_valued = prop_bounds.dim() == accessibility.dim() - 1
    if point_valued:
        prop_bounds = prop_bounds.unsqueeze(-1).expand(*prop_bounds.shape, 2)
    return prop_bounds, point_valued


def _select_terms(x: Tensor, top_k: int | None, largest: bool) -> Tensor:
    """Keep the ``top_k`` extreme aggregation terms of each row of ``x``.

    Top-k neighbourhoods must be selected on the quantity that is actually
    aggregated — ``(1 - A) + L`` for □, ``A + U - 1`` for ♢ — and per
    endpoint, never on ``A`` alone: selecting by ``A`` can drop the world
    whose ``L`` (or ``U``) carries the extremum, and the bound then over-
    (or under-) reports it, violating Theorem 1. Selecting on the terms
    themselves keeps the true extremum in the kept set, so the masked
    ``min`` / ``max`` is exact and the smooth aggregations stay within
    ``tau * log(top_k)`` of it. Only the kept terms are aggregated, so the
    result — and its gradient, which reaches exactly the selected entries
    of ``A`` — is independent of ``|W|``.

    Args:
        x: Aggregation terms, shape ``(|W|, |W|)`` (rows = source worlds).
        top_k: Number of terms to keep per row, or ``None`` for all of them.
            A ``top_k >= |W|`` also keeps all of them.
        largest: ``False`` keeps the smallest terms (□), ``True`` the largest
            (♢).

    Returns:
        ``x`` itself when nothing is dropped, else ``(|W|, top_k)``.
    """
    if top_k is None:
        return x
    if top_k < 1:
        raise ValueError(f"top_k must be a positive integer or None, got {top_k}")
    if top_k >= x.shape[-1]:
        return x
    return torch.topk(x, top_k, dim=-1, largest=largest).values


def necessity(
    prop_bounds: Tensor,
    accessibility: Tensor,
    tau: float = 0.1,
    top_k: int | None = None,
    precision: float | None = None,
) -> Tensor:
    r"""Necessity (Box / □) operator — differentiable Kripke semantics.

    Computes truth bounds for □ϕ across all worlds using the weighted
    accessibility matrix. For each world *w*:

    .. math::
        L_{\Box\phi,w} = \operatorname{smooth\_min}_\tau \bigl\{
            (1 - \tilde{A}_{w,w'}) + L_{\phi,w'} \bigr\}_{w' \in W}

    .. math::
        U_{\Box\phi,w} = \operatorname{conv\_pool}_\tau \bigl(
            x_{w'}, \; -x_{w'} \bigr), \quad
            x_{w'} = (1 - \tilde{A}_{w,w'}) + U_{\phi,w'}

    The operator acts as a "weakest link" detector: if a world is highly
    accessible (Ã ≈ 1) but ϕ is false there, the score collapses.

    **Top-k aggregation.** With ``top_k=k`` each endpoint aggregates only
    the *k smallest* of its own implication terms — ``(1 - Ã) + L`` for
    the lower bound, ``(1 - Ã) + U`` for the upper — instead of the full
    row. Because the terms are selected on the aggregated quantity (not on
    ``Ã`` alone), the true minimum is always among the kept terms:
    ``L_□ <= min`` and ``U_□ >= min`` still hold (Theorem 1), the smooth
    lower bound is within ``tau * log(k)`` of the crisp minimum, the result
    does not depend on ``|W|``, and gradients reach exactly the ``k``
    selected entries of ``Ã`` per endpoint. The ``|W| x |W|`` term matrix
    is still formed; the aggregation itself is ``O(k * |W|)``.

    .. warning::
       Do **not** emulate ``top_k`` by zeroing entries of ``Ã`` before the
       call (the ``top_k=`` of the accessibility modules up to 0.2.0). Zeroed
       entries still enter the log-sum-exp with term ``1 + L`` and their
       summed mass drives the bounds to ``[0, 1]`` as ``|W|`` grows, and
       choosing neighbours by ``Ã`` alone is unsound.

    **Accumulated slack under nesting.** Each □ level widens the interval
    by exactly :math:`\tau H(w)` — the entropy of its own softmin weights,
    returned by :func:`box_width_entropy` — bounded by
    :math:`\tau \log n` and maximal when the aggregated terms all tie. The
    cost is therefore *per level* and set by the frame's effective
    branching, not by :math:`|W|` as such. Nesting :math:`k` levels loses
    about :math:`k \tau \bar{H}`, so a lower bound starting at 1 reaches
    the floor at

    .. math::
        k^* = \left\lceil 1 / (\tau \bar{H}) \right\rceil ,

    after which the term is dead: pinned at 0 with no gradient. Measured
    with ``phi = [1, 1]``, ``tau = 0.1``, ``|W| = 8``, lower bound at
    depth 1..6:

    =====================  ==========================================  ===========
    frame                  L at depth 1, 2, 3, 4, 5, 6                 per level
    =====================  ==========================================  ===========
    complete (``A=ones``)  0.792, 0.584, 0.376, 0.168, **0.0**, 0.0    0.2079
    ring bidirectional     0.890, 0.780, 0.670, 0.561, 0.451, 0.341    0.1099
    ring (self + next)     0.931, 0.861, 0.792, 0.723, 0.653, 0.584    0.0694
    =====================  ==========================================  ===========

    The per-level figures are :math:`\tau \log 8`, :math:`\tau \log 3` and
    :math:`\tau \log 2` respectively — the frames' branching factors — and
    each matches :func:`box_width_entropy` to four decimals. The
    degradation is linear and predictable, but it is *not* negligible on a
    densely connected frame: the complete frame above floors at depth 5,
    exactly as :math:`k^*` predicts. Compute the budget rather than
    assuming it, and check deep nests with
    :func:`torchmodal.diagnostics.gradient_health`.


    **Batched input.** A leading batch dimension is accepted on both
    arguments: ``prop_bounds`` of ``(B, |W|, 2)`` against ``accessibility`` of
    ``(B, |W|, |W|)`` returns ``(B, |W|, 2)``, and any number of leading
    dimensions works. Results are bit-identical to looping over the batch.
    Point-valued input is recognised by carrying exactly one dimension fewer
    than the relation, which stays unambiguous even when ``|W| == 2``.

    Args:
        prop_bounds: Truth bounds of shape ``(..., |W|, 2)`` where columns are
            ``[L, U]``, or ``(..., |W|)`` for point-valued truth values
            (treated as both L and U).
        accessibility: Accessibility matrix of shape ``(..., |W|, |W|)``,
            values in [0, 1].
        tau: Temperature. Default 0.1. Ignored when ``precision`` is given.
        top_k: If set, aggregate only the ``top_k`` smallest implication
            terms per world and endpoint. ``None`` (default) aggregates the
            full row. Must be a positive integer.
        precision: Target bracket width, as an alternative to ``tau``: state
            the imprecision you can tolerate and the temperature is chosen by
            :func:`auto_tau` to guarantee it. Overrides ``tau`` when given.

    Returns:
        Tensor of shape ``(..., |W|, 2)`` or ``(..., |W|)`` with necessity
        bounds.
    """
    tau = _resolve_tau(tau, precision, accessibility, top_k)
    prop_bounds, point_valued = _as_bounds(prop_bounds, accessibility)

    L_phi = prop_bounds[..., 0]  # (..., |W|)
    U_phi = prop_bounds[..., 1]  # (..., |W|)

    # (..., |W|, |W|): implication terms per source-target world pair.
    # unsqueeze(-2) broadcasts the target world along the source axis.
    impl_L = (1.0 - accessibility) + L_phi.unsqueeze(-2)
    impl_U = (1.0 - accessibility) + U_phi.unsqueeze(-2)

    # Top-k: keep the k smallest terms of each endpoint (the true minimum is
    # always among them), so the aggregations below see only k terms.
    impl_L = _select_terms(impl_L, top_k, largest=False)
    impl_U = _select_terms(impl_U, top_k, largest=False)

    # Lower bound: smooth_min over target worlds (the last axis)
    L_box = smooth_min(impl_L, tau=tau, dim=-1)

    # Upper bound: conv_pool with the negated implication as the logit (z = -x)
    U_box = conv_pool(impl_U, -impl_U, tau=tau, dim=-1)

    result = torch.stack([L_box, U_box], dim=-1)
    result = torch.clamp(result, 0.0, 1.0)

    if point_valued:
        return result[..., 0]
    return result


def possibility(
    prop_bounds: Tensor,
    accessibility: Tensor,
    tau: float = 0.1,
    top_k: int | None = None,
    precision: float | None = None,
) -> Tensor:
    r"""Possibility (Diamond / ♢) operator — differentiable Kripke semantics.

    Computes truth bounds for ♢ϕ across all worlds. For each world *w*:

    .. math::
        L_{\Diamond\phi,w} = \operatorname{conv\_pool}_\tau \bigl(
            x_{w'}, \; x_{w'} \bigr), \quad
            x_{w'} = \tilde{A}_{w,w'} + L_{\phi,w'} - 1

    .. math::
        U_{\Diamond\phi,w} = \operatorname{smooth\_max}_\tau \bigl\{
            \tilde{A}_{w,w'} + U_{\phi,w'} - 1 \bigr\}_{w' \in W}

    The operator acts as an "evidence scout": it activates if it finds any
    world that is both accessible and where ϕ is true.

    **Top-k aggregation.** With ``top_k=k`` each endpoint aggregates only
    the *k largest* of its own conjunction terms — ``Ã + L - 1`` for the
    lower bound, ``Ã + U - 1`` for the upper — so the true maximum is
    always among the kept terms: ``L_♢ <= max`` and ``U_♢ >= max`` still
    hold, the smooth upper bound is within ``tau * log(k)`` of the crisp
    maximum, nothing depends on ``|W|``, and gradients reach exactly the
    ``k`` selected entries of ``Ã`` per endpoint. See :func:`necessity`
    for why the selection must be made on the aggregated terms rather than
    on ``Ã`` alone.

    **Accumulated slack under nesting.** By the duality
    ``♢ϕ ≡ ¬□¬ϕ`` the ♢ interval widens by the same
    :math:`\tau H(w) \le \tau \log n` per level as □ — see the measured
    table in :func:`necessity`. A nest of ♢ operators therefore drifts
    toward the ceiling at the same rate that a nest of □ operators drifts
    toward the floor.


    **Batched input.** A leading batch dimension is accepted on both
    arguments: ``prop_bounds`` of ``(B, |W|, 2)`` against ``accessibility`` of
    ``(B, |W|, |W|)`` returns ``(B, |W|, 2)``, and any number of leading
    dimensions works. Results are bit-identical to looping over the batch.
    Point-valued input is recognised by carrying exactly one dimension fewer
    than the relation, which stays unambiguous even when ``|W| == 2``.

    Args:
        prop_bounds: Truth bounds of shape ``(..., |W|, 2)`` or
            ``(..., |W|)``.
        accessibility: Accessibility matrix ``(..., |W|, |W|)`` in [0, 1].
        tau: Temperature. Default 0.1. Ignored when ``precision`` is given.
        top_k: If set, aggregate only the ``top_k`` largest conjunction
            terms per world and endpoint. ``None`` (default) aggregates the
            full row. Must be a positive integer.
        precision: Target bracket width, as an alternative to ``tau``. See
            :func:`auto_tau`.

    Returns:
        Tensor of shape ``(..., |W|, 2)`` or ``(..., |W|)`` with possibility
        bounds.
    """
    tau = _resolve_tau(tau, precision, accessibility, top_k)
    prop_bounds, point_valued = _as_bounds(prop_bounds, accessibility)

    L_phi = prop_bounds[..., 0]
    U_phi = prop_bounds[..., 1]

    # conjunction terms; unsqueeze(-2) broadcasts the target world
    conj_L = accessibility + L_phi.unsqueeze(-2) - 1.0
    conj_U = accessibility + U_phi.unsqueeze(-2) - 1.0

    # Top-k: keep the k largest terms of each endpoint (the true maximum is
    # always among them).
    conj_L = _select_terms(conj_L, top_k, largest=True)
    conj_U = _select_terms(conj_U, top_k, largest=True)

    # Lower bound: conv_pool with the conjunction as both value and logit (z = x)
    L_dia = conv_pool(conj_L, conj_L, tau=tau, dim=-1)

    # Upper bound: smooth_max (weighted existential)
    U_dia = smooth_max(conj_U, tau=tau, dim=-1)

    result = torch.stack([L_dia, U_dia], dim=-1)
    result = torch.clamp(result, 0.0, 1.0)

    if point_valued:
        return result[..., 1]  # for point values return upper (existential)
    return result


def auto_tau(
    accessibility: Tensor,
    target_width: float,
    prop_bounds: Tensor | None = None,
    top_k: int | None = None,
    tol: float = 1e-9,
    max_iter: int = 80,
) -> float:
    r"""Temperature achieving a target bracket width — the inverse of the gap.

    Every other entry point in this module asks for a temperature and tells
    you, afterwards, how wide the resulting bracket is. This inverts that:
    state the imprecision you can tolerate, and get the ``tau`` that delivers
    it.

    **Which direction it errs.** The returned temperature is always *safe* —
    the realised width is at most ``target_width``, never more — so a bound
    computed at this temperature encloses the crisp value to within the
    requested tolerance.

    Two modes:

    - **Closed form** (``prop_bounds=None``). Uses the frame-only bound
      :math:`\tau H(w) \le \tau \log n`, giving

      .. math:: \tau = \varepsilon / \log n,

      with ``n`` the number of aggregated terms (``top_k``, else ``|W|``).
      This holds for *any* proposition, so it is the temperature to use when
      the bounds are not yet known — during training, for instance, where they
      change every step. It is conservative: since :math:`H \le \log n` with
      equality only when every term ties, the realised width is usually well
      under target.

    - **Exact** (``prop_bounds`` supplied). Bisects on the true
      :func:`box_width_entropy` for those bounds, returning the **largest**
      ``tau`` whose worst-case per-world width still meets the target. This is
      tighter — often by a wide margin on a non-uniform frame — and a larger
      ``tau`` means better-conditioned gradients, so prefer it whenever the
      bounds are available.

    .. note::
       The width is monotone non-decreasing in ``tau`` (it vanishes as
       :math:`\tau \to 0`, where the softmin weights concentrate on a single
       term, and grows to :math:`\tau \log n` as the weights flatten), which
       is what makes the bisection well posed.

    Args:
        accessibility: Accessibility matrix ``(..., |W|, |W|)``.
        target_width: The bracket width to achieve, in truth units. Must be
            positive.
        prop_bounds: Optional ``(..., |W|, 2)`` bounds (or ``(..., |W|)``
            point values). When given, the exact mode is used.
        top_k: Match the ``top_k`` of the operator being configured, so the
            term count agrees. Default ``None``.
        tol: Bisection tolerance on ``tau``. Default 1e-9.
        max_iter: Maximum bisection steps. Default 80.

    Returns:
        A temperature, as a Python float.

    Raises:
        ValueError: If ``target_width`` is not positive.

    Example:
        >>> import torch
        >>> from torchmodal.functional import auto_tau, box_width_entropy
        >>> A = torch.ones(10, 10)
        >>> tau = auto_tau(A, target_width=0.05)
        >>> bool(box_width_entropy(A, torch.full((10, 2), 0.5),
        ...                        tau=tau).max() <= 0.05 + 1e-9)
        True
    """
    if target_width <= 0:
        raise ValueError(
            f"target_width must be positive, got {target_width}"
        )

    n = accessibility.shape[-1]
    if top_k is not None:
        n = min(top_k, n)
    if n <= 1:
        # A single aggregated term has zero entropy at any temperature, so no
        # temperature is excluded; return the closed-form value for n = 2 as a
        # finite, well-conditioned default rather than an unbounded one.
        return float(target_width / math.log(2))

    closed_form = float(target_width / math.log(n))
    if prop_bounds is None:
        return closed_form

    def width(t: float) -> float:
        return float(
            box_width_entropy(
                accessibility, prop_bounds, tau=t, top_k=top_k
            ).max()
        )

    # The closed form is a guaranteed-safe lower bracket; grow an upper one
    # until it violates, then bisect between them.
    lo = closed_form
    hi = closed_form
    for _ in range(max_iter):
        if width(hi * 2.0) > target_width:
            break
        hi *= 2.0
        lo = hi
    else:  # pragma: no cover - width stayed under target throughout
        return hi
    hi *= 2.0

    for _ in range(max_iter):
        if hi - lo <= tol:
            break
        mid = 0.5 * (lo + hi)
        if width(mid) <= target_width:
            lo = mid
        else:
            hi = mid
    return lo


def _resolve_tau(
    tau: float,
    precision: float | None,
    accessibility: Tensor,
    top_k: int | None,
) -> float:
    """Pick the temperature from either ``tau`` or ``precision``.

    ``precision`` is the inverse spelling of ``tau``: a caller states the
    bracket width they can tolerate and :func:`auto_tau` supplies the
    temperature. The two are mutually exclusive.
    """
    if precision is None:
        return tau
    return auto_tau(accessibility, precision, top_k=top_k)


def box_width_entropy(
    accessibility: Tensor,
    prop_bounds: Tensor,
    tau: float = 0.1,
    top_k: int | None = None,
) -> Tensor:
    r"""Per-world interval width contributed by one :func:`necessity` level.

    The gap between the two endpoints of a □ neuron is not a bound — it is
    an identity. For a common term vector :math:`\mathbf{x}`,

    .. math::
        \operatorname{conv\_pool}_\tau(\mathbf{x}, -\mathbf{x})
            - \operatorname{smooth\_min}_\tau(\mathbf{x})
        = \tau\, H\bigl(\operatorname{softmax}(-\mathbf{x}/\tau)\bigr),

    where :math:`H` is the Shannon entropy in nats. This function returns
    the right-hand side per world, evaluated on the □ implication terms
    :math:`x_{w,w'} = (1 - \tilde{A}_{w,w'}) + U_{\phi,w'}`.

    **What it bounds.** This is an *exact equality*, not a bound: it is the
    width that one modal level adds, so ``U_□ - L_□`` decomposes as

    .. math::
        \underbrace{\tau H(w)}_{\text{this function}}
        \;+\;
        \underbrace{
          \operatorname{smooth\_min}_\tau(\mathbf{x}_U)
          - \operatorname{smooth\_min}_\tau(\mathbf{x}_L)
        }_{\text{incoming width, propagated}} .

    When ``prop_bounds`` is point-valued (or ``L == U``) the second term
    vanishes and the return value equals ``U_□ - L_□`` exactly — *provided
    the □ output clamp does not engage*. :func:`necessity` clamps its
    result into [0, 1]; where a raw endpoint falls outside that range the
    clamp truncates the interval and the measured width is smaller than
    the entropy. Verified to 3.9e-16 in float64 over the unclamped regime.

    **Why it is useful.** The quantity is bounded by :math:`\tau \log n`
    (:math:`n` = number of aggregated terms, i.e. ``top_k`` or ``|W|``),
    with equality iff every term ties. It therefore:

    - turns the faithful-nesting depth ceiling into a computed quantity,
      :math:`k^* = \varepsilon / (\tau \bar{H})`, rather than a guess;
    - gives each :math:`\square` a cheap tightness diagnostic — a large
      value means the frame is near-uniform and the bound is loose;
    - is *exactly* the dead zone of :func:`contradiction` applied after a
      □ neuron: a bound crossing smaller than this width is absorbed and
      produces neither loss nor gradient.

    Args:
        accessibility: Accessibility matrix ``(..., |W|, |W|)`` in [0, 1].
        prop_bounds: Truth bounds ``(..., |W|, 2)`` as ``[L, U]``, or
            ``(..., |W|)`` for point-valued truth values. A leading batch
            dimension is accepted, as on :func:`necessity`.
        tau: Temperature. Must match the ``tau`` of the □ level being
            diagnosed. Default 0.1.
        top_k: Match the ``top_k`` of the □ level being diagnosed, so the
            entropy is taken over the same kept terms. Default ``None``.

    Returns:
        Tensor of shape ``(..., |W|)``: the width, in truth units, that this
        □ level contributes at each source world.

    Example:
        >>> import torch
        >>> from torchmodal.functional import box_width_entropy, necessity
        >>> A = torch.ones(6, 6)
        >>> b = torch.full((6, 2), 0.5)          # point-valued: L == U
        >>> w = box_width_entropy(A, b, tau=0.1)
        >>> box = necessity(b, A, tau=0.1)
        >>> bool(torch.allclose(w, box[:, 1] - box[:, 0], atol=1e-6))
        True
    """
    prop_bounds, _ = _as_bounds(prop_bounds, accessibility)
    U_phi = prop_bounds[..., 1]

    terms = (1.0 - accessibility) + U_phi.unsqueeze(-2)
    terms = _select_terms(terms, top_k, largest=False)

    # Computed from log-softmax rather than softmax-then-log: at small tau the
    # smallest weights underflow to exactly 0, and 0 * log(0) is NaN. A
    # clamp_min floor does not rescue this in float32, where any floor below
    # ~1e-38 is itself flushed to zero. Masking the zero-weight terms — which
    # contribute 0 to the entropy in the limit — is exact and dtype-agnostic.
    log_w = torch.log_softmax(-terms / tau, dim=-1)
    weights = log_w.exp()
    plogp = torch.where(
        weights > 0, weights * log_w, torch.zeros_like(weights)
    )
    entropy = -plogp.sum(dim=-1)
    # Entropy is non-negative by definition; clamp away the -0.0 / tiny
    # negative values float arithmetic produces in the degenerate
    # single-term case (top_k=1), where the exact answer is 0.
    return (tau * entropy).clamp_min(0.0)


def until(
    phi_bounds: Tensor,
    psi_bounds: Tensor,
    accessibility: Tensor,
    tau: float = _UNSET_TAU,
) -> Tensor:
    r"""Until (U) operator — differentiable temporal semantics.

    Computes truth bounds for ``ϕ U ψ`` ("ϕ holds until ψ becomes true")
    over a forward-time accessibility structure.  For each time step *t*:

    .. math::
        (\phi\;\mathcal{U}\;\psi)_t = \bigvee_{t' \geq t}
            \Bigl(\psi_{t'} \;\wedge\; \bigwedge_{t \leq s < t'} \phi_s\Bigr)

    The implementation uses a backward dynamic-programming sweep that
    remains fully differentiable:

    .. math::
        U_t = \psi_t \;\lor\; (\phi_t \;\land\; U_{t+1})

    with ``U_T = ψ_T`` at the final time step.  All connectives use
    Łukasiewicz fuzzy logic (see :func:`conjunction`, :func:`disjunction`)
    so that the computation stays in [0, 1] and gradients flow smoothly.

    This closes the expressiveness gap with STLCG (Leung et al., 2023)
    which supports the Until operator for signal temporal logic.

    .. warning::
       **This operator is inert with respect to its relation.** Only
       ``accessibility.shape[0]`` is read, to obtain ``T``; no aggregation
       over ``accessibility`` takes place and *no autograd path runs from
       the result back to it*. ``until(phi, psi, A)`` is bit-identical for
       ``A = triu(ones)`` and ``A = zeros``, and deleting an edge of the
       chain does not change the output. The operator is correct for a
       **total order** — consecutive time steps — and only for that. For
       an arbitrary or learned relation, where "is there still a path?" is
       the question, use :func:`until_graph`, whose bounds do depend on
       the relation and do carry gradient into it.

    .. warning::
       Its Łukasiewicz backward sweep **floors the lower bound**. Each
       step costs ``1 - L_phi``: over a 6-step chain with ``L_phi = 0.9``
       and ψ true only at the end, the lower bounds are
       ``0.5, 0.6, 0.7, 0.8, 0.9, 1.0``. That decay is the operator, not
       the data, and for a long enough horizon the lower bound reaches 0
       with no gradient. :func:`until_graph` uses idempotent Gödel
       connectives instead and does not floor.

    Args:
        phi_bounds: Truth bounds for ϕ, shape ``(T, 2)`` or ``(T,)``.
        psi_bounds: Truth bounds for ψ, shape ``(T, 2)`` or ``(T,)``.
        accessibility: Forward-time accessibility matrix ``(T, T)``.
            **Used only for its size.** Only the temporal ordering
            matters; the matrix determines the number of time steps. No
            aggregation over ``accessibility`` takes place, so there is no
            ``top_k`` parameter here (see :func:`necessity` /
            :func:`possibility`).
        tau: **Deprecated and unused.** The DP formulation has no smooth
            aggregation, so no temperature enters it; passing this
            argument raises a :class:`DeprecationWarning`. Kept only for
            API compatibility and scheduled for removal in 0.4.0.

    Returns:
        Truth bounds for ``ϕ U ψ``, same shape as inputs.
    """
    if tau is not _UNSET_TAU:
        warnings.warn(
            "torchmodal.functional.until's `tau` argument is unused: the "
            "backward DP has no smooth aggregation, so the result is "
            "identical for every value. It will be removed in 0.4.0. If "
            "you wanted a temperature-controlled, relation-aware Until, "
            "use until_graph.",
            DeprecationWarning,
            stacklevel=2,
        )

    point_valued = phi_bounds.dim() == 1
    if point_valued:
        phi_bounds = phi_bounds.unsqueeze(-1).expand(-1, 2)
        psi_bounds = psi_bounds.unsqueeze(-1).expand(-1, 2)

    T = phi_bounds.shape[0]

    L_phi, U_phi = phi_bounds[:, 0], phi_bounds[:, 1]
    L_psi, U_psi = psi_bounds[:, 0], psi_bounds[:, 1]

    # Build results as lists to avoid in-place ops (autograd-safe)
    L_list: list[Tensor] = [torch.tensor(0.0)] * T
    U_list: list[Tensor] = [torch.tensor(0.0)] * T

    # Base case: at the last step, Until reduces to ψ
    L_list[T - 1] = L_psi[T - 1]
    U_list[T - 1] = U_psi[T - 1]

    # Backward sweep: U_t = ψ_t ∨ (ϕ_t ∧ U_{t+1})
    for t in range(T - 2, -1, -1):
        # ϕ_t ∧ U_{t+1}  (Łukasiewicz conjunction)
        L_continue = torch.clamp(L_phi[t] + L_list[t + 1] - 1.0, min=0.0)
        U_continue = torch.min(U_phi[t], U_list[t + 1])

        # ψ_t ∨ (ϕ_t ∧ U_{t+1})  (Łukasiewicz disjunction)
        L_list[t] = torch.max(L_psi[t], L_continue)
        U_list[t] = torch.clamp(U_psi[t] + U_continue, max=1.0)

    result = torch.stack(
        [torch.stack(L_list), torch.stack(U_list)], dim=-1
    )

    if point_valued:
        return result[:, 0]
    return result


def until_graph(
    phi_bounds: Tensor,
    psi_bounds: Tensor,
    accessibility: Tensor,
    tau: float = 0.1,
    tau_decay: float = 0.5,
    max_iter: int = 50,
    tol: float = 1e-6,
    quantifier: str = "diamond",
) -> Tensor:
    r"""Relation-aware Until — least fixpoint of ``U = ψ ∨ (φ ∧ ♢U)``.

    Unlike :func:`until`, which reads its ``accessibility`` only for its
    size, this operator **aggregates over the relation**: its bounds
    depend on which edges exist and gradients flow back into them. That
    makes it the operator to use with a learned or arbitrary (cyclic,
    branching, disconnected) Kripke frame, where the question "is there
    still a path along which ϕ holds until ψ?" has a non-trivial answer.

    The fixpoint is reached by iterating

    .. math::
        U^{(0)} = \psi, \qquad
        U^{(j+1)} = \psi \;\vee_G\;
            \bigl(\phi \;\wedge_G\; \Diamond_{\tau_j} U^{(j)}\bigr)

    to convergence, where :math:`\vee_G` and :math:`\wedge_G` are the
    **Gödel** connectives (``max`` and ``min``). Gödel is used rather than
    Łukasiewicz because it is *idempotent*: the iteration therefore has a
    genuine fixpoint instead of decaying by ``1 - L_phi`` per step the way
    :func:`until` does, so the lower bound does not floor. The sequence is
    monotone non-decreasing and bounded above by 1, so it converges.

    The temperature is **annealed** across sweeps, :math:`\tau_j = \tau
    \rho^j` with :math:`\rho` = ``tau_decay``, so successive modal steps
    are progressively sharper and the slack accumulated over the whole
    iteration is a geometric series rather than a growing one.

    **What it bounds, and the gap.** The returned ``[L, U]`` brackets the
    crisp (``τ → 0``, Boolean-relation) value of ``ϕ U ψ`` evaluated over
    the same frame: ``L <= crisp <= U``. The connectives are exact — Gödel
    ``min`` / ``max`` introduce no error — so the only relaxation is the
    modal ♢ step, which contributes at most :math:`\tau_j \log |W|` per
    sweep on each endpoint. Summed over the annealed schedule the total
    gap is bounded by

    .. math::
        \frac{\tau \log |W|}{1 - \rho},

    independent of the number of iterations. Setting ``tau_decay=1.0``
    disables annealing and the gap grows linearly in the sweep count
    instead.

    .. warning::
       ``quantifier="box"`` computes **AU** ("along *every* path") and is
       sound **only on a serial relation** — one where every world has at
       least one successor. At a dead end ``□U`` is vacuously 1, so a path
       that merely stops satisfies the formula. This is measured, not
       hypothetical: on a 6-step chain the box variant returns 0.9
       everywhere regardless of connectivity. Use ``"diamond"`` (**EU**,
       "there is a path") unless the frame is known to be serial.

    Args:
        phi_bounds: Truth bounds for ϕ, shape ``(|W|, 2)`` as ``[L, U]``,
            or ``(|W|,)`` for point-valued truth values.
        psi_bounds: Truth bounds for ψ, same shape as ``phi_bounds``.
        accessibility: Accessibility matrix ``(|W|, |W|)`` in [0, 1]. Read
            as a relation, not merely for its size.
        tau: Initial temperature for the modal step. Default 0.1.
        tau_decay: Geometric annealing factor :math:`\rho \in (0, 1]`
            applied per sweep. ``1.0`` disables annealing. Default 0.5.
        max_iter: Maximum fixpoint sweeps. Default 50.
        tol: Stop once the largest bound change in a sweep falls below
            this. Default 1e-6.
        quantifier: ``"diamond"`` for EU (default, sound on any frame) or
            ``"box"`` for AU (sound only on a serial frame — see the
            warning above).

    Returns:
        Truth bounds for ``ϕ U ψ``, same shape as the inputs.

    Raises:
        ValueError: If ``quantifier`` is not ``"diamond"`` or ``"box"``,
            or if ``tau_decay`` is outside ``(0, 1]``.

    Example:
        >>> import torch
        >>> from torchmodal.functional import until_graph
        >>> T = 6
        >>> A = torch.zeros(T, T)
        >>> A[torch.arange(T - 1), torch.arange(1, T)] = 1.0  # a chain
        >>> phi = torch.stack([torch.full((T,), 0.9), torch.ones(T)], -1)
        >>> psi = torch.zeros(T, 2)
        >>> psi[T - 1] = 1.0
        >>> round(until_graph(phi, psi, A)[0, 0].item(), 3)
        0.9
        >>> A[2, 3] = 0.0  # cut the path
        >>> round(until_graph(phi, psi, A)[0, 0].item(), 3)
        0.0
    """
    if quantifier not in ("diamond", "box"):
        raise ValueError(
            f"quantifier must be 'diamond' (EU) or 'box' (AU), "
            f"got {quantifier!r}"
        )
    if not 0.0 < tau_decay <= 1.0:
        raise ValueError(
            f"tau_decay must lie in (0, 1], got {tau_decay}"
        )

    point_valued = phi_bounds.dim() == 1
    if point_valued:
        phi_b = phi_bounds.unsqueeze(-1).expand(-1, 2)
        psi_b = psi_bounds.unsqueeze(-1).expand(-1, 2)
    else:
        phi_b = phi_bounds
        psi_b = psi_bounds

    modal = necessity if quantifier == "box" else possibility

    current = psi_b
    tau_j = tau
    for _ in range(max_iter):
        modal_b = modal(current, accessibility, tau=tau_j)

        # Gödel conjunction: elementwise min on both endpoints.
        cont = torch.minimum(phi_b, modal_b)
        # Gödel disjunction: elementwise max on both endpoints.
        nxt = torch.maximum(psi_b, cont)
        nxt = torch.clamp(nxt, 0.0, 1.0)

        # detach: the convergence test is control flow, not part of the
        # graph, and reading a grad-tracking tensor as a Python float
        # otherwise warns.
        delta = (nxt - current).detach().abs().max()
        current = nxt
        if float(delta) < tol:
            break
        tau_j *= tau_decay

    if point_valued:
        return current[:, 0]
    return current


# ---------------------------------------------------------------------------
# Contradiction measure
# ---------------------------------------------------------------------------


def contradiction(bounds: Tensor, upper: Tensor | None = None) -> Tensor:
    r"""Compute contradiction loss from truth bounds.

    A contradiction arises when a lower bound exceeds an upper bound, an
    inconsistency no classical truth assignment can satisfy:

    .. math::
        \mathcal{L}_{\text{contra}} = \sum \max(0,\; L - U).

    Two equivalent call forms are accepted:

    - **Stacked** — ``contradiction(bounds)`` with ``bounds`` of shape
      ``(..., 2)`` holding ``[L, U]`` on the last dimension. This is the
      bound contradiction ``ReLU(L_phi - U_phi)``.
    - **Split** — ``contradiction(L, U)`` with the lower and upper sources
      passed as separate tensors of matching shape, for when they are
      computed apart. For example ``contradiction(box, dia)`` penalises a
      necessity that exceeds its possibility (the modal
      ``Box phi -> Diamond phi`` consistency requirement).

    The two forms agree by construction::

        contradiction(L, U) == contradiction(torch.stack([L, U], dim=-1))

    .. warning::
       **Dead zone after a modal neuron.** Applied to the output of a □
       neuron this loss is *identically zero, with zero gradient*, until
       the underlying bound crossing exceeds the box width
       :func:`box_width_entropy` — that is, :math:`\tau H(w)`, at most
       :math:`\tau \log n`. The modal level widens the interval by
       exactly that much, so any smaller crossing is absorbed before it
       reaches the loss.

       The correspondence is exact, not approximate. Driving a crossing
       ``c`` through a complete frame at ``tau = 0.1``:

       ===========  ====================  ==================
       fan-in *n*   dead zone (measured)  :math:`\tau\log n`
       ===========  ====================  ==================
       3            0.109861              0.109861
       6            0.179176              0.179176
       10           0.230259              0.230259
       ===========  ====================  ==================

       Past the edge the loss is linear with unit slope per world.

       **Consequence:** ``L_contra`` must not be the *sole* guard against
       a degenerate optimum. A model can sit in a mildly contradictory
       state indefinitely, paying nothing and receiving no gradient to
       leave it. Either anneal ``tau`` downward (shrinking the dead zone
       toward 0), check the raw pre-modal bounds as well, or pair the
       loss with :func:`torchmodal.diagnostics.gradient_health`.

    Args:
        bounds: Either a ``(..., 2)`` bound tensor (stacked form), or the
            lower-bound tensor (split form, when ``upper`` is provided).
        upper: Upper-bound tensor matching ``bounds``. If omitted,
            ``bounds`` is read as a stacked ``[L, U]`` pair.

    Returns:
        Scalar contradiction loss, summed over all elements.
    """
    if upper is None:
        L = bounds[..., 0]
        U = bounds[..., 1]
    else:
        L = bounds
        U = upper
    return torch.relu(L - U).sum()


# ---------------------------------------------------------------------------
# Dynamic epistemic logic: model-update operators
# ---------------------------------------------------------------------------


def announce(
    accessibility: Tensor,
    psi_bounds: Tensor,
    trust: Tensor | float = 1.0,
    tnorm: str = "product",
) -> tuple[Tensor, Tensor]:
    r"""Graded, trust-weighted public announcement of :math:`\psi`.

    A public announcement is the dynamic-epistemic-logic update that *edits the
    model*: crisp public announcement logic relativises the relation to the
    worlds where :math:`\psi` holds. This is its graded, interval-valued
    counterpart, returning the two relations that bracket it:

    .. math::
        A^{lo}_{w,w'} &= A_{w,w'} \otimes (1 - t \cdot (1 - U_{\psi,w'})) \\
        A^{hi}_{w,w'} &= A_{w,w'} \otimes (1 - t \cdot (1 - L_{\psi,w'}))

    :math:`A^{lo}` cuts only the worlds that are *certainly* :math:`\neg\psi`,
    so it is the **largest** surviving relation; :math:`A^{hi}` cuts every world
    not *certainly* :math:`\psi`, so it is the **smallest**.

    **What it bounds.** Since :math:`L_\psi \le V_\psi \le U_\psi`, the crisp
    relativised relation is sandwiched pointwise:

    .. math:: A^{hi} \le A^{crisp} \le A^{lo}.

    That sandwich is what makes :func:`necessity_after` interval-sound; it is
    verified over random graded inputs in the test-suite. With ``trust=1`` and a
    crisp :math:`\psi` the pair collapses onto crisp PAL relativisation, and
    ``trust=0`` returns the relation unchanged.

    .. warning::
        Worlds cut by the announcement become **dead ends**, where
        :math:`\square` is vacuously true, whereas crisp PAL *deletes* them from
        the model. The two therefore disagree at removed worlds by construction
        — a cut world reports :math:`K \approx [1, 1]`. Restrict any comparison
        against a crisp checker to the surviving worlds, and check that the
        actual world survives.

    Args:
        accessibility: ``(|W|, |W|)`` or ``(|G|, |W|, |W|)`` relation in [0, 1].
        psi_bounds: ``(|W|, 2)`` or ``(|W|,)`` bounds of the announced formula.
        trust: How far the announcement cuts, in [0, 1]. Scalar, ``(|G|,)`` or
            ``(|G|, |W|)`` for recipient-specific trust in the sender.
            Default 1.0 (a fully trusted announcement).
        tnorm: Conjunction combining the relation with the survival factor —
            ``"product"`` (default), ``"godel"`` (min) or ``"luk"``.

    Returns:
        ``(A_lo, A_hi)``, each the shape of ``accessibility``.

    Example:
        >>> import torch
        >>> from torchmodal.functional import announce
        >>> A = torch.ones(3, 3)
        >>> psi = torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
        >>> lo, hi = announce(A, psi)
        >>> bool((lo[:, 1] == 0).all() and (hi[:, 1] == 0).all())
        True
    """
    if psi_bounds.dim() == 1:
        psi_bounds = torch.stack([psi_bounds, psi_bounds], dim=-1)
    L_psi, U_psi = psi_bounds[..., 0], psi_bounds[..., 1]

    t = torch.as_tensor(trust, dtype=accessibility.dtype, device=accessibility.device)
    while t.dim() < accessibility.dim() - 1:
        t = t.unsqueeze(-1)
    t = t.unsqueeze(-1) if t.dim() == accessibility.dim() - 1 else t

    keep_lo = 1.0 - t * (1.0 - U_psi)
    keep_hi = 1.0 - t * (1.0 - L_psi)

    def _combine(A: Tensor, keep: Tensor) -> Tensor:
        keep = keep.expand_as(A) if keep.dim() == A.dim() else keep
        if tnorm == "product":
            return A * keep
        if tnorm == "godel":
            return torch.minimum(A, keep.expand_as(A))
        if tnorm == "luk":
            return torch.clamp(A + keep.expand_as(A) - 1.0, min=0.0)
        raise ValueError(f"tnorm must be 'product', 'godel' or 'luk', got {tnorm!r}")

    return _combine(accessibility, keep_lo), _combine(accessibility, keep_hi)


def necessity_after(
    prop_bounds: Tensor,
    accessibility: Tensor,
    psi_bounds: Tensor,
    trust: Tensor | float = 1.0,
    tau: float = 0.1,
    tnorm: str = "product",
    top_k: int | None = None,
) -> Tensor:
    r"""Knowledge after an announcement: :math:`[\psi] \square \varphi`.

    The lower endpoint is taken through :math:`A^{lo}` and the upper through
    :math:`A^{hi}` (see :func:`announce`). The direction is not arbitrary: a
    *larger* relation constrains :math:`\square` more, so the largest surviving
    relation gives the lower bound.

    **What it bounds.** Sound in both directions —
    :math:`L \le \square\varphi^{crisp} \le U` on surviving worlds — reducing
    to crisp public-announcement logic as ``tau -> 0`` with ``trust=1``.

    The proof route matters, because the obvious one is wrong: it is *not*
    monotonicity of the box neuron, whose upper endpoint uses
    :func:`conv_pool` and is not monotone. It is monotonicity of the **hard**
    ``min`` under the pointwise sandwich of :func:`announce`, composed with the
    one-sided enclosure of the aggregators.

    Args:
        prop_bounds: ``(|W|, 2)`` or ``(|W|,)`` bounds for φ.
        accessibility: ``(|W|, |W|)`` relation in [0, 1].
        psi_bounds: ``(|W|, 2)`` or ``(|W|,)`` bounds for the announced ψ.
        trust: See :func:`announce`. Default 1.0.
        tau: Temperature. Default 0.1.
        tnorm: See :func:`announce`. Default ``"product"``.
        top_k: Passed through to :func:`necessity`.

    Returns:
        ``(|W|, 2)`` bounds for φ known after the announcement.
    """
    A_lo, A_hi = announce(accessibility, psi_bounds, trust=trust, tnorm=tnorm)
    lower = necessity(prop_bounds, A_lo, tau=tau, top_k=top_k)[..., 0]
    upper = necessity(prop_bounds, A_hi, tau=tau, top_k=top_k)[..., 1]
    return torch.stack([lower, upper], dim=-1)


def group_announce(
    accessibility: Tensor,
    psi_bounds: Tensor,
    recipients: Tensor,
    trust: Tensor | float = 1.0,
    tnorm: str = "product",
) -> tuple[Tensor, Tensor]:
    r"""Action-model update for a message delivered to part of the group.

    A public announcement reaches everyone; a message on a private or group
    channel does not. Recipients update their relation by :func:`announce`,
    non-recipients keep theirs unchanged — the graded counterpart of a
    product update with two events, "heard" and "did not hear".

    **Why this is not a public announcement.** Common knowledge is created only
    by an event public to the whole group. After a group announcement the
    recipients' knowledge rises while the others' does not, so the group's
    :math:`C_G` need not move at all — which is exactly what makes
    who-hears-what a real design question rather than a formality.

    Args:
        accessibility: ``(|G|, |W|, |W|)`` — one relation per agent.
        psi_bounds: ``(|W|, 2)`` or ``(|W|,)`` bounds of the announced formula.
        recipients: ``(|G|,)`` mask in [0, 1]; 1 means the agent received the
            message. Fractional values interpolate (a partially attentive
            listener) by scaling that agent's effective trust.
        trust: See :func:`announce`. Default 1.0.
        tnorm: See :func:`announce`. Default ``"product"``.

    Returns:
        ``(A_lo, A_hi)``, each ``(|G|, |W|, |W|)``.
    """
    if accessibility.dim() != 3:
        raise ValueError(
            "group_announce expects (|G|, |W|, |W|); use announce() for one relation"
        )
    t = torch.as_tensor(trust, dtype=accessibility.dtype, device=accessibility.device)
    effective = recipients.to(accessibility.dtype) * t
    return announce(accessibility, psi_bounds, trust=effective, tnorm=tnorm)
