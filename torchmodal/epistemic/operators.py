"""Multi-agent group-knowledge operators over several accessibility relations.

Single-agent knowledge is :func:`torchmodal.functional.necessity` routed
through that agent's relation. This module adds the *group* operators that
epistemic logic needs and that a single relation cannot express:

- :func:`everybody_knows` — :math:`E_G\\varphi = \\bigwedge_{a \\in G} K_a\\varphi`
- :func:`mutual_knowledge` — the bounded tower :math:`E_G^k\\varphi`
- :func:`distributed_knowledge` — :math:`D_G\\varphi`, knowledge of the pooled relation
- :func:`common_knowledge` — :math:`C_G\\varphi`, the greatest fixpoint

**Read the temperature warning on** :func:`common_knowledge` **before using it.**
Every modal level costs at least the box width
:func:`~torchmodal.functional.box_width_entropy`, so an operator that iterates
to convergence drives its lower bound to exactly zero. That is a property of
graded modal logic, not a bug, but it decides which operator belongs in a loss.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import torch
from torch import Tensor

from torchmodal import functional as F

__all__ = [
    "and_bounds",
    "everybody_knows",
    "mutual_knowledge",
    "distributed_knowledge",
    "common_knowledge",
    "pooled_accessibility",
]

Group = Optional[Sequence[int]]
TauSchedule = Optional[Union[float, Callable[[int], float]]]


def _as_bounds(x: Tensor) -> Tensor:
    """Accept ``(N,)`` point values or ``(N, 2)`` bounds; return ``(N, 2)``."""
    if x.dim() == 1:
        return torch.stack([x, x], dim=-1)
    return x


def _group_index(accessibilities: Tensor, group: Group) -> Tensor:
    if group is None:
        return torch.arange(accessibilities.shape[0], device=accessibilities.device)
    return torch.as_tensor(list(group), dtype=torch.long, device=accessibilities.device)


def _tau_at(schedule: TauSchedule, tau: float, step: int) -> float:
    """Resolve the temperature for iteration ``step``.

    ``None`` keeps ``tau`` fixed, a float is a geometric decay factor
    (``tau * schedule ** step``), and a callable is applied to ``step``.
    """
    if schedule is None:
        return tau
    if callable(schedule):
        return float(schedule(step))
    return float(tau * (float(schedule) ** step))


def and_bounds(bounds: Tensor, dim: int = 0, tnorm: str = "godel") -> Tensor:
    r"""n-ary conjunction over a stack of ``[L, U]`` bounds.

    The upper bound is always :math:`U_\wedge = \min_i U_i`. The lower bound
    uses the chosen t-norm:

    - ``"godel"`` (default, min): :math:`L_\wedge = \min_i L_i`
    - ``"product"``: :math:`L_\wedge = \prod_i L_i`
    - ``"luk"`` (Łukasiewicz): :math:`L_\wedge = \max(0, \sum_i L_i - (n-1))`

    **What it bounds.** All three are exact logical AND on crisp ``{0, 1}``
    inputs and are sound lower bounds on the crisp conjunction, ordered
    :math:`L_{\mathrm{luk}} \le L_{\mathrm{prod}} \le L_{\mathrm{godel}}`.

    **Why Gödel is the default here.** It is both the tightest of the three and
    the only *idempotent* one. Łukasiewicz is sub-idempotent
    (:math:`L \wedge L < L` for :math:`L \in (0,1)`) and amplifies each term's
    deficit by ``n``, so folding a group of 6 agents at ``L = 0.93`` yields
    0.58 rather than 0.93 — and an *iterated* fold, as in
    :func:`mutual_knowledge`, reaches exactly zero at the second level and stays
    there with no gradient. Choose ``"luk"`` only for a single, non-iterated
    read-out where the extra looseness is wanted.

    Args:
        bounds: Stack of bounds with ``[L, U]`` on the last axis.
        dim: Axis to fold over. Default 0.
        tnorm: ``"godel"`` | ``"product"`` | ``"luk"``. Default ``"godel"``.

    Returns:
        Bounds with ``dim`` reduced.
    """
    lower, upper = bounds[..., 0], bounds[..., 1]
    if tnorm == "godel":
        new_lower = lower.min(dim=dim).values
    elif tnorm == "product":
        new_lower = lower.prod(dim=dim)
    elif tnorm == "luk":
        n = bounds.shape[dim]
        new_lower = torch.clamp(lower.sum(dim=dim) - (n - 1), min=0.0)
    else:  # pragma: no cover - guarded by the raise
        raise ValueError(
            f"tnorm must be 'godel', 'product' or 'luk', got {tnorm!r}"
        )
    return torch.stack([new_lower, upper.min(dim=dim).values], dim=-1)


def everybody_knows(
    prop_bounds: Tensor,
    accessibilities: Tensor,
    group: Group = None,
    tau: float = 0.1,
    tnorm: str = "godel",
    top_k: int | None = None,
) -> Tensor:
    r"""Everybody-knows :math:`E_G\varphi = \bigwedge_{a \in G} K_a\varphi`.

    Each agent's knowledge is the box neuron routed through *that agent's*
    relation, then the group is folded with :func:`and_bounds`.

    **What it bounds.** The lower endpoint is a sound lower bound on crisp
    :math:`E_G`, the upper a sound upper bound; it reduces to the crisp
    operator as ``tau -> 0``. Each agent's box contributes at most
    :func:`~torchmodal.functional.box_width_entropy` of slack.

    Args:
        prop_bounds: ``(|W|, 2)`` or ``(|W|,)`` truth bounds for φ.
        accessibilities: ``(|G|, |W|, |W|)`` — one relation per agent. A single
            ``(|W|, |W|)`` matrix is broadcast to every agent in ``group``.
        group: Agent indices into ``accessibilities``. Default: all of them.
        tau: Temperature. Default 0.1.
        tnorm: Group fold; see :func:`and_bounds`. Default ``"godel"``.
        top_k: Passed through to :func:`~torchmodal.functional.necessity`.

    Returns:
        ``(|W|, 2)`` bounds for :math:`E_G\varphi` at each world.
    """
    prop_bounds = _as_bounds(prop_bounds)
    if accessibilities.dim() == 2:
        accessibilities = accessibilities.unsqueeze(0)
    idx = _group_index(accessibilities, group)
    per_agent = torch.stack(
        [
            F.necessity(prop_bounds, accessibilities[int(a)], tau=tau, top_k=top_k)
            for a in idx
        ],
        dim=0,
    )  # (|G|, |W|, 2)
    return and_bounds(per_agent, dim=0, tnorm=tnorm)


def mutual_knowledge(
    prop_bounds: Tensor,
    accessibilities: Tensor,
    group: Group = None,
    depth: int = 1,
    tau: float = 0.1,
    tnorm: str = "godel",
    tau_schedule: TauSchedule = None,
    top_k: int | None = None,
) -> Tensor:
    r"""The bounded tower :math:`E_G^k\varphi` — mutual knowledge to depth ``k``.

    :math:`E_G^1 = E_G\varphi`, :math:`E_G^{k+1} = E_G(E_G^k\varphi)`.

    **This is the operator to put in a loss**, not :func:`common_knowledge`.
    The tower degrades gracefully and keeps its gradient, whereas the fixpoint
    does not (see that function's warning). With ``tnorm="godel"`` and a dense
    6-agent relation at ``tau=0.1``, the lower bound runs
    ``0.896, 0.791, 0.687, 0.583, 0.478`` for ``k = 1..5``; with ``"luk"`` it is
    ``0.374`` then exactly ``0`` from ``k = 2`` on, with no gradient.

    **Choosing the depth.** Each level costs
    :func:`~torchmodal.functional.box_width_entropy`, so the faithful depth for
    a precision ``eps`` is ``k* = eps / (tau * H_bar)``. Compute ``H_bar``
    rather than guessing it; ``tau_schedule`` keeps the accumulated slack
    bounded when a deeper tower is needed.

    Args:
        prop_bounds: ``(|W|, 2)`` or ``(|W|,)`` truth bounds for φ.
        accessibilities: ``(|G|, |W|, |W|)`` or a single ``(|W|, |W|)``.
        group: Agent indices. Default: all.
        depth: Tower depth ``k >= 1``. Default 1 (plain ``E_G``).
        tau: Base temperature. Default 0.1.
        tnorm: Group fold; see :func:`and_bounds`. Default ``"godel"``.
        tau_schedule: ``None`` for a fixed ``tau``; a float ``rho`` for a
            geometric decay ``tau * rho ** level``, whose total slack is bounded
            by ``tau * H / (1 - rho)``; or a callable ``level -> tau``.
        top_k: Passed through to :func:`~torchmodal.functional.necessity`.

    Returns:
        ``(|W|, 2)`` bounds for :math:`E_G^{depth}\varphi`.

    Raises:
        ValueError: If ``depth < 1``.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    bounds = _as_bounds(prop_bounds)
    for level in range(depth):
        bounds = everybody_knows(
            bounds,
            accessibilities,
            group=group,
            tau=_tau_at(tau_schedule, tau, level),
            tnorm=tnorm,
            top_k=top_k,
        )
    return bounds


def pooled_accessibility(accessibilities: Tensor, group: Group = None) -> Tensor:
    r"""Pooled relation for :math:`D_G`: the elementwise min over the group.

    Distributed knowledge is what the group would know if its members shared
    everything, so it is the knowledge of an agent whose accessible set is the
    *intersection* of the members' — the smaller the relation, the stronger the
    knowledge.
    """
    if accessibilities.dim() == 2:
        return accessibilities
    idx = _group_index(accessibilities, group)
    return accessibilities.index_select(0, idx).min(dim=0).values


def distributed_knowledge(
    prop_bounds: Tensor,
    accessibilities: Tensor,
    group: Group = None,
    tau: float = 0.1,
    top_k: int | None = None,
) -> Tensor:
    r"""Distributed knowledge :math:`D_G\varphi` over the pooled relation.

    **What it bounds.** A sound bracket of crisp :math:`D_G`, reducing to it as
    ``tau -> 0``. Because the pooled relation is contained in every member's,
    :math:`D_G\varphi \ge K_a\varphi \ge E_G\varphi` pointwise on the lower
    endpoint, up to the smoothing gap.
    """
    return F.necessity(
        _as_bounds(prop_bounds),
        pooled_accessibility(accessibilities, group),
        tau=tau,
        top_k=top_k,
    )


def common_knowledge(
    prop_bounds: Tensor,
    accessibilities: Tensor,
    group: Group = None,
    tau: float = 0.1,
    tnorm: str = "godel",
    tau_decay: Optional[float] = 0.5,
    max_depth: Optional[int] = None,
    tol: float = 1e-4,
    max_iter: int = 200,
) -> Tensor:
    r"""Common knowledge :math:`C_G\varphi`, the greatest fixpoint of
    :math:`X \mapsto E_G(\varphi \wedge X)`.

    .. warning::
        **With the default settings the lower bound is exactly zero, for every
        input, and it carries no gradient.** This is measured, not incidental:
        every modal level costs at least
        :func:`~torchmodal.functional.box_width_entropy`, so an iteration that
        runs to convergence can only settle on the floor. At ``tau=0.1`` with 6
        agents, ``L = 0.0000`` and ``U = 1.0000`` for every φ and relation
        tried, with ``dL/dA`` identically 0.

        Consequences for users:

        - **Do not put the lower bound in a loss, a metric or a figure.** Use
          :func:`mutual_knowledge` at a bounded depth instead — that is what
          the fixpoint approximates and it keeps its gradient.
        - The **upper** bound remains informative and is the right read-out for
          "is common knowledge still attainable?".
        - ``tau_decay`` therefore **defaults to 0.5**, matching
          :func:`~torchmodal.functional.until_graph`, so the operator is
          usable as delivered. With a geometric schedule the accumulated slack
          is bounded by ``tau * H / (1 - tau_decay)`` instead of growing
          without limit. Measured at ``tau=0.1``, 6 agents, φ = ``[0.9, 1]``,
          ``tnorm="godel"``: complete 0.542, star 0.647, ring 0.680, path
          0.688 — against **0.000, with exactly zero gradient**, for all four
          when the schedule is disabled with ``tau_decay=None``.
        - This is the same failure as the *greatest-fixpoint cliff*: iterating
          a gfp down from ⊤ through a smooth ♢ loses a little each sweep, and
          below roughly 0.999 edge weight there is no non-zero fixed point to
          land on — measured on a 6-cycle at ``tau=0.1``, the value falls
          0.955 -> 0.754 -> 0.000 as the weight goes 1.0 -> 0.999 -> 0.99.
          The annealed schedule is what makes the sequence summable. Setting
          ``tau_decay=None`` restores the unannealed behaviour and the floor
          with it.

        The crisp limit is unaffected: as ``tau -> 0`` with crisp inputs the
        operator recovers classical common knowledge. The floor is a
        finite-temperature artefact of graded modal logic and is *not* evidence
        for the Halpern–Moses coordinated-attack result, which is about
        unreliable channels; cite that theorem as context, never as something
        these numbers demonstrate.

    Args:
        prop_bounds: ``(|W|, 2)`` or ``(|W|,)`` truth bounds for φ.
        accessibilities: ``(|G|, |W|, |W|)`` or a single ``(|W|, |W|)``.
        group: Agent indices. Default: all.
        tau: Base temperature. Default 0.1.
        tnorm: Conjunction and group fold; see :func:`and_bounds`.
        tau_decay: Geometric temperature decay per iteration, in ``(0, 1]``.
            Default ``0.5``. ``None`` disables annealing and restores the
            vacuous-lower-bound regime described above; it is kept only for
            reproducing that behaviour deliberately.
        max_depth: Stop after this many iterations instead of converging.
        tol: Sup-norm convergence threshold. Default 1e-4.
        max_iter: Iteration cap. Default 200.

    Returns:
        ``(|W|, 2)`` bounds for :math:`C_G\varphi`.
    """
    phi = _as_bounds(prop_bounds)
    x = torch.ones_like(phi)
    x = torch.stack([x[..., 0], x[..., 1]], dim=-1)
    limit = max_depth if max_depth is not None else max_iter
    for step in range(limit):
        conj = and_bounds(torch.stack([phi, x], dim=0), dim=0, tnorm=tnorm)
        nxt = everybody_knows(
            conj,
            accessibilities,
            group=group,
            tau=_tau_at(tau_decay, tau, step),
            tnorm=tnorm,
        )
        # Greatest fixpoint from above: the iterate may only decrease.
        nxt = torch.minimum(nxt, x)
        delta = float((nxt - x).abs().max().detach())
        x = nxt
        if max_depth is None and delta < tol:
            break
    return x
