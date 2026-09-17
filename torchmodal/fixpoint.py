"""
torchmodal.fixpoint
~~~~~~~~~~~~~~~~~~~

Least and greatest fixpoint combinators, and the six CTL temporal operators
built from them.

:func:`torchmodal.functional.until_graph` is already the least fixpoint of
``EU``, and :class:`torchmodal.systems.TemporalOperator` is a single box over
a precomputed reachability matrix. This module generalises both: ``lfp`` and
``gfp`` take an arbitrary step function, and ``EX``, ``EF``, ``EG``, ``EU``,
``AX``, ``AF``, ``AG`` and ``AU`` are defined on top of them.

Three things the design has to get right, each learned from a measurement.

**1. The greatest-fixpoint cliff.** Iterating a gfp *down* from ``⊤`` through a
smooth ``♢`` loses a little on every sweep, and below roughly 0.999 edge weight
there is no non-zero fixed point to land on. On a 6-cycle at ``tau = 0.1`` with
``phi`` true everywhere — where the crisp answer is 1 at every world —
``EG`` falls ``0.955 -> 0.754 -> 0.000`` as the weight goes
``1.0 -> 0.999 -> 0.99``. The collapse is **not** a t-norm artefact: Gödel,
product and Łukasiewicz all do it, because the lossy step is the modal one and
not the conjunction. Two things help, and both are available here: an annealed
temperature (``tau_decay``), which makes the per-sweep loss summable, and
``mode="exact"``.

Be precise about what exact mode buys, though. It removes the **temperature**
gap, not the **gradedness**: on a 0.99-weighted cycle it still decays, because
``max(A + L - 1)`` is genuinely 0.99 and then 0.98 and so on. What saves the
answer is that the iteration stops on rounding stabilisation while the crisp
label is still correct — exact mode returns 0.97 there, which rounds to the
right certificate. For a trustworthy answer on a learned relation, **round the
relation first** (``(A >= 0.5).to(A.dtype)``) and then evaluate exactly; that
is the only combination with no decay at all.

**2. A stop rule that terminates.** A strict tolerance does not: even at edge
weight 1.0 the soft iteration is still creeping at a 200-sweep cap. The
combinators therefore also stop on **rounding stabilisation** — when the crisp
label implied by the bounds has not changed for ``patience`` sweeps — because
the certificate is final long before the soft value settles. Both combinators
return the iteration count and whether they converged, since both enter the
gap statement.

**3. Seriality.** The universal operators are sound only on a serial frame: at
a dead end ``AX`` is vacuously satisfied, so a path that simply stops counts as
success. Pass ``serial=True`` (the default) and the frame is repaired with
:func:`torchmodal.functional.serialize` before evaluation.

Example::

    import torch
    from torchmodal import fixpoint as fp

    A = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    phi = torch.tensor([1., 1., 0.])

    fp.eg(phi, A, mode="exact").bounds     # infinite phi-path?
    fp.ef(phi, A, mode="exact").bounds     # phi reachable?
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional

import torch
from torch import Tensor

from torchmodal import functional as F

__all__ = [
    "FixpointResult",
    "lfp",
    "gfp",
    "ex",
    "ax",
    "ef",
    "eg",
    "eu",
    "af",
    "ag",
    "au",
]


class FixpointResult(NamedTuple):
    """The outcome of a fixpoint iteration.

    Attributes:
        bounds: ``(..., |W|, 2)`` truth bounds at the fixpoint.
        n_iters: Sweeps actually performed.
        converged: ``True`` when the iteration stopped on its own — by
            tolerance or by rounding stabilisation — rather than exhausting
            ``max_iter``. A ``False`` here means the reported bounds are a
            truncation, not a fixpoint, and the gap statement does not hold.
        stopped_by: ``"tolerance"``, ``"rounding"`` or ``"max_iter"``.
    """

    bounds: Tensor
    n_iters: int
    converged: bool
    stopped_by: str


def _as_bounds(x: Tensor) -> Tensor:
    """Accept ``(..., |W|)`` point values or ``(..., |W|, 2)`` bounds."""
    return torch.stack([x, x], dim=-1) if x.dim() == 1 else x


def _crisp_label(bounds: Tensor) -> Tensor:
    """The certificate implied by the bounds: midpoint at or above 0.5."""
    return 0.5 * (bounds[..., 0] + bounds[..., 1]) >= 0.5


def _iterate(
    step: Callable[[Tensor, float], Tensor],
    init: Tensor,
    tau: float,
    tau_decay: Optional[float],
    tol: float,
    max_iter: int,
    round_stable: bool,
    patience: int,
) -> FixpointResult:
    """Kleene-iterate ``step`` from ``init`` until it settles."""
    Z = init
    prev_label = _crisp_label(Z)
    stable = 0
    tau_j = tau

    for it in range(1, max_iter + 1):
        Z_next = torch.clamp(step(Z, tau_j), 0.0, 1.0)
        # detach: the stopping test is control flow, not part of the graph
        change = (Z_next - Z).detach().abs().max().item()
        label = _crisp_label(Z_next)
        Z = Z_next

        if change < tol:
            return FixpointResult(Z, it, True, "tolerance")

        if round_stable:
            if torch.equal(label, prev_label):
                stable += 1
                if stable >= patience:
                    return FixpointResult(Z, it, True, "rounding")
            else:
                stable = 0
        prev_label = label

        if tau_decay is not None:
            tau_j *= tau_decay

    return FixpointResult(Z, max_iter, False, "max_iter")


def _default_max_iter(n: int, max_iter: Optional[int]) -> int:
    """A cap that scales with the frame, as crisp CTL's depth bound does."""
    return max_iter if max_iter is not None else 4 * n + 20


def lfp(
    step: Callable[[Tensor, float], Tensor],
    n_worlds: int,
    tau: float = 0.1,
    tau_decay: Optional[float] = None,
    tol: float = 1e-6,
    max_iter: Optional[int] = None,
    round_stable: bool = True,
    patience: int = 3,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> FixpointResult:
    r"""Least fixpoint: Kleene-iterate ``step`` up from :math:`\bot`.

    Args:
        step: ``(bounds, tau) -> bounds``, the monotone operator to iterate.
        n_worlds: ``|W|``.
        tau: Initial temperature handed to ``step``.
        tau_decay: Geometric decay per sweep, or ``None`` for a fixed
            temperature. An lfp does not generally need annealing — see
            :func:`gfp`, which does.
        tol: Sup-norm change below which the iteration has converged.
        max_iter: Sweep cap. Default ``4 * n_worlds + 20``.
        round_stable: Also stop once the crisp label is unchanged for
            ``patience`` sweeps.
        patience: Sweeps of label stability required. Default 3.
        dtype: Dtype of the initial bottom element.
        device: Device of the initial bottom element.

    Returns:
        A :class:`FixpointResult`.
    """
    init = torch.zeros(n_worlds, 2, dtype=dtype, device=device)
    return _iterate(
        step, init, tau, tau_decay, tol,
        _default_max_iter(n_worlds, max_iter), round_stable, patience,
    )


def gfp(
    step: Callable[[Tensor, float], Tensor],
    n_worlds: int,
    tau: float = 0.1,
    tau_decay: Optional[float] = 0.5,
    tol: float = 1e-6,
    max_iter: Optional[int] = None,
    round_stable: bool = True,
    patience: int = 3,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> FixpointResult:
    r"""Greatest fixpoint: Kleene-iterate ``step`` down from :math:`\top`.

    .. warning::
       **The greatest-fixpoint cliff.** In soft mode over a graded relation
       this iteration has no non-zero fixed point to land on below roughly
       0.999 edge weight, and collapses to 0 with no gradient — see the module
       docstring for the measured table. ``tau_decay`` therefore defaults to
       **0.5** here, unlike :func:`lfp`, because an annealed schedule makes the
       accumulated loss summable. For a learned relation, rounding it *and*
       evaluating with ``mode="exact"`` is more reliable than any schedule:
       exact mode alone removes the temperature gap but not the decay caused
       by sub-unit edge weights.

    Args:
        step: ``(bounds, tau) -> bounds``, the monotone operator to iterate.
        n_worlds: ``|W|``.
        tau: Initial temperature handed to ``step``.
        tau_decay: Geometric decay per sweep. Default 0.5; ``None`` disables
            annealing and restores the collapse.
        tol: Sup-norm change below which the iteration has converged.
        max_iter: Sweep cap. Default ``4 * n_worlds + 20``.
        round_stable: Also stop once the crisp label is unchanged for
            ``patience`` sweeps. This is usually what terminates a gfp, since
            a strict tolerance does not.
        patience: Sweeps of label stability required. Default 3.
        dtype: Dtype of the initial top element.
        device: Device of the initial top element.

    Returns:
        A :class:`FixpointResult`.
    """
    init = torch.ones(n_worlds, 2, dtype=dtype, device=device)
    return _iterate(
        step, init, tau, tau_decay, tol,
        _default_max_iter(n_worlds, max_iter), round_stable, patience,
    )


# ---------------------------------------------------------------------------
# Connectives on bounds
# ---------------------------------------------------------------------------


def _and(a: Tensor, b: Tensor) -> Tensor:
    """Gödel conjunction on both endpoints — idempotent, so it iterates."""
    return torch.minimum(a, b)


def _or(a: Tensor, b: Tensor) -> Tensor:
    """Gödel disjunction on both endpoints."""
    return torch.maximum(a, b)


def _not(a: Tensor) -> Tensor:
    """Negation swaps and complements the endpoints."""
    return torch.stack([1.0 - a[..., 1], 1.0 - a[..., 0]], dim=-1)


# ---------------------------------------------------------------------------
# The CTL operators
# ---------------------------------------------------------------------------


def _prep(accessibility: Tensor, serial: bool) -> Tensor:
    return F.serialize(accessibility) if serial else accessibility


def ex(
    phi: Tensor,
    accessibility: Tensor,
    tau: float = 0.1,
    mode: str = "soft",
    serial: bool = True,
) -> Tensor:
    r"""``EX phi`` — some successor satisfies ``phi``.

    This is exactly :func:`torchmodal.functional.possibility`; it is exposed
    here so the CTL vocabulary is complete in one place.

    Args:
        phi: ``(|W|, 2)`` bounds or ``(|W|,)`` point values.
        accessibility: ``(|W|, |W|)`` relation.
        tau: Temperature. Default 0.1.
        mode: ``"soft"`` or ``"exact"``.
        serial: Repair dead ends first. Harmless for ``EX``; kept for a
            uniform signature.

    Returns:
        ``(|W|, 2)`` bounds.
    """
    return F.possibility(
        _as_bounds(phi), _prep(accessibility, serial), tau=tau, mode=mode
    )


def ax(
    phi: Tensor,
    accessibility: Tensor,
    tau: float = 0.1,
    mode: str = "soft",
    serial: bool = True,
) -> Tensor:
    r"""``AX phi`` — every successor satisfies ``phi``.

    Defined as ``¬EX¬phi``, which is sound **only on a serial frame**: at a
    dead end there is no successor to violate ``phi``, so ``AX`` is vacuously
    true and a computation that simply stops counts as success. ``serial=True``
    (the default) repairs the frame first.

    Args:
        phi: ``(|W|, 2)`` bounds or ``(|W|,)`` point values.
        accessibility: ``(|W|, |W|)`` relation.
        tau: Temperature. Default 0.1.
        mode: ``"soft"`` or ``"exact"``.
        serial: Repair dead ends with
            :func:`~torchmodal.functional.serialize` first. Default ``True``;
            set ``False`` only when the frame is known to be serial.

    Returns:
        ``(|W|, 2)`` bounds.
    """
    A = _prep(accessibility, serial)
    return _not(F.possibility(_not(_as_bounds(phi)), A, tau=tau, mode=mode))


def _ef_step(phi_b: Tensor, A: Tensor, mode: str):
    def step(Z: Tensor, t: float) -> Tensor:
        return _or(phi_b, F.possibility(Z, A, tau=t, mode=mode))

    return step


def _eg_step(phi_b: Tensor, A: Tensor, mode: str):
    def step(Z: Tensor, t: float) -> Tensor:
        return _and(phi_b, F.possibility(Z, A, tau=t, mode=mode))

    return step


def _eu_step(phi_b: Tensor, psi_b: Tensor, A: Tensor, mode: str):
    def step(Z: Tensor, t: float) -> Tensor:
        return _or(psi_b, _and(phi_b, F.possibility(Z, A, tau=t, mode=mode)))

    return step


def ef(
    phi: Tensor,
    accessibility: Tensor,
    tau: float = 0.1,
    mode: str = "soft",
    serial: bool = True,
    **kwargs: Any,
) -> FixpointResult:
    r"""``EF phi`` — ``phi`` is reachable. Least fixpoint of ``Z = phi ∨ ♢Z``.

    Args:
        phi: ``(|W|, 2)`` bounds or ``(|W|,)`` point values.
        accessibility: ``(|W|, |W|)`` relation.
        tau: Temperature. Default 0.1.
        mode: ``"soft"`` or ``"exact"``.
        serial: Repair dead ends first.
        **kwargs: Forwarded to :func:`lfp` (``tol``, ``max_iter``, ...).

    Returns:
        A :class:`FixpointResult`.
    """
    A = _prep(accessibility, serial)
    phi_b = _as_bounds(phi)
    return lfp(
        _ef_step(phi_b, A, mode), A.shape[-1], tau=tau,
        dtype=phi_b.dtype, device=phi_b.device, **kwargs,
    )


def eg(
    phi: Tensor,
    accessibility: Tensor,
    tau: float = 0.1,
    mode: str = "soft",
    serial: bool = True,
    **kwargs: Any,
) -> FixpointResult:
    r"""``EG phi`` — some infinite path stays in ``phi``.

    Greatest fixpoint of ``Z = phi ∧ ♢Z``.

    .. warning::
       Subject to the greatest-fixpoint cliff in soft mode; see :func:`gfp`
       and the module docstring. On a learned relation, round it and use
       ``mode="exact"``.

    Args:
        phi: ``(|W|, 2)`` bounds or ``(|W|,)`` point values.
        accessibility: ``(|W|, |W|)`` relation.
        tau: Temperature. Default 0.1.
        mode: ``"soft"`` or ``"exact"``.
        serial: Repair dead ends first. Without it a dead end cannot satisfy
            ``EG`` at all, since it has no infinite path.
        **kwargs: Forwarded to :func:`gfp`.

    Returns:
        A :class:`FixpointResult`.
    """
    A = _prep(accessibility, serial)
    phi_b = _as_bounds(phi)
    return gfp(
        _eg_step(phi_b, A, mode), A.shape[-1], tau=tau,
        dtype=phi_b.dtype, device=phi_b.device, **kwargs,
    )


def eu(
    phi: Tensor,
    psi: Tensor,
    accessibility: Tensor,
    tau: float = 0.1,
    mode: str = "soft",
    serial: bool = True,
    **kwargs: Any,
) -> FixpointResult:
    r"""``E[phi U psi]`` — least fixpoint of ``Z = psi ∨ (phi ∧ ♢Z)``.

    The same formula as :func:`torchmodal.functional.until_graph`, exposed
    through the fixpoint machinery so it reports its iteration count and can
    be evaluated in exact mode.

    Args:
        phi: The invariant, ``(|W|, 2)`` or ``(|W|,)``.
        psi: The goal, same shape.
        accessibility: ``(|W|, |W|)`` relation.
        tau: Temperature. Default 0.1.
        mode: ``"soft"`` or ``"exact"``.
        serial: Repair dead ends first.
        **kwargs: Forwarded to :func:`lfp`.

    Returns:
        A :class:`FixpointResult`.
    """
    A = _prep(accessibility, serial)
    phi_b, psi_b = _as_bounds(phi), _as_bounds(psi)
    return lfp(
        _eu_step(phi_b, psi_b, A, mode), A.shape[-1], tau=tau,
        dtype=phi_b.dtype, device=phi_b.device, **kwargs,
    )


def af(
    phi: Tensor,
    accessibility: Tensor,
    tau: float = 0.1,
    mode: str = "soft",
    serial: bool = True,
    **kwargs: Any,
) -> FixpointResult:
    r"""``AF phi`` — every path eventually reaches ``phi``.

    Least fixpoint of ``Z = phi ∨ □Z``. Sound only on a serial frame.

    Args:
        phi: ``(|W|, 2)`` bounds or ``(|W|,)`` point values.
        accessibility: ``(|W|, |W|)`` relation.
        tau: Temperature. Default 0.1.
        mode: ``"soft"`` or ``"exact"``.
        serial: Repair dead ends first. Default ``True``.
        **kwargs: Forwarded to :func:`lfp`.

    Returns:
        A :class:`FixpointResult`.
    """
    A = _prep(accessibility, serial)
    phi_b = _as_bounds(phi)

    def step(Z: Tensor, t: float) -> Tensor:
        return _or(phi_b, F.necessity(Z, A, tau=t, mode=mode))

    return lfp(
        step, A.shape[-1], tau=tau,
        dtype=phi_b.dtype, device=phi_b.device, **kwargs,
    )


def ag(
    phi: Tensor,
    accessibility: Tensor,
    tau: float = 0.1,
    mode: str = "soft",
    serial: bool = True,
    **kwargs: Any,
) -> FixpointResult:
    r"""``AG phi`` — ``phi`` holds on every reachable world.

    Greatest fixpoint of ``Z = phi ∧ □Z``. Sound only on a serial frame, and
    subject to the greatest-fixpoint cliff in soft mode.

    Args:
        phi: ``(|W|, 2)`` bounds or ``(|W|,)`` point values.
        accessibility: ``(|W|, |W|)`` relation.
        tau: Temperature. Default 0.1.
        mode: ``"soft"`` or ``"exact"``.
        serial: Repair dead ends first. Default ``True``.
        **kwargs: Forwarded to :func:`gfp`.

    Returns:
        A :class:`FixpointResult`.
    """
    A = _prep(accessibility, serial)
    phi_b = _as_bounds(phi)

    def step(Z: Tensor, t: float) -> Tensor:
        return _and(phi_b, F.necessity(Z, A, tau=t, mode=mode))

    return gfp(
        step, A.shape[-1], tau=tau,
        dtype=phi_b.dtype, device=phi_b.device, **kwargs,
    )


def au(
    phi: Tensor,
    psi: Tensor,
    accessibility: Tensor,
    tau: float = 0.1,
    mode: str = "soft",
    serial: bool = True,
    **kwargs: Any,
) -> FixpointResult:
    r"""``A[phi U psi]`` — least fixpoint of ``Z = psi ∨ (phi ∧ □Z)``.

    Sound only on a serial frame: at a dead end ``□Z`` is vacuously true, so a
    path that stops would satisfy the formula.

    Args:
        phi: The invariant, ``(|W|, 2)`` or ``(|W|,)``.
        psi: The goal, same shape.
        accessibility: ``(|W|, |W|)`` relation.
        tau: Temperature. Default 0.1.
        mode: ``"soft"`` or ``"exact"``.
        serial: Repair dead ends first. Default ``True``.
        **kwargs: Forwarded to :func:`lfp`.

    Returns:
        A :class:`FixpointResult`.
    """
    A = _prep(accessibility, serial)
    phi_b, psi_b = _as_bounds(phi), _as_bounds(psi)

    def step(Z: Tensor, t: float) -> Tensor:
        return _or(psi_b, _and(phi_b, F.necessity(Z, A, tau=t, mode=mode)))

    return lfp(
        step, A.shape[-1], tau=tau,
        dtype=phi_b.dtype, device=phi_b.device, **kwargs,
    )
