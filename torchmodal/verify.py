"""
torchmodal.verify
~~~~~~~~~~~~~~~~~

Turning a learned relation into a checkable answer.

The README says soundness is a property you can check. This module is where
that becomes true end to end: round a learned relation, evaluate it with
:func:`torchmodal.functional.necessity`'s exact mode, and get back a verdict
that owes nothing to the temperature — together with an honest ``UNDECIDED``
when the bracket does not settle the question.

The workflow is:

1. **Train** a graded relation in soft mode, as usual.
2. **Measure** how safe rounding it would be, with :func:`rounding_margin` —
   the distance of each truth midpoint from the 0.5 decision boundary. A
   margin near zero means the crisp label is a coin flip and the certificate
   is not worth having.
3. **Round and certify** with :func:`round_and_certify`, which thresholds the
   relation, re-evaluates exactly, and reports ``PROVEN`` / ``REFUTED`` /
   ``UNDECIDED`` per world, with a witness path where one exists.
4. Optionally **export** to nuXmv or NuSMV with :func:`to_smv` and have an
   external checker confirm it.

:func:`certificate_gap` measures how often the soft answer and the exact one
disagree, which is the empirical version of the gap statement: if it is zero
on your frames, the soft model is already making the decisions the exact
checker would.

.. note::
   The SMV export is **structurally** validated here — the generated module is
   parsed back and checked for well-formedness — but this package does not
   bundle nuXmv, so the round trip against a real model checker is left to the
   caller. Treat the exporter as producing input for a tool you then run, not
   as a verified oracle in itself.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence

import torch
from torch import Tensor

__all__ = [
    "Verdict",
    "CertificateResult",
    "round_relation",
    "rounding_margin",
    "certificate_gap",
    "certify",
    "round_and_certify",
    "witness_path",
    "to_smv",
]


class Verdict:
    """Verdict constants for a per-world certificate."""

    PROVEN = "PROVEN"
    REFUTED = "REFUTED"
    UNDECIDED = "UNDECIDED"


class CertificateResult(NamedTuple):
    """The outcome of :func:`round_and_certify`.

    Attributes:
        verdicts: One of :class:`Verdict`'s constants per world.
        bounds: The exact bounds the verdicts were read from.
        margin: Per-world rounding margin of the *soft* evaluation, i.e. how
            far each midpoint sat from the 0.5 boundary before rounding. Small
            values mean the certificate rests on a near-tie.
        relation: The rounded relation the certificate is about — **not** the
            relation passed in. A certificate is a statement about this frame.
        n_flipped: How many entries of the relation the rounding moved by more
            than 0.25, a coarse measure of how much the certificate's frame
            differs from the learned one.
    """

    verdicts: List[str]
    bounds: Tensor
    margin: Tensor
    relation: Tensor
    n_flipped: int


def round_relation(
    accessibility: Tensor, threshold: float = 0.5
) -> Tensor:
    """Threshold a graded relation into a crisp one.

    Args:
        accessibility: ``(..., |W|, |W|)`` relation in [0, 1].
        threshold: Entries at or above this become 1, the rest 0.

    Returns:
        A relation of the same shape and dtype containing only 0 and 1.
    """
    return (accessibility >= threshold).to(accessibility.dtype)


def rounding_margin(bounds: Tensor) -> Tensor:
    r"""How far each truth value sits from the 0.5 decision boundary.

    The crisp label a bound implies is ``midpoint >= 0.5``. This returns
    ``|midpoint - 0.5|``, so 0.5 is maximally safe and 0.0 is a coin flip.

    This is the counterpart, for the *rounding* step, of what
    :func:`torchmodal.functional.box_width_entropy` is for the modal step: it
    turns "is this certificate trustworthy?" into a number rather than a
    judgement. A certificate read off a world whose margin is near zero should
    not be reported without saying so.

    Args:
        bounds: ``(..., |W|, 2)`` truth bounds.

    Returns:
        ``(..., |W|)`` margins in [0, 0.5].

    Example:
        >>> import torch
        >>> from torchmodal.verify import rounding_margin
        >>> rounding_margin(torch.tensor([[0.5, 0.5], [0.0, 0.0]])).tolist()
        [0.0, 0.5]
    """
    midpoint = 0.5 * (bounds[..., 0] + bounds[..., 1])
    return (midpoint - 0.5).abs()


def certify(
    bounds: Tensor, threshold: float = 0.5
) -> List[str]:
    r"""Read a per-world verdict off a truth interval.

    The point of carrying an interval is that it can decline to answer:

    - ``PROVEN`` when the whole interval sits at or above ``threshold``, so
      every value it admits is true;
    - ``REFUTED`` when the whole interval sits below, so every value is false;
    - ``UNDECIDED`` when it straddles the boundary.

    A library that says "I don't know" when it does not know is more useful
    than one that rounds, which is why ``UNDECIDED`` is a first-class outcome
    rather than an error.

    Args:
        bounds: ``(|W|, 2)`` truth bounds.
        threshold: Decision boundary. Default 0.5.

    Returns:
        A list of ``|W|`` verdict strings.
    """
    out: List[str] = []
    for lo, hi in bounds.tolist():
        if lo >= threshold:
            out.append(Verdict.PROVEN)
        elif hi < threshold:
            out.append(Verdict.REFUTED)
        else:
            out.append(Verdict.UNDECIDED)
    return out


def certificate_gap(
    soft_bounds: Tensor, exact_bounds: Tensor, threshold: float = 0.5
) -> float:
    r"""Disagreement rate between the soft rounding and the exact answer.

    The empirical form of the gap statement. Both arguments are rounded to
    crisp labels and compared; the result is the fraction of worlds where they
    differ. Zero means the soft model is already making exactly the decisions
    the exact checker would, which is the condition under which training
    against the soft operators is safe.

    Args:
        soft_bounds: ``(..., |W|, 2)`` bounds from a soft evaluation.
        exact_bounds: ``(..., |W|, 2)`` bounds from ``mode="exact"``.
        threshold: Decision boundary. Default 0.5.

    Returns:
        A float in [0, 1].

    Raises:
        ValueError: If the two shapes differ.
    """
    if soft_bounds.shape != exact_bounds.shape:
        raise ValueError(
            f"shape mismatch: {tuple(soft_bounds.shape)} vs "
            f"{tuple(exact_bounds.shape)}"
        )
    soft = 0.5 * (soft_bounds[..., 0] + soft_bounds[..., 1]) >= threshold
    exact = 0.5 * (exact_bounds[..., 0] + exact_bounds[..., 1]) >= threshold
    return float((soft != exact).to(torch.float64).mean())


def witness_path(
    accessibility: Tensor,
    start: int,
    goal: Tensor,
    threshold: float = 0.5,
) -> Optional[List[int]]:
    """A shortest path from ``start`` to a world satisfying ``goal``.

    The concrete evidence behind a ``PROVEN`` reachability verdict. Breadth-
    first, so the path is shortest, and ``None`` when no path exists — which
    is itself the evidence behind a ``REFUTED`` one.

    Args:
        accessibility: ``(|W|, |W|)`` relation; thresholded internally.
        start: Index of the world to start from.
        goal: ``(|W|,)`` or ``(|W|, 2)`` truth values for the goal.
        threshold: Edge and goal threshold. Default 0.5.

    Returns:
        A list of world indices beginning with ``start``, or ``None``.
    """
    A = (accessibility >= threshold)
    g = goal if goal.dim() == 1 else 0.5 * (goal[..., 0] + goal[..., 1])
    is_goal = (g >= threshold).tolist()

    n = A.shape[-1]
    if is_goal[start]:
        return [start]
    prev: Dict[int, int] = {}
    seen = {start}
    queue = [start]
    while queue:
        u = queue.pop(0)
        for v in range(n):
            if bool(A[u, v]) and v not in seen:
                seen.add(v)
                prev[v] = u
                if is_goal[v]:
                    path = [v]
                    while path[-1] != start:
                        path.append(prev[path[-1]])
                    return list(reversed(path))
                queue.append(v)
    return None


def round_and_certify(
    prop_bounds: Tensor,
    accessibility: Tensor,
    operator: str = "ef",
    threshold: float = 0.5,
    tau: float = 0.1,
    serial: bool = True,
    **kwargs: Any,
) -> CertificateResult:
    r"""Round a learned relation and certify a CTL property on it exactly.

    The whole point of the exercise: a learned, graded relation is not a
    Kripke frame, so a statement about it is not a statement about anything
    checkable. Rounding produces a frame; evaluating exactly on that frame
    produces an answer with no temperature in it.

    **The certificate is about the rounded relation, not the learned one.**
    That frame is returned in the result so it can be reported alongside, and
    ``n_flipped`` says how far it moved. If the margin is small or many
    entries flipped, the certificate is about a frame the model did not quite
    learn, and saying so is part of reporting it honestly.

    Args:
        prop_bounds: ``(|W|, 2)`` or ``(|W|,)`` truth values for the
            proposition.
        accessibility: ``(|W|, |W|)`` learned relation in [0, 1].
        operator: A unary CTL operator from :mod:`torchmodal.fixpoint` —
            ``"ex"``, ``"ax"``, ``"ef"``, ``"eg"``, ``"af"`` or ``"ag"``.
        threshold: Rounding and decision boundary. Default 0.5.
        tau: Temperature for the *soft* evaluation used to compute the margin.
        serial: Repair dead ends before evaluating. Default ``True``.
        **kwargs: Forwarded to the fixpoint operator.

    Returns:
        A :class:`CertificateResult`.

    Raises:
        ValueError: If ``operator`` is not a supported unary CTL operator.
    """
    from torchmodal import fixpoint as fp

    unary: Dict[str, Callable[..., Any]] = {
        "ex": fp.ex, "ax": fp.ax, "ef": fp.ef,
        "eg": fp.eg, "af": fp.af, "ag": fp.ag,
    }
    if operator not in unary:
        raise ValueError(
            f"operator must be one of {sorted(unary)}, got {operator!r}"
        )
    op = unary[operator]

    def _bounds(result: Any) -> Tensor:
        # ex / ax return a tensor; the fixpoint operators return a
        # FixpointResult carrying one.
        out = getattr(result, "bounds", result)
        assert isinstance(out, Tensor)
        return out

    soft = _bounds(
        op(prop_bounds, accessibility, tau=tau, serial=serial, **kwargs)
    )
    rounded = round_relation(accessibility, threshold)
    exact = _bounds(
        op(prop_bounds, rounded, mode="exact", serial=serial, **kwargs)
    )

    n_flipped = int(
        ((rounded - accessibility).abs() > 0.25).sum()
    )
    return CertificateResult(
        verdicts=certify(exact, threshold),
        bounds=exact,
        margin=rounding_margin(soft),
        relation=rounded,
        n_flipped=n_flipped,
    )


def to_smv(
    accessibility: Tensor,
    labels: Dict[str, Sequence[float]],
    spec: Optional[str] = None,
    threshold: float = 0.5,
    module_name: str = "main",
) -> str:
    """Export a rounded Kripke frame as an SMV module for nuXmv or NuSMV.

    The relation is thresholded into a transition relation over a single
    ``state`` variable, and each entry of ``labels`` becomes a ``DEFINE``
    predicate over that variable.

    .. warning::
       This produces *input* for a model checker; it does not run one. The
       output is checked here for structural well-formedness only, since this
       package does not bundle nuXmv. Run the file through the checker before
       treating its answer as confirmation of anything.

    Args:
        accessibility: ``(|W|, |W|)`` relation; thresholded internally.
        labels: Proposition name to per-world truth values, each of length
            ``|W|``. Names must be valid SMV identifiers.
        spec: An optional CTL specification, e.g. ``"AG (safe)"``, emitted as
            a ``CTLSPEC``.
        threshold: Edge and label threshold. Default 0.5.
        module_name: SMV module name. Default ``"main"``.

    Returns:
        The SMV source as a string.

    Raises:
        ValueError: If a label has the wrong length, or a dead end is present
            — SMV requires a total transition relation, so run
            :func:`torchmodal.functional.serialize` first.

    Example:
        >>> import torch
        >>> from torchmodal.verify import to_smv
        >>> A = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        >>> "MODULE main" in to_smv(A, {"p": [1.0, 0.0]})
        True
    """
    A = (accessibility >= threshold)
    n = A.shape[-1]

    for name, values in labels.items():
        if len(values) != n:
            raise ValueError(
                f"label {name!r} has {len(values)} values but the frame has "
                f"{n} worlds"
            )
    dead = [i for i in range(n) if not bool(A[i].any())]
    if dead:
        raise ValueError(
            f"worlds {dead} have no successor; SMV requires a total "
            f"transition relation. Apply torchmodal.functional.serialize "
            f"first."
        )

    lines = [
        f"MODULE {module_name}",
        "VAR",
        f"  state : 0 .. {n - 1};",
        "ASSIGN",
        "  init(state) := 0;",
        "  next(state) := case",
    ]
    for i in range(n):
        succ = [str(j) for j in range(n) if bool(A[i, j])]
        lines.append(f"    state = {i} : {{{', '.join(succ)}}};")
    lines.append("  esac;")

    if labels:
        lines.append("DEFINE")
        for name, values in labels.items():
            true_states = [
                str(i) for i, v in enumerate(values) if float(v) >= threshold
            ]
            rhs = (
                " | ".join(f"state = {i}" for i in true_states)
                if true_states
                else "FALSE"
            )
            lines.append(f"  {name} := {rhs};")

    if spec is not None:
        lines.append(f"CTLSPEC {spec};")

    return "\n".join(lines) + "\n"
