"""
torchmodal.diagnostics
~~~~~~~~~~~~~~~~~~~~~~

Diagnostics for silently-dead terms in differentiable logic.

The characteristic failure mode of a differentiable logic is not an
exception — it is a term that has been pinned to the floor or the ceiling
of ``[0, 1]`` and whose gradient has vanished. Nothing raises; the term
simply stops contributing while the rest of the model keeps training, and
the symptom surfaces much later as "the constraint had no effect".

Three constructs in this library can reach that state:

- :func:`torchmodal.functional.until` — its Łukasiewicz backward sweep
  loses ``1 - L_phi`` per step, so a long chain drives the lower bound to
  zero regardless of the data.
- Iterated :func:`torchmodal.functional.necessity` — each level costs
  ``tau * H(w)`` (see :func:`torchmodal.functional.box_width_entropy`),
  so a deep enough nest floors at exactly
  ``ceil(1 / (tau * H))`` levels.
- :func:`torchmodal.functional.contradiction` after a modal neuron — it
  is identically zero, with zero gradient, until the bound crossing
  exceeds the box width.

:func:`gradient_health` reports the condition; :func:`assert_has_signal`
is the raising variant for tests.

Example::

    from torchmodal.diagnostics import gradient_health

    A = torch.ones(8, 8, requires_grad=True)
    report = gradient_health(
        lambda: nested_necessity(A, depth=6), {"A": A}
    )
    print(report["healthy"])   # False
    print(report["issues"])    # ["bound 'output' is vacuous: ...", ...]
    print(report["warnings"])  # ["term 'output.L' is dead: ...", ...]
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "gradient_health",
    "assert_has_signal",
    "GradientHealthError",
    "vacuity_report",
    "monotone_in_accessibility",
    "MONOTONICITY",
]


class GradientHealthError(AssertionError):
    """Raised by :func:`assert_has_signal` when a term carries no signal."""


ParamSpec = Union[Tensor, Sequence[Tensor], Mapping[str, Tensor], nn.Module]


def _normalize_params(params: ParamSpec) -> Dict[str, Tensor]:
    """Coerce the accepted parameter spellings into a ``name -> tensor`` map."""
    if isinstance(params, nn.Module):
        return {n: p for n, p in params.named_parameters()}
    if isinstance(params, Tensor):
        return {"param": params}
    if isinstance(params, Mapping):
        return dict(params)
    return {f"param{i}": p for i, p in enumerate(params)}


def _normalize_outputs(
    out: Any, names: Optional[Sequence[str]]
) -> Dict[str, Tensor]:
    """Coerce a function result into a ``name -> tensor`` map.

    Accepts a bare tensor, a sequence of tensors, or a mapping. Non-tensor
    entries are dropped, so a function may return bookkeeping alongside
    its bounds.
    """
    if isinstance(out, Tensor):
        collected: Dict[str, Tensor] = {"output": out}
    elif isinstance(out, Mapping):
        collected = {str(k): v for k, v in out.items() if isinstance(v, Tensor)}
    elif isinstance(out, (list, tuple)):
        collected = {
            f"output{i}": v for i, v in enumerate(out) if isinstance(v, Tensor)
        }
    else:
        raise TypeError(
            "fn must return a Tensor, or a sequence/mapping containing "
            f"Tensors; got {type(out).__name__}"
        )
    if not collected:
        raise ValueError("fn returned no tensors to diagnose")
    if names is not None:
        if len(names) != len(collected):
            raise ValueError(
                f"names has {len(names)} entries but fn returned "
                f"{len(collected)} tensors"
            )
        collected = dict(zip(names, collected.values()))
    return collected


def _split_bounds(
    out_map: Dict[str, Tensor], bounds: Union[bool, str]
) -> Dict[str, Tensor]:
    """Split ``(..., 2)`` bound tensors into separate ``.L`` / ``.U`` terms.

    A modal bound tensor is the library's central data structure, and its
    two columns fail *in opposite directions*: the dead state of a box
    neuron is ``L = 0`` together with ``U = 1``. Checked as one tensor
    neither column looks pinned, so the endpoints must be diagnosed apart.

    Args:
        out_map: Collected output tensors.
        bounds: ``"auto"`` splits any tensor whose last dimension is 2,
            ``True`` requires every tensor to be bounds, ``False``
            disables splitting.

    Returns:
        A new map in which split tensors appear as ``"<name>.L"`` and
        ``"<name>.U"``.
    """
    if bounds is False:
        return out_map
    split: Dict[str, Tensor] = {}
    for name, t in out_map.items():
        is_bounds = t.dim() >= 1 and t.shape[-1] == 2
        if bounds is True and not is_bounds:
            raise ValueError(
                f"bounds=True but output '{name}' has shape {tuple(t.shape)}, "
                f"whose last dimension is not 2"
            )
        if is_bounds:
            split[f"{name}.L"] = t[..., 0]
            split[f"{name}.U"] = t[..., 1]
        else:
            split[name] = t
    return split


def gradient_health(
    fn: Callable[..., Any],
    params: ParamSpec,
    *args: Any,
    names: Optional[Sequence[str]] = None,
    bounds: Union[bool, str] = "auto",
    floor: float = 0.0,
    ceiling: float = 1.0,
    sat_atol: float = 1e-6,
    grad_atol: float = 1e-12,
    saturated_frac: float = 1.0,
    vacuous_atol: float = 1e-6,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Report whether a differentiable-logic term still carries signal.

    Calls ``fn(*args, **kwargs)`` and reports, **per output term and per
    parameter tensor**, whether the value has been pinned to the floor or
    the ceiling of the truth interval and whether its gradient vanished.
    Gradients are attributed *per term* — one :func:`torch.autograd.grad`
    call each — so a dead endpoint is still located when the other
    endpoint of the same bound is alive.

    This detects the failure mode described in the module docstring: a
    term that is silently stuck, contributing nothing to training while
    raising nothing.

    Bound tensors of shape ``(..., 2)`` are split into ``"<name>.L"`` and
    ``"<name>.U"`` before checking, because the dead state of a modal
    neuron is ``L = 0`` *with* ``U = 1`` — checked jointly, neither column
    looks pinned.

    **What counts as unhealthy.** A term is reported as **dead** when it is
    pinned *and* its gradient to every parameter has vanished, and as
    **saturated** when pinned but still differentiable. Both go to
    ``"warnings"``, not ``"issues"``: on its own, a pinned endpoint does
    *not* mean anything is wrong. A sound upper bound that has legitimately
    reached 1 is indistinguishable, at the level of a single term, from a
    broken one — and whether the clamp at the interval edge passes gradient
    exactly *at* the boundary is a torch-version convention (2.8 passes
    1.0, 2.14 passes 0.0), so keying ``healthy`` on it would be both noisy
    and version-dependent.

    ``healthy`` is therefore ``False`` only on signals that mean the term
    has genuinely stopped carrying information:

    - a **vacuous** bound — width spanning the whole interval everywhere,
      so the pair says nothing at all (this is what a collapsed nest of
      modal operators produces, on every torch version);
    - **no parameter receiving a usable gradient** from any term;
    - a term with **no autograd path** at all, or a parameter with
      ``requires_grad=False``.

    Read ``"warnings"`` as well when diagnosing: a dead endpoint whose
    bound is not yet vacuous is often the first sign of a nest about to
    collapse.

    .. note::
       The call runs under ``torch.enable_grad`` and leaves ``.grad``
       untouched on the caller's tensors: gradients are taken with
       :func:`torch.autograd.grad`, not ``.backward()``.

    Args:
        fn: Callable to diagnose. May return a tensor, or a sequence or
            mapping containing tensors. Non-tensor entries are ignored.
        params: Tensors whose gradients are checked. Accepts a single
            tensor, a sequence, a ``name -> tensor`` mapping, or an
            :class:`torch.nn.Module` (its named parameters are used).
        *args: Positional arguments forwarded to ``fn``.
        names: Optional replacement names for the returned tensors, in
            order, applied *before* bound splitting. Must match the number
            of tensors returned.
        bounds: ``"auto"`` (default) splits any returned tensor whose last
            dimension is 2 into ``L`` / ``U`` terms; ``True`` requires
            every returned tensor to be bounds; ``False`` disables it.
        floor: Lower end of the truth interval. Default 0.0.
        ceiling: Upper end of the truth interval. Default 1.0.
        sat_atol: Tolerance for calling a value saturated at an end of the
            interval. Default 1e-6.
        grad_atol: A gradient whose maximum absolute entry is at or below
            this is reported as vanished. Default 1e-12.
        saturated_frac: Fraction of a term's entries that must sit at one
            end before it counts as pinned. Default 1.0 (every entry).
        vacuous_atol: Tolerance for calling a split bound pair vacuous
            (``U - L == ceiling - floor``). Default 1e-6.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        A dict with keys:

        - ``"outputs"`` — per output term: ``min``, ``max``, ``mean``,
            ``frac_at_floor``, ``frac_at_ceiling``, ``pinned_at_floor``,
            ``pinned_at_ceiling``, ``requires_grad``, ``dead``, and
            ``grads`` (per parameter: ``grad_norm``, ``grad_max_abs``,
            ``grad_vanished``).
        - ``"params"`` — per parameter tensor, aggregated over terms:
            ``has_grad``, ``grad_norm``, ``grad_max_abs``,
            ``grad_vanished``, ``requires_grad``.
        - ``"vacuous"`` — names of split bound pairs whose width is the
            full interval everywhere.
        - ``"healthy"`` — ``True`` when no bound is vacuous, every term
            has an autograd path, and at least one parameter received a
            non-vanishing gradient.
        - ``"issues"`` — human-readable strings, one per vacuous bound or
            unreachable parameter. Empty when healthy.
        - ``"warnings"`` — dead and saturated terms. Informative, but do
            not affect ``healthy``; see the note above on why.

    Example:
        >>> import torch
        >>> from torchmodal import functional as F
        >>> from torchmodal.diagnostics import gradient_health
        >>> A = torch.ones(8, 8, requires_grad=True)
        >>> def deep_box():
        ...     b = torch.ones(8, 2)
        ...     for _ in range(6):
        ...         b = F.necessity(b, A, tau=0.1)
        ...     return b
        >>> gradient_health(deep_box, {"A": A})["healthy"]
        False
    """
    param_map = _normalize_params(params)
    for name, p in param_map.items():
        if not isinstance(p, Tensor):
            raise TypeError(f"param '{name}' is not a Tensor")

    tracked_names = [n for n, p in param_map.items() if p.requires_grad]
    tracked = [param_map[n] for n in tracked_names]

    with torch.enable_grad():
        raw = _normalize_outputs(fn(*args, **kwargs), names)
        raw_widths = {
            name: (t[..., 1] - t[..., 0]).detach()
            for name, t in raw.items()
            if t.dim() >= 1 and t.shape[-1] == 2
        }
        out_map = _split_bounds(raw, bounds)

        # Per-term gradients: a dead L endpoint must still be located when
        # the U endpoint of the same bound is alive, so terms cannot share
        # one backward pass.
        term_grads: Dict[str, Dict[str, Optional[Tensor]]] = {}
        for term, t in out_map.items():
            if not (t.requires_grad and tracked):
                term_grads[term] = {n: None for n in tracked_names}
                continue
            gs = torch.autograd.grad(
                t.sum(), tracked, allow_unused=True, retain_graph=True
            )
            term_grads[term] = dict(zip(tracked_names, gs))

    issues: List[str] = []
    warns: List[str] = []

    out_report: Dict[str, Dict[str, Any]] = {}
    for name, t in out_map.items():
        d = t.detach()
        n = max(d.numel(), 1)
        at_floor = (d <= floor + sat_atol).sum().item() / n
        at_ceiling = (d >= ceiling - sat_atol).sum().item() / n
        pinned_floor = at_floor >= saturated_frac
        pinned_ceiling = at_ceiling >= saturated_frac
        pinned = pinned_floor or pinned_ceiling

        grads_report: Dict[str, Dict[str, Any]] = {}
        term_alive = False
        for pname in tracked_names:
            g = term_grads[name][pname]
            if g is None:
                grads_report[pname] = {
                    "grad_norm": 0.0,
                    "grad_max_abs": 0.0,
                    "grad_vanished": True,
                }
                continue
            gmax = g.abs().max().item()
            vanished = gmax <= grad_atol
            grads_report[pname] = {
                "grad_norm": g.norm().item(),
                "grad_max_abs": gmax,
                "grad_vanished": vanished,
            }
            if not vanished:
                term_alive = True

        # A term with no autograd path at all is reported as such, once;
        # "dead" is reserved for a term that is differentiable in principle
        # but pinned with no gradient reaching any parameter.
        no_path = tracked and not t.requires_grad
        dead = pinned and not term_alive and not no_path
        out_report[name] = {
            "min": d.min().item() if d.numel() else float("nan"),
            "max": d.max().item() if d.numel() else float("nan"),
            "mean": (
                d.mean().item()
                if d.numel() and d.is_floating_point()
                else float("nan")
            ),
            "frac_at_floor": at_floor,
            "frac_at_ceiling": at_ceiling,
            "pinned_at_floor": pinned_floor,
            "pinned_at_ceiling": pinned_ceiling,
            "requires_grad": t.requires_grad,
            "dead": dead,
            "grads": grads_report,
        }

        # A single pinned endpoint is reported but does not by itself make the
        # report unhealthy. A *correctly saturated* bound looks identical to a
        # broken one at the term level — an upper bound that has legitimately
        # reached 1 is pinned with no gradient, because the clamp at the
        # interval edge stops it (and whether a clamp passes gradient exactly
        # *at* the boundary is a torch-version convention: 2.8 passes 1.0,
        # 2.14 passes 0.0). Keying `healthy` on that would make this tool both
        # noisy and version-dependent.
        #
        # The version-stable signal that a modal term has actually collapsed is
        # **vacuity** — the bound spanning the whole interval, carrying no
        # information at all — which is checked below, per bound pair.
        where = "floor" if pinned_floor else "ceiling"
        if dead:
            warns.append(
                f"term '{name}' is dead: pinned at the {where} "
                f"({floor if pinned_floor else ceiling}) with no gradient "
                f"to any parameter"
            )
        elif pinned and not no_path:
            warns.append(
                f"term '{name}' is saturated at the {where} but still "
                f"differentiable"
            )
        if no_path:
            issues.append(
                f"term '{name}' has no autograd path (requires_grad=False)"
            )

    vacuous: List[str] = []
    interval = ceiling - floor
    for name, w in raw_widths.items():
        if bounds is not False and w.numel() and bool(
            (w - interval).abs().max().item() <= vacuous_atol
        ):
            vacuous.append(name)
            issues.append(
                f"bound '{name}' is vacuous: width is {interval} everywhere — "
                f"the bound has collapsed to [{floor}, {ceiling}] and carries "
                f"no information"
            )

    param_report: Dict[str, Dict[str, Any]] = {}
    any_signal = False
    for pname, p in param_map.items():
        if not p.requires_grad:
            param_report[pname] = {
                "has_grad": False,
                "grad_norm": 0.0,
                "grad_max_abs": 0.0,
                "grad_vanished": True,
                "requires_grad": False,
            }
            issues.append(f"param '{pname}' has requires_grad=False")
            continue
        per_term = [
            out_report[term]["grads"][pname] for term in out_report
        ]
        gmax = max((r["grad_max_abs"] for r in per_term), default=0.0)
        gnorm = max((r["grad_norm"] for r in per_term), default=0.0)
        vanished = gmax <= grad_atol
        param_report[pname] = {
            "has_grad": not vanished,
            "grad_norm": gnorm,
            "grad_max_abs": gmax,
            "grad_vanished": vanished,
            "requires_grad": True,
        }
        if vanished:
            issues.append(
                f"param '{pname}' receives no gradient from any term "
                f"(max |grad| = {gmax:.3e}) — it is disconnected from the "
                f"output"
            )
        else:
            any_signal = True

    if param_map and not any_signal:
        issues.append("no parameter received a usable gradient")

    return {
        "outputs": out_report,
        "params": param_report,
        "vacuous": vacuous,
        "healthy": not issues,
        "issues": issues,
        "warnings": warns,
    }


def assert_has_signal(
    fn: Callable[..., Any],
    params: ParamSpec,
    *args: Any,
    msg: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Raising variant of :func:`gradient_health`, for use in tests.

    Args:
        fn: Callable to diagnose.
        params: Tensors whose gradients are checked.
        *args: Positional arguments forwarded to ``fn``.
        msg: Optional prefix for the raised message.
        **kwargs: Keyword arguments forwarded to :func:`gradient_health`
            and, beyond its own parameters, to ``fn``.

    Returns:
        The report from :func:`gradient_health`, when healthy.

    Raises:
        GradientHealthError: If any output is pinned at an end of the
            truth interval, or any parameter's gradient vanished.
    """
    report = gradient_health(fn, params, *args, **kwargs)
    if not report["healthy"]:
        detail = "\n  - ".join(report["issues"])
        prefix = f"{msg}: " if msg else ""
        raise GradientHealthError(
            f"{prefix}term carries no usable training signal:\n  - {detail}"
        )
    return report


# ---------------------------------------------------------------------------
# Vacuity: is a term satisfied because it is TRUE, or because it is EMPTY?
# ---------------------------------------------------------------------------


def vacuity_report(
    term_fn: Callable[[Tensor], Tensor],
    accessibility: Tensor,
    *,
    margin_atol: float = 1e-4,
) -> Dict[str, Any]:
    r"""Distinguish a term that is *satisfied* from one that is *vacuous*.

    Every :math:`\square`-built quantity is **maximal on the empty relation**:
    an agent that can see nothing vacuously knows everything, because
    ``L_□ = smooth_min((1 - A) + L_φ)`` has no small terms left to find. A
    specification written only in :math:`\square` therefore has a global
    optimum that satisfies every axiom and constrains nothing, and — the part
    that catches people — an :math:`\ell_1` sparsity penalty pushes *toward*
    that optimum rather than against it.

    This evaluates ``term_fn`` twice, on the supplied relation and on the
    all-zero relation of the same shape, and reports the margin between them.
    A term whose value is no better than its own vacuous value is carrying no
    information about the relation, however satisfied it looks.

    This is the tool the :func:`torchmodal.functional.contradiction` docstring
    asks for when it warns that ``L_contra`` "must not be the sole guard
    against a degenerate optimum".

    Args:
        term_fn: Callable taking an accessibility matrix and returning a
            tensor — a bound, a residual, or a scalar score.
        accessibility: The relation to test, ``(..., |W|, |W|)``.
        margin_atol: A margin at or below this counts as vacuous. Default
            1e-4.

    Returns:
        A dict with ``observed_value`` and ``vacuous_value`` (means over the
        returned tensor), ``margin_over_vacuous`` (observed minus vacuous),
        ``vacuous`` (``True`` when the margin is not positive beyond
        ``margin_atol``), and ``direction`` — ``"maximal_when_empty"`` for a
        box-like term, ``"maximal_when_full"`` for a diamond-like one — which
        is the signal for whether a specification is one-sided.

    Example:
        >>> import torch
        >>> from torchmodal import functional as F
        >>> from torchmodal.diagnostics import vacuity_report
        >>> A = torch.rand(6, 6)
        >>> phi = torch.zeros(6, 2)          # unsupported everywhere
        >>> r = vacuity_report(lambda a: F.necessity(phi, a)[:, 0], A)
        >>> r["vacuous"]                      # box on an unsupported prop
        True
    """
    with torch.no_grad():
        empty = torch.zeros_like(accessibility)
        full = torch.ones_like(accessibility)
        observed = term_fn(accessibility)
        vacuous = term_fn(empty)
        saturated = term_fn(full)

        obs = float(observed.mean())
        vac = float(vacuous.mean())
        sat = float(saturated.mean())

    margin = obs - vac
    return {
        "observed_value": obs,
        "vacuous_value": vac,
        "saturated_value": sat,
        "margin_over_vacuous": margin,
        "vacuous": bool(margin <= margin_atol),
        "direction": (
            "maximal_when_empty" if vac >= sat else "maximal_when_full"
        ),
    }


# ---------------------------------------------------------------------------
# Monotonicity of each bound endpoint in the accessibility relation
# ---------------------------------------------------------------------------

#: Which bound endpoints are monotone in ``A``, and which are not.
#:
#: Only the two log-sum-exp aggregators are monotone. The two
#: :func:`~torchmodal.functional.conv_pool` endpoints are not, because
#: ``conv_pool`` is not monotone in its own argument — raising a term already
#: far above the pooled value *lowers* the result (see that function's
#: docstring for the derivative). Measured over 400 random perturbations per
#: endpoint: ``L_box`` 0 violations, ``U_dia`` 0, ``U_box`` 247, ``L_dia``
#: 252.
#:
#: This matters when constructing a soundness argument: "the bound is monotone
#: in ``A``, therefore ..." is available only for the two endpoints below
#: marked ``True``.
MONOTONICITY: Dict[str, Dict[str, Any]] = {
    "necessity.L": {
        "aggregator": "smooth_min",
        "monotone": True,
        "note": "non-decreasing in A: more access can only lower the min's "
                "terms via (1 - A), so the bound tightens predictably",
    },
    "necessity.U": {
        "aggregator": "conv_pool",
        "monotone": False,
        "note": "conv_pool is not monotone in its argument",
    },
    "possibility.L": {
        "aggregator": "conv_pool",
        "monotone": False,
        "note": "conv_pool is not monotone in its argument",
    },
    "possibility.U": {
        "aggregator": "smooth_max",
        "monotone": True,
        "note": "non-decreasing in A",
    },
}


def monotone_in_accessibility(operator: str, endpoint: str) -> bool:
    """Is this bound endpoint monotone in the accessibility relation?

    Args:
        operator: ``"necessity"`` / ``"box"``, or ``"possibility"`` /
            ``"diamond"``.
        endpoint: ``"L"`` or ``"U"`` (case-insensitive).

    Returns:
        ``True`` when the endpoint is monotone in ``A``, so a monotonicity
        argument is available for it.

    Raises:
        KeyError: If the operator/endpoint pair is not recognised.

    Example:
        >>> from torchmodal.diagnostics import monotone_in_accessibility
        >>> monotone_in_accessibility("necessity", "L")
        True
        >>> monotone_in_accessibility("necessity", "U")
        False
    """
    alias = {"box": "necessity", "diamond": "possibility"}
    op = alias.get(operator.lower(), operator.lower())
    key = f"{op}.{endpoint.upper()}"
    if key not in MONOTONICITY:
        raise KeyError(
            f"unknown endpoint {key!r}; expected one of "
            f"{sorted(MONOTONICITY)}"
        )
    return bool(MONOTONICITY[key]["monotone"])
