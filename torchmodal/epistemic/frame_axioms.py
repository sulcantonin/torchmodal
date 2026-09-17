r"""Frame-axiom audit for a learned accessibility relation, vacuity-corrected.

A learned relation :math:`A_\theta` may or may not satisfy the frame conditions
that give a modal logic its character — reflexivity (T), seriality (D),
symmetry (B), transitivity (4), the Euclidean axiom (5). Measuring that
post-hoc is the *audit regime*: impose nothing, train, then read the recovered
structure off the matrix.

**Why a plain average is not enough for axioms 4 and 5.** Both are
implications, and Łukasiewicz conjunction sends the antecedent to 0 whenever
:math:`A_{uv} + A_{vw} \le 1`, while :math:`\mathrm{impl}(0, b) = 1` for every
``b``. So any triple with two weak links counts as satisfied whatever the third
link does, and a sparse or low-variance relation scores near 1 for the wrong
reason. This module therefore never returns a bare average for 4 and 5: it
reports the coverage, the non-vacuous score, and a shape-matched null, so a
score can be read as evidence rather than as an artefact of shape.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor

__all__ = ["AxiomReport", "frame_audit", "shuffled_null"]


class AxiomReport(dict):
    """Per-axiom audit result.

    Keys: ``score`` (mean satisfaction), ``coverage`` (fraction of triples that
    actually test the axiom; ``None`` for non-implication axioms),
    ``score_non_vacuous`` (mean over testing triples only — ``None`` when
    coverage is 0, never a meaningless 1.0), and ``null`` (the same score on
    shape-matched shuffled relations).

    ``credited`` is ``True`` only when the non-vacuous score clearly exceeds the
    null at non-negligible coverage — the condition under which the axiom is
    evidence of structure rather than of shape.
    """

    @property
    def credited(self) -> bool:
        cov = self.get("coverage")
        nv = self.get("score_non_vacuous")
        null = self.get("null")
        if null is None:
            return False
        if cov is not None and (cov < 0.1 or nv is None):
            return False
        value = nv if nv is not None else self.get("score")
        return bool(value is not None and value > null + 0.01)


def _luk_and(a: Tensor, b: Tensor) -> Tensor:
    return torch.clamp(a + b - 1.0, min=0.0)


def _luk_impl(a: Tensor, b: Tensor) -> Tensor:
    return torch.clamp(1.0 - a + b, max=1.0)


def _triple_terms(A: Tensor, axiom: str) -> tuple[Tensor, Tensor]:
    """Return ``(antecedent, satisfaction)`` over all triples for 4 or 5."""
    # a_uv * a_vw -> a_uw  (transitivity);  a_uv * a_uw -> a_vw  (Euclidean)
    a_uv = A.unsqueeze(2)  # (u, v, 1)
    if axiom == "transitive":
        a_second = A.unsqueeze(0)  # (1, v, w)
        consequent = A.unsqueeze(1)  # (u, 1, w)
    elif axiom == "euclidean":
        a_second = A.unsqueeze(1)  # (u, 1, w)
        consequent = A.unsqueeze(0)  # (1, v, w)
    else:  # pragma: no cover
        raise ValueError(axiom)
    n = A.shape[0]
    antecedent = _luk_and(a_uv.expand(n, n, n), a_second.expand(n, n, n))
    sat = _luk_impl(antecedent, consequent.expand(n, n, n))
    return antecedent, sat


def shuffled_null(A: Tensor, shuffles: int, axiom: str, generator=None) -> float:
    """Mean score on ``shuffles`` shape-matched copies of ``A``.

    Each copy keeps every world's self-weight and its multiset of outgoing
    weights, and only permutes which targets those weights point at. The null
    hypothesis is therefore "this relation has no genuine structure of this
    kind; its score is fixed by how strong each world's links are, not by where
    they point".
    """
    n = A.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=A.device)
    scores = []
    for _ in range(shuffles):
        B = A.clone()
        for u in range(n):
            row = A[u][off[u]]
            perm = torch.randperm(row.numel(), generator=generator, device=A.device)
            B[u][off[u]] = row[perm]
        scores.append(_score(B, axiom))
    return float(sum(scores) / len(scores)) if scores else float("nan")


def _score(A: Tensor, axiom: str) -> float:
    if axiom == "reflexive":
        return float(A.diagonal().mean())
    if axiom == "serial":
        return float(A.max(dim=1).values.mean())
    if axiom == "symmetric":
        return float((1.0 - (A - A.T).abs()).mean())
    _, sat = _triple_terms(A, axiom)
    return float(sat.mean())


def frame_audit(
    A: Tensor,
    shuffles: int = 100,
    coverage_eps: float = 1e-6,
    generator=None,
) -> Dict[str, AxiomReport]:
    r"""Audit a relation against T, D, B, 4 and 5, with vacuity correction.

    Returns one :class:`AxiomReport` per axiom, keyed ``reflexive``, ``serial``,
    ``symmetric``, ``transitive``, ``euclidean``.

    Reflexivity and seriality are plain averages over the diagonal and the row
    maxima and need no correction — shuffling leaves them unchanged, so their
    ``null`` is reported as ``None``. Symmetry gets a shuffled null but has no
    coverage, being a comparison rather than an implication. Transitivity and
    the Euclidean axiom get all four numbers, and ``score_non_vacuous`` is
    ``None`` when no triple tests the axiom.

    Args:
        A: ``(|W|, |W|)`` relation in [0, 1].
        shuffles: Shape-matched null samples. Default 100.
        coverage_eps: A triple counts as testing the axiom when its antecedent
            exceeds this. Default 1e-6.
        generator: Optional ``torch.Generator`` for reproducible shuffles.

    Returns:
        Mapping from axiom name to :class:`AxiomReport`.

    Example:
        >>> import torch
        >>> from torchmodal.epistemic import frame_audit
        >>> A = torch.eye(4)
        >>> frame_audit(A, shuffles=5)["reflexive"]["score"]
        1.0
    """
    A = A.detach()
    out: Dict[str, AxiomReport] = {}

    for axiom in ("reflexive", "serial"):
        out[axiom] = AxiomReport(
            score=_score(A, axiom), coverage=None, score_non_vacuous=None, null=None
        )

    out["symmetric"] = AxiomReport(
        score=_score(A, "symmetric"),
        coverage=None,
        score_non_vacuous=None,
        null=shuffled_null(A, shuffles, "symmetric", generator),
    )

    for axiom in ("transitive", "euclidean"):
        antecedent, sat = _triple_terms(A, axiom)
        testing = antecedent > coverage_eps
        coverage = float(testing.float().mean())
        nv: Optional[float] = (
            float(sat[testing].mean()) if bool(testing.any()) else None
        )
        out[axiom] = AxiomReport(
            score=float(sat.mean()),
            coverage=coverage,
            score_non_vacuous=nv,
            null=shuffled_null(A, shuffles, axiom, generator),
        )
    return out
