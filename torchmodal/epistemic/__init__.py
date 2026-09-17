"""Group-knowledge operators and frame-axiom auditing for multi-agent models.

Single-agent knowledge is :func:`torchmodal.functional.necessity` routed through
one agent's relation. This subpackage adds what a single relation cannot say:
what a *group* knows (:func:`everybody_knows`, :func:`mutual_knowledge`,
:func:`distributed_knowledge`, :func:`common_knowledge`), and whether a learned
relation actually has the frame structure it appears to (:func:`frame_audit`).

Two design notes that decide how these are used, both measured rather than
assumed:

- **Fold with Gödel when you iterate.** Łukasiewicz conjunction is
  sub-idempotent, so an iterated group fold reaches zero at the second level
  and stops producing gradients. :func:`and_bounds` defaults to Gödel, which is
  also the tightest sound choice.
- **Train against** :func:`mutual_knowledge`, **not** :func:`common_knowledge`.
  A fixpoint that iterates to convergence drives its lower bound onto the floor,
  because every modal level costs at least
  :func:`~torchmodal.functional.box_width_entropy`. The bounded tower is what
  the fixpoint approximates and it keeps its gradient.
"""

from __future__ import annotations

from torchmodal.epistemic.frame_axioms import AxiomReport, frame_audit, shuffled_null
from torchmodal.epistemic.operators import (
    and_bounds,
    common_knowledge,
    distributed_knowledge,
    everybody_knows,
    mutual_knowledge,
    pooled_accessibility,
)

__all__ = [
    "and_bounds",
    "everybody_knows",
    "mutual_knowledge",
    "distributed_knowledge",
    "common_knowledge",
    "pooled_accessibility",
    "frame_audit",
    "shuffled_null",
    "AxiomReport",
]
