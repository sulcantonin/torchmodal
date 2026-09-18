"""
torchmodal — Differentiable Modal Logic for PyTorch
====================================================

A PyTorch library implementing Modal Logic Neural Networks (MLNNs),
enabling differentiable reasoning over necessity and possibility by
integrating neural networks with Kripke semantics from modal logic.

The library provides:

- **Differentiable modal operators**: □ (necessity), ♢ (possibility),
  and U (until) as neural network modules that aggregate truth values
  across possible worlds.

- **Flexible accessibility relations**: Fixed, learnable (direct matrix),
  metric-learning, and attention-based parameterizations.

- **Kripke model management**: Complete framework for managing worlds,
  propositions, and formula evaluation.

- **Higher-level modal systems**: Ready-to-use epistemic (K_a),
  doxastic (B_a), and temporal (G, F, U) logic operators.

- **Loss functions**: Contradiction loss, combined modal loss,
  sparsity regularization, crystallization loss, and a SemanticLoss
  baseline (Xu et al., 2018) for comparison with non-modal NeSy
  approaches.

- **Inference**: Upward-downward bound propagation algorithm with
  cycle detection. The downward pass inverts every propositional
  connective on both endpoints and each modal operator on the one
  endpoint that factorises per world (□ lower, ♢ upper); it iterates
  to a fixed point and warns if the iteration budget runs out.

Quick Start::

    import torch
    import torchmodal
    from torchmodal import nn, KripkeModel

    # Create a 3-world Kripke model with learnable accessibility
    model = KripkeModel(
        num_worlds=3,
        accessibility=nn.LearnableAccessibility(3),
    )

    # Add propositions
    model.add_proposition("safe", learnable=True)

    # Evaluate necessity: "safe is necessarily true"
    A = model.get_accessibility()
    box_safe = model.necessity("safe", A)

    # Compute contradiction loss
    loss = model.contradiction_loss()

Reference: Sulc & Naddour (2026), "Modal Logic Neural Networks",
Proceedings of the 20th Conference on Neurosymbolic Learning and
Reasoning (NeSy 2026), PMLR vol. 284 — oral.
https://openreview.net/pdf?id=uLOdtBm0Cx
"""

__version__ = "0.7.0"

# Core functional API
from torchmodal import diagnostics, epistemic, functional, nn
from torchmodal.diagnostics import (
    MONOTONICITY,
    GradientHealthError,
    assert_has_signal,
    gradient_health,
    monotone_in_accessibility,
    vacuity_report,
)
from torchmodal.inference import (
    FormulaGraph,
    FormulaNode,
    FormulaType,
    upward_downward,
)

# High-level modules
from torchmodal.kripke import KripkeModel, Proposition
from torchmodal.losses import (
    AxiomRegularization,
    ContradictionLoss,
    CrystallizationLoss,
    ModalLoss,
    SemanticLoss,
    SparsityLoss,
)
from torchmodal.systems import (
    DoxasticOperator,
    EpistemicOperator,
    MultiAgentKripke,
    TemporalOperator,
)
from torchmodal.utils import (
    anneal_temperature,
    bounds_to_labels,
    build_grid_accessibility,
    build_ring_accessibility,
    build_sudoku_accessibility,
    decode_one_hot,
)

__all__ = [
    # Version
    "__version__",
    # Subpackages
    "functional",
    "epistemic",
    "nn",
    "diagnostics",
    # Diagnostics
    "gradient_health",
    "assert_has_signal",
    "GradientHealthError",
    "vacuity_report",
    "monotone_in_accessibility",
    "MONOTONICITY",
    # Kripke model
    "KripkeModel",
    "Proposition",
    # Losses
    "ContradictionLoss",
    "ModalLoss",
    "SparsityLoss",
    "CrystallizationLoss",
    "AxiomRegularization",
    "SemanticLoss",
    # Inference
    "FormulaGraph",
    "FormulaNode",
    "FormulaType",
    "upward_downward",
    # Systems
    "EpistemicOperator",
    "DoxasticOperator",
    "TemporalOperator",
    "MultiAgentKripke",
    # Utilities
    "anneal_temperature",
    "build_ring_accessibility",
    "build_sudoku_accessibility",
    "build_grid_accessibility",
    "decode_one_hot",
    "bounds_to_labels",
]
