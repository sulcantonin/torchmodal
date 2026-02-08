"""
torchmodal — Differentiable Modal Logic for PyTorch
====================================================

A PyTorch library implementing Modal Logical Neural Networks (MLNNs),
enabling differentiable reasoning over necessity and possibility by
integrating neural networks with Kripke semantics from modal logic.

The library provides:

- **Differentiable modal operators**: □ (necessity) and ♢ (possibility)
  as neural network modules that aggregate truth values across possible
  worlds.

- **Flexible accessibility relations**: Fixed, learnable (direct matrix),
  and scalable metric-learning parameterizations.

- **Kripke model management**: Complete framework for managing worlds,
  propositions, and formula evaluation.

- **Higher-level modal systems**: Ready-to-use epistemic (K_a),
  doxastic (B_a), and temporal (G, F) logic operators.

- **Loss functions**: Contradiction loss, combined modal loss,
  sparsity regularization, and crystallization loss.

- **Inference**: Upward-downward bound propagation algorithm.

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

Reference: Sulc (2026), "Modal Logical Neural Networks", ICML.
"""

__version__ = "0.1.0"

# Core functional API
from torchmodal import functional
from torchmodal import nn

# High-level modules
from torchmodal.kripke import KripkeModel, Proposition
from torchmodal.losses import (
    ContradictionLoss,
    ModalLoss,
    SparsityLoss,
    CrystallizationLoss,
    AxiomRegularization,
)
from torchmodal.inference import (
    FormulaGraph,
    FormulaNode,
    FormulaType,
    upward_downward,
)
from torchmodal.systems import (
    EpistemicOperator,
    DoxasticOperator,
    TemporalOperator,
    MultiAgentKripke,
)
from torchmodal.utils import (
    anneal_temperature,
    build_ring_accessibility,
    build_sudoku_accessibility,
    build_grid_accessibility,
    decode_one_hot,
    bounds_to_labels,
)

__all__ = [
    # Version
    "__version__",
    # Subpackages
    "functional",
    "nn",
    # Kripke model
    "KripkeModel",
    "Proposition",
    # Losses
    "ContradictionLoss",
    "ModalLoss",
    "SparsityLoss",
    "CrystallizationLoss",
    "AxiomRegularization",
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
