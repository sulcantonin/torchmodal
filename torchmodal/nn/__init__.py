"""
torchmodal.nn
~~~~~~~~~~~~~

Neural network modules for differentiable modal logic.

This subpackage provides ``nn.Module`` implementations for:

- **Operators**: Differentiable soft aggregations (softmin, softmax, conv_pool)
- **Connectives**: Propositional logic (AND, OR, NOT, IMPLIES)
- **Modal**: Core modal neurons (Necessity □, Possibility ♢)
- **Accessibility**: Kripke accessibility relations (Fixed, Learnable, Metric)
"""

from torchmodal.nn.operators import Softmin, Softmax, ConvPool
from torchmodal.nn.connectives import (
    Negation,
    Conjunction,
    Disjunction,
    Implication,
)
from torchmodal.nn.modal import Necessity, Possibility
from torchmodal.nn.accessibility import (
    FixedAccessibility,
    LearnableAccessibility,
    MetricAccessibility,
    top_k_mask,
)

__all__ = [
    # Aggregation operators
    "Softmin",
    "Softmax",
    "ConvPool",
    # Propositional connectives
    "Negation",
    "Conjunction",
    "Disjunction",
    "Implication",
    # Modal operators
    "Necessity",
    "Possibility",
    # Accessibility relations
    "FixedAccessibility",
    "LearnableAccessibility",
    "MetricAccessibility",
    "top_k_mask",
]
