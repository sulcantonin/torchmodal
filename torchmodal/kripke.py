"""
torchmodal.kripke
~~~~~~~~~~~~~~~~~

Kripke model and formula graph for Modal Logical Neural Networks.

A Kripke model M = ⟨W, R, V⟩ consists of:
- W: a finite set of possible worlds
- R: a binary accessibility relation on W
- V: a valuation function assigning truth values to propositions in worlds

This module provides :class:`KripkeModel`, the central data structure
that manages worlds, propositions, accessibility, and formula evaluation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from torchmodal import functional as F
from torchmodal.nn.accessibility import (
    FixedAccessibility,
    LearnableAccessibility,
    MetricAccessibility,
)
from torchmodal.nn.modal import Necessity, Possibility

__all__ = [
    "KripkeModel",
    "Proposition",
]


class Proposition(nn.Module):
    """A named atomic proposition with truth bounds across worlds.

    Each proposition stores ``[L, U]`` bounds per world in [0, 1].

    Args:
        name: Human-readable name for the proposition.
        num_worlds: Number of worlds |W|.
        learnable: If ``True``, bounds are learnable parameters (for CSPs
            / satisfiability mode). If ``False``, they are buffers set
            externally. Default ``True``.
        init: Initial value for both L and U bounds. Default 0.5
            (maximum uncertainty).
    """

    def __init__(
        self,
        name: str,
        num_worlds: int,
        learnable: bool = True,
        init: float = 0.5,
    ) -> None:
        super().__init__()
        self.name = name
        self._num_worlds = num_worlds

        bounds = torch.full((num_worlds, 2), init)

        if learnable:
            # Store as logits, apply sigmoid for [0,1] guarantee
            self._logits = nn.Parameter(torch.zeros(num_worlds, 2))
            with torch.no_grad():
                # Initialize logits to match desired init value
                self._logits.fill_(torch.logit(torch.tensor(init)).item())
        else:
            self.register_buffer("_bounds", bounds)
            self._logits = None

    @property
    def num_worlds(self) -> int:
        return self._num_worlds

    @property
    def bounds(self) -> Tensor:
        """Truth bounds ``(|W|, 2)`` with ``[L, U]`` per world."""
        if self._logits is not None:
            raw = torch.sigmoid(self._logits)
            # Ensure L <= U
            L = torch.min(raw[..., 0], raw[..., 1])
            U = torch.max(raw[..., 0], raw[..., 1])
            return torch.stack([L, U], dim=-1)
        return self._bounds

    @property
    def lower(self) -> Tensor:
        """Lower bounds ``(|W|,)``."""
        return self.bounds[..., 0]

    @property
    def upper(self) -> Tensor:
        """Upper bounds ``(|W|,)``."""
        return self.bounds[..., 1]

    def set_bounds(self, bounds: Tensor) -> None:
        """Set bounds externally (only for non-learnable propositions).

        Args:
            bounds: Tensor of shape ``(|W|, 2)``.
        """
        if self._logits is not None:
            raise RuntimeError(
                "Cannot set bounds on a learnable proposition. "
                "Use the optimizer to update bounds."
            )
        self._bounds.copy_(bounds)

    def set_world(
        self, world_idx: int, lower: float, upper: float
    ) -> None:
        """Set truth bounds for a single world.

        Args:
            world_idx: Index of the world.
            lower: Lower truth bound.
            upper: Upper truth bound.
        """
        if self._logits is not None:
            raise RuntimeError("Cannot set bounds on learnable proposition.")
        self._bounds[world_idx, 0] = lower
        self._bounds[world_idx, 1] = upper

    def extra_repr(self) -> str:
        learnable = self._logits is not None
        return (
            f"name='{self.name}', "
            f"num_worlds={self._num_worlds}, "
            f"learnable={learnable}"
        )


class KripkeModel(nn.Module):
    """Differentiable Kripke model M = ⟨W, R, V⟩.

    Central data structure for MLNN computation. Manages:
    - Possible worlds and their propositions (valuation V)
    - Accessibility relation R (fixed or learnable)
    - Modal operator evaluation (□, ♢)
    - Contradiction loss computation

    The model supports two learning modes:

    - **Deductive** (fixed R, learnable V): Enforces known axioms by
      updating proposition truth values through gradient descent.
    - **Inductive** (fixed V, learnable R): Discovers relational structure
      by learning the accessibility relation from data.

    Args:
        num_worlds: Number of possible worlds |W|.
        accessibility: Accessibility relation module. One of
            :class:`FixedAccessibility`, :class:`LearnableAccessibility`,
            or :class:`MetricAccessibility`.
        tau: Temperature for modal operators. Default 0.1.
        world_names: Optional list of human-readable world names.

    Example::

        >>> from torchmodal import KripkeModel
        >>> from torchmodal.nn import LearnableAccessibility
        >>> model = KripkeModel(
        ...     num_worlds=3,
        ...     accessibility=LearnableAccessibility(3),
        ... )
        >>> model.add_proposition("p", learnable=True)
        >>> model.add_proposition("q", learnable=False)
    """

    def __init__(
        self,
        num_worlds: int,
        accessibility: Union[
            FixedAccessibility, LearnableAccessibility, MetricAccessibility
        ],
        tau: float = 0.1,
        world_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self._num_worlds = num_worlds
        self.accessibility = accessibility
        self.tau = tau
        self.box = Necessity(tau=tau)
        self.diamond = Possibility(tau=tau)
        self.propositions = nn.ModuleDict()

        if world_names is not None:
            assert len(world_names) == num_worlds
        self.world_names = world_names or [
            f"w{i}" for i in range(num_worlds)
        ]

    @property
    def num_worlds(self) -> int:
        return self._num_worlds

    @property
    def num_propositions(self) -> int:
        return len(self.propositions)

    def add_proposition(
        self,
        name: str,
        learnable: bool = True,
        init: float = 0.5,
    ) -> Proposition:
        """Add an atomic proposition to the model.

        Args:
            name: Proposition name (must be unique).
            learnable: Whether bounds are learnable. Default ``True``.
            init: Initial truth value. Default 0.5.

        Returns:
            The created :class:`Proposition` module.
        """
        if name in self.propositions:
            raise ValueError(f"Proposition '{name}' already exists")
        prop = Proposition(name, self._num_worlds, learnable=learnable, init=init)
        self.propositions[name] = prop
        return prop

    def get_proposition(self, name: str) -> Proposition:
        """Retrieve a proposition by name."""
        return self.propositions[name]

    def get_accessibility(
        self, features: Optional[Tensor] = None
    ) -> Tensor:
        """Compute the current accessibility matrix.

        Args:
            features: Optional features for :class:`MetricAccessibility`.

        Returns:
            Accessibility matrix ``(|W|, |W|)`` in [0, 1].
        """
        if isinstance(self.accessibility, MetricAccessibility):
            return self.accessibility(features)
        return self.accessibility()

    def necessity(
        self,
        prop_name: str,
        accessibility: Optional[Tensor] = None,
    ) -> Tensor:
        """Evaluate □ϕ (necessity) for a proposition.

        Args:
            prop_name: Name of the proposition.
            accessibility: Pre-computed accessibility matrix. If ``None``,
                computed from the model's accessibility module.

        Returns:
            Truth bounds ``(|W|, 2)`` for □ϕ.
        """
        if accessibility is None:
            accessibility = self.get_accessibility()
        prop = self.propositions[prop_name]
        return self.box(prop.bounds, accessibility)

    def possibility(
        self,
        prop_name: str,
        accessibility: Optional[Tensor] = None,
    ) -> Tensor:
        """Evaluate ♢ϕ (possibility) for a proposition.

        Args:
            prop_name: Name of the proposition.
            accessibility: Pre-computed accessibility matrix. If ``None``,
                computed from the model's accessibility module.

        Returns:
            Truth bounds ``(|W|, 2)`` for ♢ϕ.
        """
        if accessibility is None:
            accessibility = self.get_accessibility()
        prop = self.propositions[prop_name]
        return self.diamond(prop.bounds, accessibility)

    def contradiction_loss(self) -> Tensor:
        """Compute the total contradiction loss across all propositions.

        .. math::
            \\mathcal{L}_{\\text{contra}} =
                \\sum_{w \\in W} \\sum_\\phi \\max(0,\\; L_{\\phi,w} - U_{\\phi,w})

        Returns:
            Scalar contradiction loss.
        """
        total = torch.tensor(0.0, device=self._get_device())
        for prop in self.propositions.values():
            total = total + F.contradiction(prop.bounds)
        return total

    def all_bounds(self) -> Dict[str, Tensor]:
        """Return a dict mapping proposition names to their bounds."""
        return {name: p.bounds for name, p in self.propositions.items()}

    def _get_device(self) -> torch.device:
        """Infer the device from model parameters."""
        for p in self.parameters():
            return p.device
        return torch.device("cpu")

    def forward(
        self, features: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Compute accessibility and return all proposition bounds.

        Args:
            features: Optional features for MetricAccessibility.

        Returns:
            Dictionary of proposition name → bounds ``(|W|, 2)``.
        """
        # Trigger accessibility computation (ensures it's in the graph)
        _ = self.get_accessibility(features)
        return self.all_bounds()

    def extra_repr(self) -> str:
        return (
            f"num_worlds={self._num_worlds}, "
            f"tau={self.tau}, "
            f"num_propositions={self.num_propositions}"
        )
