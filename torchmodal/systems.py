"""
torchmodal.systems
~~~~~~~~~~~~~~~~~~

Higher-level modal logic systems built on top of the core operators.

Provides ready-to-use modules for specific modal logics:

- **Epistemic Logic** (K_a): Agent *a* knows ϕ iff ϕ is true in all
  worlds accessible to *a*.
- **Doxastic Logic** (B_a): Agent *a* believes ϕ, where beliefs may
  differ from reality.
- **Temporal Logic** (G, F): Globally ϕ (necessity over future states)
  and Finally ϕ (possibility of eventual truth).
- **Composite Operators** (K∘G, K∘F): Nested modal operators for
  complex multi-agent temporal reasoning.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
    "EpistemicOperator",
    "DoxasticOperator",
    "TemporalOperator",
    "MultiAgentKripke",
]


class EpistemicOperator(nn.Module):
    r"""Epistemic knowledge operator K_a.

    ``K_a(ϕ)`` asserts "agent *a* knows ϕ" — ϕ is true in all worlds
    accessible to agent *a*.

    This is equivalent to □ restricted to agent *a*'s accessibility
    row in the relation matrix.

    Args:
        tau: Temperature. Default 0.1.

    Example::

        >>> K = EpistemicOperator()
        >>> # agent_accessibility: (|W|,) row for agent a
        >>> knowledge = K(prop_bounds, agent_accessibility)
    """

    def __init__(self, tau: float = 0.1) -> None:
        super().__init__()
        self.box = Necessity(tau=tau)

    def forward(
        self,
        prop_bounds: Tensor,
        agent_accessibility: Tensor,
    ) -> Tensor:
        """Evaluate K_a(ϕ) for a single agent.

        Args:
            prop_bounds: ``(|W|, 2)`` or ``(|W|,)`` truth bounds for ϕ.
            agent_accessibility: ``(|W|,)`` accessibility weights from
                agent *a* to all worlds.

        Returns:
            Scalar or ``(2,)`` bounds for K_a(ϕ).
        """
        # Reshape agent accessibility to (1, |W|) for single-source eval
        A_row = agent_accessibility.unsqueeze(0)  # (1, |W|)
        result = self.box(prop_bounds, A_row)       # (1, 2) or (1,)
        return result.squeeze(0)


class DoxasticOperator(nn.Module):
    r"""Doxastic belief operator B_a.

    ``B_a(ϕ)`` asserts "agent *a* believes ϕ" — ϕ is true in all
    worlds compatible with *a*'s beliefs, which may differ from reality.

    Structurally identical to :class:`EpistemicOperator`, but
    semantically distinct: epistemic accessibility requires *veridical*
    knowledge (ϕ must actually hold), while doxastic accessibility
    permits *false beliefs*.

    This distinction is captured by the accessibility relation:
    epistemic relations are typically reflexive (T axiom: K_a(ϕ) → ϕ),
    while doxastic relations may not be.

    Args:
        tau: Temperature. Default 0.1.
    """

    def __init__(self, tau: float = 0.1) -> None:
        super().__init__()
        self.box = Necessity(tau=tau)

    def forward(
        self,
        prop_bounds: Tensor,
        agent_accessibility: Tensor,
    ) -> Tensor:
        """Evaluate B_a(ϕ) for a single agent.

        Args:
            prop_bounds: ``(|W|, 2)`` or ``(|W|,)`` truth bounds.
            agent_accessibility: ``(|W|,)`` accessibility row for agent *a*.

        Returns:
            Bounds for B_a(ϕ).
        """
        A_row = agent_accessibility.unsqueeze(0)
        result = self.box(prop_bounds, A_row)
        return result.squeeze(0)


class TemporalOperator(nn.Module):
    r"""Temporal logic operators G (Globally) and F (Finally).

    - **G(ϕ)** ≡ □ϕ over temporal accessibility: ϕ holds at all future
      time steps. Uses necessity over forward-reachable states.
    - **F(ϕ)** ≡ ♢ϕ over temporal accessibility: ϕ holds at some
      future time step. Uses possibility over forward-reachable states.

    Args:
        num_steps: Number of discrete time steps.
        tau: Temperature. Default 0.1.

    Example::

        >>> temporal = TemporalOperator(num_steps=5)
        >>> A_temporal = temporal.build_forward_accessibility()
        >>> globally_phi = temporal.globally(prop_bounds, A_temporal)
        >>> finally_phi = temporal.finally_(prop_bounds, A_temporal)
    """

    def __init__(self, num_steps: int, tau: float = 0.1) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.box = Necessity(tau=tau)
        self.diamond = Possibility(tau=tau)

    def build_forward_accessibility(
        self,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        """Build a forward-time accessibility matrix.

        Creates a lower-triangular-inverted matrix where each time step
        can access all current and future steps.

        Returns:
            ``(num_steps, num_steps)`` binary accessibility matrix.
        """
        n = self.num_steps
        # i can access j if j >= i (forward-time)
        A = torch.zeros(n, n, device=device)
        for i in range(n):
            A[i, i:] = 1.0
        return A

    def globally(
        self, prop_bounds: Tensor, temporal_accessibility: Tensor
    ) -> Tensor:
        """G(ϕ) — globally, ϕ holds at all accessible future states.

        Args:
            prop_bounds: ``(num_steps, 2)`` or ``(num_steps,)``.
            temporal_accessibility: ``(num_steps, num_steps)``.

        Returns:
            Bounds for G(ϕ).
        """
        return self.box(prop_bounds, temporal_accessibility)

    def finally_(
        self, prop_bounds: Tensor, temporal_accessibility: Tensor
    ) -> Tensor:
        """F(ϕ) — finally, ϕ holds at some accessible future state.

        Args:
            prop_bounds: ``(num_steps, 2)`` or ``(num_steps,)``.
            temporal_accessibility: ``(num_steps, num_steps)``.

        Returns:
            Bounds for F(ϕ).
        """
        return self.diamond(prop_bounds, temporal_accessibility)


class MultiAgentKripke(nn.Module):
    r"""Multi-agent Kripke structure with temporal and epistemic dimensions.

    Creates a spacetime state space S = W × T where:
    - W is a set of agent worlds
    - T is a set of time steps

    Supports composite modal operators like K∘G (epistemic-temporal
    knowledge) and K∘F (epistemic-temporal possibility).

    This mirrors the architecture used in the Diplomacy and CaSiNo
    experiments from the paper.

    Args:
        num_agents: Number of agents |W|.
        num_steps: Number of time steps |T|.
        tau: Temperature. Default 0.1.
        learnable_epistemic: If ``True``, the epistemic accessibility is
            learnable. Default ``True``.
        init_bias: Initial bias for learnable epistemic logits.
            Default -2.0 ("prior of distrust").
    """

    def __init__(
        self,
        num_agents: int,
        num_steps: int = 1,
        tau: float = 0.1,
        learnable_epistemic: bool = True,
        init_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.tau = tau
        self.num_states = num_agents * num_steps

        # Temporal accessibility (fixed: forward-time flow)
        self.temporal = TemporalOperator(num_steps, tau=tau)
        A_temporal = self._build_spacetime_temporal()
        self.register_buffer("A_temporal", A_temporal)

        # Epistemic accessibility (learnable)
        if learnable_epistemic:
            self.epistemic_access = LearnableAccessibility(
                num_agents,
                init_bias=init_bias,
                reflexive=True,
            )
        else:
            # Identity: each agent only sees itself
            eye = torch.eye(num_agents)
            self.epistemic_access = FixedAccessibility(eye)

        # Modal operators
        self.box = Necessity(tau=tau)
        self.diamond = Possibility(tau=tau)

    def _build_spacetime_temporal(self) -> Tensor:
        """Build temporal accessibility over the full spacetime grid.

        State (a, t) can access state (a, t') for t' >= t.
        """
        n_s = self.num_states
        A = torch.zeros(n_s, n_s)
        for a in range(self.num_agents):
            for t in range(self.num_steps):
                src = a * self.num_steps + t
                for t2 in range(t, self.num_steps):
                    dst = a * self.num_steps + t2
                    A[src, dst] = 1.0
        return A

    def get_epistemic_accessibility(
        self, features: Optional[Tensor] = None
    ) -> Tensor:
        """Get the epistemic (agent-to-agent) accessibility matrix.

        Returns:
            ``(num_agents, num_agents)`` matrix in [0, 1].
        """
        if isinstance(self.epistemic_access, MetricAccessibility):
            return self.epistemic_access(features)
        return self.epistemic_access()

    def get_full_accessibility(
        self, features: Optional[Tensor] = None
    ) -> Tensor:
        """Get the combined spacetime accessibility matrix.

        Combines temporal accessibility (within-agent time flow)
        with epistemic accessibility (between-agent trust).

        Returns:
            ``(num_states, num_states)`` matrix in [0, 1].
        """
        A_epi = self.get_epistemic_accessibility(features)
        A_full = torch.zeros(
            self.num_states, self.num_states,
            device=A_epi.device,
        )

        for a in range(self.num_agents):
            for b in range(self.num_agents):
                trust = A_epi[a, b]
                for t in range(self.num_steps):
                    for t2 in range(t, self.num_steps):
                        src = a * self.num_steps + t
                        dst = b * self.num_steps + t2
                        A_full[src, dst] = trust * self.A_temporal[src, a * self.num_steps + t2]

        return A_full

    def K(
        self,
        prop_bounds: Tensor,
        features: Optional[Tensor] = None,
    ) -> Tensor:
        """Epistemic knowledge operator over agents.

        Args:
            prop_bounds: ``(num_agents, 2)`` bounds per agent.
            features: Optional features for metric accessibility.

        Returns:
            ``(num_agents, 2)`` bounds for K(ϕ).
        """
        A_epi = self.get_epistemic_accessibility(features)
        return self.box(prop_bounds, A_epi)

    def G(self, prop_bounds: Tensor) -> Tensor:
        """Temporal globally operator.

        Args:
            prop_bounds: ``(num_states, 2)`` bounds over spacetime.

        Returns:
            ``(num_states, 2)`` bounds for G(ϕ).
        """
        return self.box(prop_bounds, self.A_temporal)

    def F(self, prop_bounds: Tensor) -> Tensor:
        """Temporal finally operator.

        Args:
            prop_bounds: ``(num_states, 2)`` bounds over spacetime.

        Returns:
            ``(num_states, 2)`` bounds for F(ϕ).
        """
        return self.diamond(prop_bounds, self.A_temporal)

    def K_G(
        self,
        prop_bounds: Tensor,
        features: Optional[Tensor] = None,
    ) -> Tensor:
        """Composite K∘G: agent knows ϕ holds globally.

        First applies G (temporal necessity), then K (epistemic).

        Args:
            prop_bounds: ``(num_states, 2)`` bounds.
            features: Optional features for metric accessibility.

        Returns:
            ``(num_states, 2)`` bounds for K(G(ϕ)).
        """
        g_bounds = self.G(prop_bounds)
        A_full = self.get_full_accessibility(features)
        return self.box(g_bounds, A_full)

    def K_F(
        self,
        prop_bounds: Tensor,
        features: Optional[Tensor] = None,
    ) -> Tensor:
        """Composite K∘F: agent knows ϕ holds eventually.

        First applies F (temporal possibility), then K (epistemic).

        Args:
            prop_bounds: ``(num_states, 2)`` bounds.
            features: Optional features for metric accessibility.

        Returns:
            ``(num_states, 2)`` bounds for K(F(ϕ)).
        """
        f_bounds = self.F(prop_bounds)
        A_full = self.get_full_accessibility(features)
        return self.box(f_bounds, A_full)

    def forward(
        self,
        prop_bounds: Tensor,
        operator: str = "K",
        features: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply a named modal operator.

        Args:
            prop_bounds: Truth bounds tensor.
            operator: One of ``"K"``, ``"G"``, ``"F"``, ``"K_G"``, ``"K_F"``.
            features: Optional features for metric accessibility.

        Returns:
            Transformed truth bounds.
        """
        ops = {
            "K": lambda b: self.K(b, features),
            "G": self.G,
            "F": self.F,
            "K_G": lambda b: self.K_G(b, features),
            "K_F": lambda b: self.K_F(b, features),
        }
        if operator not in ops:
            raise ValueError(
                f"Unknown operator '{operator}'. "
                f"Choose from: {list(ops.keys())}"
            )
        return ops[operator](prop_bounds)

    def extra_repr(self) -> str:
        return (
            f"num_agents={self.num_agents}, "
            f"num_steps={self.num_steps}, "
            f"num_states={self.num_states}, "
            f"tau={self.tau}"
        )
