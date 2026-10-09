"""
torchmodal.kripke
~~~~~~~~~~~~~~~~~

Kripke model and formula graph for Modal Logic Neural Networks.

A Kripke model M = ⟨W, R, V⟩ consists of:
- W: a finite set of possible worlds
- R: a binary accessibility relation on W
- V: a valuation function assigning truth values to propositions in worlds

This module provides :class:`KripkeModel`, the central data structure
that manages worlds, propositions, accessibility, and formula evaluation.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Mapping, Optional, Union, cast

import torch
import torch.nn as nn
from torch import Tensor

from torchmodal import functional as F
from torchmodal.nn.accessibility import (
    AttentionAccessibility,
    FixedAccessibility,
    LearnableAccessibility,
    MetricAccessibility,
)
from torchmodal.nn.modal import Necessity, Possibility

__all__ = [
    "KripkeModel",
    "Proposition",
]

#: Learnable propositions are parameterised as logits. ``init`` values are
#: clamped into ``[_LOGIT_EPS, 1 - _LOGIT_EPS]`` so that ``init=0.0`` / ``1.0``
#: do not produce infinite logits with zero gradient.
_LOGIT_EPS = 1e-4


class Proposition(nn.Module):
    """A named atomic proposition with truth bounds across worlds.

    Each proposition stores ``[L, U]`` bounds per world in [0, 1].

    For learnable propositions, use :meth:`set_bounds_value` to temporarily
    set bounds in-place (e.g. in adversarial or minimax setups).

    Args:
        name: Human-readable name for the proposition.
        num_worlds: Number of worlds |W|.
        learnable: If ``True``, bounds are learnable parameters (for CSPs
            / satisfiability mode). If ``False``, they are buffers set
            externally. Default ``True``.
        init: Initial value for both L and U bounds. Default 0.5
            (maximum uncertainty). For a learnable proposition the value is
            clamped to ``[1e-4, 1 - 1e-4]``: exactly 0 or 1 would set the
            logits to ±inf and leave the proposition with no gradient.
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

        self._bounds: Tensor
        self._logits: Optional[nn.Parameter]
        if learnable:
            # Store as logits, apply sigmoid for [0,1] guarantee. An ``init``
            # of exactly 0 or 1 has logit ±inf, which sigmoid maps back to the
            # requested value but with a gradient of exactly 0 forever — the
            # proposition would be silently untrainable. Clamp to the nearest
            # value that still carries gradient (sigmoid'(±9.2) ≈ 1e-4, which
            # an adaptive optimiser such as Adam turns into a usable step).
            self._logits = nn.Parameter(torch.zeros(num_worlds, 2))
            init_c = min(max(float(init), _LOGIT_EPS), 1.0 - _LOGIT_EPS)
            with torch.no_grad():
                self._logits.fill_(torch.logit(torch.tensor(init_c)).item())
        else:
            self.register_buffer("_bounds", bounds)
            self._logits = None

    @property
    def num_worlds(self) -> int:
        return self._num_worlds

    @property
    def bounds(self) -> Tensor:
        """Truth bounds ``(|W|, 2)`` with ``[L, U]`` per world.

        For a learnable proposition the two sigmoid outputs are sorted into
        ``(min, max)``, so ``L <= U`` holds by construction and the pair is
        never contradictory on its own. Contradictions arise only against
        bounds *derived* for the proposition by inference — see
        :meth:`KripkeModel.contradiction_loss`.
        """
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

    def set_bounds_value(self, bounds: Tensor) -> None:
        """Set current bounds in-place.

        For learnable propositions, updates internal logits so that the next
        :attr:`bounds` read returns (approximately) ``bounds``. Use this to
        temporarily inject bounds (e.g. adversary values in minimax) without
        removing learnability. For non-learnable propositions, equivalent to
        :meth:`set_bounds`.

        Args:
            bounds: Tensor of shape ``(num_worlds, 2)`` in [0, 1].
        """
        bounds = bounds.clamp(1e-7, 1.0 - 1e-7)
        if self._logits is not None:
            with torch.no_grad():
                self._logits.data.copy_(torch.logit(bounds))
        else:
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

    Use :meth:`get_proposition` to get a proposition by name, :meth:`get_bounds`
    for its truth bounds, and :meth:`all_bounds` for a dict of all bounds.

    Args:
        num_worlds: Number of possible worlds |W|.
        accessibility: Accessibility relation module. One of
            :class:`FixedAccessibility`, :class:`LearnableAccessibility`,
            :class:`MetricAccessibility` or :class:`AttentionAccessibility`
            (the last two take ``features`` in :meth:`get_accessibility`).
        tau: Temperature for modal operators. Default 0.1.
        world_names: Optional list of human-readable world names.
        top_k: Top-k aggregation for the model's □ / ♢ operators (see
            :class:`torchmodal.nn.Necessity`): each endpoint aggregates
            only its ``top_k`` extreme terms. This replaces the deprecated
            ``top_k`` of the accessibility modules. Default ``None``.

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
            FixedAccessibility,
            LearnableAccessibility,
            MetricAccessibility,
            AttentionAccessibility,
        ],
        tau: float = 0.1,
        world_names: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._num_worlds = num_worlds
        self.accessibility = accessibility
        self.tau = tau
        self.top_k = top_k
        self.box = Necessity(tau=tau, top_k=top_k)
        self.diamond = Possibility(tau=tau, top_k=top_k)
        self.propositions = nn.ModuleDict()

        if world_names is not None and len(world_names) != num_worlds:
            raise ValueError(
                f"world_names has {len(world_names)} entries for "
                f"{num_worlds} worlds"
            )
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
        """Retrieve a proposition by name.

        ``nn.ModuleDict`` is typed as returning a bare ``Module``, so the
        cast records what the container actually holds — every entry is put
        there by :meth:`add_proposition`.
        """
        return cast(Proposition, self.propositions[name])

    def get_bounds(self, name: str) -> Tensor:
        """Return truth bounds for proposition ``name``. Shape ``(|W|, 2)``."""
        return self.get_proposition(name).bounds

    def get_accessibility(
        self, features: Optional[Tensor] = None
    ) -> Tensor:
        """Compute the current accessibility matrix.

        Args:
            features: Optional features for :class:`MetricAccessibility`
                or :class:`AttentionAccessibility`.

        Returns:
            Accessibility matrix ``(|W|, |W|)`` in [0, 1].
        """
        if isinstance(
            self.accessibility, (MetricAccessibility, AttentionAccessibility)
        ):
            return cast(Tensor, self.accessibility(features))
        return cast(Tensor, self.accessibility())

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
        return cast(Tensor, self.box(prop.bounds, accessibility))

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
        return cast(Tensor, self.diamond(prop.bounds, accessibility))

    def contradiction_loss(
        self, derived: Optional[Mapping[str, Tensor]] = None
    ) -> Tensor:
        r"""Contradiction loss between asserted and derived bounds.

        .. math::
            \mathcal{L}_{\text{contra}} =
                \sum_{w \in W} \sum_\phi \max(0,\; L_{\phi,w} - U_{\phi,w})

        **Where a contradiction can come from.** A proposition's own bounds
        are never contradictory on their own: a *learnable* proposition
        stores a sigmoid pair sorted into ``(min, max)`` (see
        :attr:`Proposition.bounds`), so ``L <= U`` by construction, and a
        non-learnable one holds whatever was set. The contradiction that the
        MLNN objective is about arises, as in LNN, when a proposition's
        *asserted* interval meets an interval *derived* for the same
        proposition by inference — ``□safe`` evaluated on a reflexive frame
        bounds ``safe`` itself, an axiom tightens it from above, a sibling
        formula tightens it from below — and the two cannot both hold.

        Pass those derived intervals as ``derived``: for each name the
        asserted and derived intervals are intersected,
        ``L = max(L_asserted, L_derived)``, ``U = min(U_asserted, U_derived)``,
        and ``relu(L - U)`` is summed. This is differentiable in both the
        proposition and whatever produced the derived bounds (typically the
        accessibility relation), so it trains either learning mode.

        Without ``derived`` the method sums the raw contradiction of each
        proposition's own bounds, which is identically zero on a model whose
        propositions are all learnable. Up to 0.8.0 that silent zero was the
        only behaviour; it now emits a ``UserWarning`` in that case so a
        training loop cannot build on a loss that never moves.

        Args:
            derived: Optional mapping from proposition name to ``(|W|, 2)``
                bounds derived for it — for example the output of
                :meth:`necessity` on a reflexive frame, or the tightened
                bounds from :func:`torchmodal.inference.upward_downward`.
                Names not in the model raise ``KeyError``.

        Returns:
            Scalar contradiction loss.

        Example::

            >>> A = model.get_accessibility()
            >>> # On a reflexive frame, □safe must not exceed safe.
            >>> loss = model.contradiction_loss(
            ...     {"safe": model.necessity("safe", A)}
            ... )
        """
        total = torch.tensor(0.0, device=self._get_device())
        if derived is None:
            if self.propositions and all(
                self.get_proposition(n)._logits is not None
                for n in self.propositions
            ):
                warnings.warn(
                    "KripkeModel.contradiction_loss() was called without "
                    "`derived` bounds on a model whose propositions are all "
                    "learnable. Learnable bounds are sorted so L <= U always "
                    "holds and this loss is identically zero. Pass the bounds "
                    "derived by inference, e.g. "
                    "model.contradiction_loss({'p': model.necessity('p', A)}).",
                    UserWarning,
                    stacklevel=2,
                )
            for name in self.propositions:
                total = total + F.contradiction(
                    self.get_proposition(name).bounds
                )
            return total

        for name, d in derived.items():
            asserted = self.get_bounds(name)
            L = torch.max(asserted[..., 0], d[..., 0])
            U = torch.min(asserted[..., 1], d[..., 1])
            total = total + F.contradiction(L, U)
        return total

    def all_bounds(self) -> Dict[str, Tensor]:
        """Return a dict mapping proposition names to their bounds."""
        return {
            name: self.get_proposition(name).bounds
            for name in self.propositions
        }

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
            f"top_k={self.top_k}, "
            f"num_propositions={self.num_propositions}"
        )
