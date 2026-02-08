"""
torchmodal.nn.accessibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accessibility relation modules for Kripke structures.

Provides three parameterizations (Section 3.3 of the paper):

- **FixedAccessibility**: Static, user-defined binary relation.
- **LearnableAccessibility**: Direct learnable logit matrix → sigmoid.
- **MetricAccessibility**: Scalable metric-learning parameterization
  using latent embeddings with kernel similarity.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "FixedAccessibility",
    "LearnableAccessibility",
    "MetricAccessibility",
    "top_k_mask",
]


def top_k_mask(A: Tensor, k: int) -> Tensor:
    """Apply top-k sparsification to an accessibility matrix.

    For each world (row), only the *k* highest accessibility values
    are kept; all others are zeroed. This reduces computational cost
    from O(|W|²) to O(k·|W|).

    Args:
        A: Accessibility matrix of shape ``(|W|, |W|)``.
        k: Number of neighbors to retain per world.

    Returns:
        Masked accessibility matrix of same shape.
    """
    if k >= A.shape[-1]:
        return A
    topk_vals, _ = torch.topk(A, k, dim=-1)
    threshold = topk_vals[..., -1:] 
    mask = (A >= threshold).float()
    return A * mask


class FixedAccessibility(nn.Module):
    """Fixed (non-learnable) accessibility relation.

    Wraps a user-defined binary relation matrix as a frozen buffer.
    Useful for deductive mode where the logical structure is known
    (e.g., Sudoku constraints, temporal flow, grammatical rules).

    Args:
        relation: Binary accessibility matrix of shape ``(|W|, |W|)``.
            Values should be 0 or 1.
        top_k: If set, apply top-k masking. Default ``None``.

    Example::

        >>> # Sudoku: cells in same row/col/box are accessible
        >>> R = build_sudoku_accessibility(9)
        >>> access = FixedAccessibility(R)
        >>> A = access()  # (81, 81) binary matrix
    """

    def __init__(
        self,
        relation: Tensor,
        top_k: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("relation", relation.float())
        self.top_k = top_k

    @property
    def num_worlds(self) -> int:
        return self.relation.shape[0]

    def forward(self) -> Tensor:
        """Returns the accessibility matrix ``(|W|, |W|)``."""
        A = self.relation
        if self.top_k is not None:
            A = top_k_mask(A, self.top_k)
        return A

    def extra_repr(self) -> str:
        return (
            f"num_worlds={self.num_worlds}, "
            f"top_k={self.top_k}"
        )


class LearnableAccessibility(nn.Module):
    """Learnable accessibility relation via direct logit matrix.

    Parameterizes R as a matrix of learnable logits passed through
    sigmoid: ``A = σ(logits)``. Suitable for small-to-medium world
    sets (|W| ≤ ~1000).

    The parameter space is O(|W|²).

    Args:
        num_worlds: Number of possible worlds |W|.
        init_bias: Initial bias for logits. Negative values encode a
            "prior of distrust" (default -2.0).
        reflexive: If ``True``, enforce self-accessibility (diagonal = 1).
            Default ``True``.
        top_k: If set, apply top-k masking after sigmoid. Default ``None``.

    Example::

        >>> access = LearnableAccessibility(7, reflexive=True)
        >>> A = access()  # (7, 7) matrix in [0, 1]
    """

    def __init__(
        self,
        num_worlds: int,
        init_bias: float = -2.0,
        reflexive: bool = True,
        top_k: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._num_worlds = num_worlds
        self.reflexive = reflexive
        self.top_k = top_k

        self.logits = nn.Parameter(
            torch.full((num_worlds, num_worlds), init_bias)
        )

        if reflexive:
            # Initialize diagonal to high logit (self-trust)
            with torch.no_grad():
                self.logits.diagonal().fill_(5.0)

    @property
    def num_worlds(self) -> int:
        return self._num_worlds

    def forward(self) -> Tensor:
        """Returns the accessibility matrix ``(|W|, |W|)`` in [0, 1]."""
        A = torch.sigmoid(self.logits)

        if self.reflexive:
            # Clamp diagonal to 1.0
            A = A.clone()
            A.fill_diagonal_(1.0)

        if self.top_k is not None:
            A = top_k_mask(A, self.top_k)

        return A

    def extra_repr(self) -> str:
        return (
            f"num_worlds={self._num_worlds}, "
            f"reflexive={self.reflexive}, "
            f"top_k={self.top_k}"
        )


class MetricAccessibility(nn.Module):
    """Scalable metric-learning accessibility relation.

    Maps each world to a latent embedding and computes accessibility
    via a kernel function:

    .. math::
        A(w_i, w_j) = \\sigma\\bigl(h_{w_i}^\\top h_{w_j}\\bigr)

    This reduces the parameter space from O(|W|²) to O(d·|W|) and
    enables scaling to |W| = 20,000+ on a single GPU.

    The encoder can optionally accept external features per world.

    Args:
        num_worlds: Number of possible worlds |W|.
        embed_dim: Embedding dimension *d*. Default 64.
        input_dim: If provided, the encoder takes external features of
            this dimension. Otherwise, uses learnable embeddings.
        hidden_dim: Hidden dimension of the encoder MLP. Default 128.
        reflexive: Enforce self-accessibility. Default ``True``.
        top_k: Top-k masking. Default ``None``.

    Example::

        >>> access = MetricAccessibility(1000, embed_dim=64)
        >>> A = access()  # (1000, 1000) accessibility matrix
        >>> # With external features:
        >>> access = MetricAccessibility(100, embed_dim=32, input_dim=384)
        >>> A = access(features)  # features: (100, 384)
    """

    def __init__(
        self,
        num_worlds: int,
        embed_dim: int = 64,
        input_dim: Optional[int] = None,
        hidden_dim: int = 128,
        reflexive: bool = True,
        top_k: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._num_worlds = num_worlds
        self.embed_dim = embed_dim
        self.reflexive = reflexive
        self.top_k = top_k

        if input_dim is not None:
            # Encoder from external features
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
            )
            self.embeddings = None
        else:
            # Learnable embeddings per world
            self.encoder = None
            self.embeddings = nn.Parameter(
                torch.randn(num_worlds, embed_dim) * 0.01
            )

    @property
    def num_worlds(self) -> int:
        return self._num_worlds

    def forward(self, features: Optional[Tensor] = None) -> Tensor:
        """Compute the accessibility matrix.

        Args:
            features: Optional external features ``(|W|, input_dim)``.
                Required if ``input_dim`` was set at construction.

        Returns:
            Accessibility matrix ``(|W|, |W|)`` in [0, 1].
        """
        if self.encoder is not None:
            if features is None:
                raise ValueError(
                    "MetricAccessibility with input_dim requires features"
                )
            h = self.encoder(features)
        else:
            h = self.embeddings

        # Kernel: inner product → sigmoid
        A = torch.sigmoid(h @ h.t())

        if self.reflexive:
            A = A.clone()
            A.fill_diagonal_(1.0)

        if self.top_k is not None:
            A = top_k_mask(A, self.top_k)

        return A

    def extra_repr(self) -> str:
        return (
            f"num_worlds={self._num_worlds}, "
            f"embed_dim={self.embed_dim}, "
            f"reflexive={self.reflexive}, "
            f"top_k={self.top_k}"
        )
