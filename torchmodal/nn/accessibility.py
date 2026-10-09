"""
torchmodal.nn.accessibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accessibility relation modules for Kripke structures.

Provides four parameterizations:

- **FixedAccessibility**: Static, user-defined binary relation.
- **LearnableAccessibility**: Direct learnable logit matrix → sigmoid.
  O(|W|²) parameters — suitable for |W| ≤ ~1000.
- **MetricAccessibility**: Metric-learning parameterization using latent
  embeddings with inner-product kernel.  O(d·|W|) parameters — scales
  to |W| = 20,000+.
- **AttentionAccessibility**: Multi-head query/key attention scores over
  world representations, squashed per edge by a sigmoid.  O(d²) parameters
  — suitable when worlds have rich feature representations and the relation
  is asymmetric.  Addresses the reviewer concern (R1) that the kernel
  parameterization is not the only sub-quadratic alternative.

**Top-k is not an accessibility-module concern.** Earlier releases took a
``top_k`` argument here and zeroed all but the *k* largest entries of each
row of ``A`` before the modal operators saw it. That was unsound: □ and ♢
aggregate ``(1 - A) + L`` and ``A + U - 1``, so choosing neighbours by
``A`` alone can drop the world whose ``L`` / ``U`` carries the extremum,
and the zeroed entries still enter the log-sum-exp with mass that grows
with ``|W|`` and drives every bound to ``[0, 1]``. Top-k aggregation now
lives on :class:`torchmodal.nn.Necessity` / :class:`~torchmodal.nn.Possibility`
(``top_k=``), which select the *k* extreme aggregation terms per endpoint.
``top_k`` here is deprecated and ignored (with a ``DeprecationWarning``).

What remains available here is ``sparsify=k``: a *deliberately sparsified
relation* in which each world accesses only its *k* most accessible
worlds. That is a modelling choice — it defines a different Kripke frame —
not an aggregation optimisation, and the operators then reason soundly
about the sparsified frame.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "FixedAccessibility",
    "LearnableAccessibility",
    "MetricAccessibility",
    "AttentionAccessibility",
    "top_k_mask",
]


_TOP_K_DEPRECATION = (
    "{cls}(top_k=...) is deprecated and no longer masks the relation (the "
    "argument is ignored). Zeroing all but the k largest entries of A before "
    "aggregation was unsound: the modal operators aggregate (1 - A) + L and "
    "A + U - 1, so selecting neighbours by A alone can drop the world whose "
    "L / U carries the extremum (Theorem 1 violated), and the zeroed entries "
    "still enter the log-sum-exp and drive the bounds to [0, 1] as |W| grows. "
    "Top-k aggregation now lives on the operators: pass top_k=k to "
    "torchmodal.nn.Necessity / Possibility (or functional.necessity / "
    "possibility), which select the k extreme aggregation terms per "
    "endpoint. If you deliberately want a *sparsified relation* — each world "
    "accesses only its k most accessible worlds, i.e. a different Kripke "
    "frame — use sparsify=k instead."
)


def _warn_top_k_deprecated(module: nn.Module) -> None:
    warnings.warn(
        _TOP_K_DEPRECATION.format(cls=type(module).__name__),
        DeprecationWarning,
        stacklevel=3,
    )


def top_k_mask(A: Tensor, k: int) -> Tensor:
    """Sparsify an accessibility matrix to its *k* largest entries per row.

    For each world (row), only the *k* highest accessibility values are
    kept; all others are set to 0, i.e. those worlds become
    *inaccessible*. This defines a different (sparser) Kripke frame and is
    what the ``sparsify=`` option of the accessibility modules applies.

    .. warning::
       This is a **modelling choice, not an aggregation optimisation**.
       It does not reduce the cost of □ / ♢ (the operators still aggregate
       over the full ``(|W|, |W|)`` row) and it must not be used to emulate
       top-k aggregation: neighbours are chosen by ``A`` alone rather than
       by the aggregated terms, and the zeroed entries still enter the
       log-sum-exp with term ``1 + L`` each, so the bounds drift to
       ``[0, 1]`` as ``|W|`` grows. For sound top-k aggregation with a
       ``tau * log(k)`` gap use ``top_k=`` on
       :func:`torchmodal.functional.necessity` /
       :func:`~torchmodal.functional.possibility` or the
       :class:`torchmodal.nn.Necessity` / :class:`~torchmodal.nn.Possibility`
       modules.

    Args:
        A: Accessibility matrix of shape ``(|W|, |W|)``.
        k: Number of neighbors to retain per world.

    Returns:
        Sparsified accessibility matrix of the same shape.
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
        sparsify: If set, keep only the ``sparsify`` largest entries of
            each row and make every other world inaccessible (a different,
            sparser Kripke frame — a modelling choice, see
            :func:`top_k_mask`). Default ``None``.
        top_k: **Deprecated and ignored.** Masking ``A`` before
            aggregation was unsound; pass ``top_k`` to
            :class:`torchmodal.nn.Necessity` / :class:`~torchmodal.nn.Possibility`
            instead, or use ``sparsify`` for a sparsified relation.

    Example::

        >>> # Sudoku: cells in same row/col/box are accessible
        >>> R = build_sudoku_accessibility(9)
        >>> access = FixedAccessibility(R)
        >>> A = access()  # (81, 81) binary matrix
    """

    #: Declared so mypy knows the registered buffer is a Tensor;
    #: ``nn.Module.__getattr__`` otherwise widens it to
    #: ``Union[Tensor, Module]`` and every use has to be narrowed.
    relation: Tensor

    def __init__(
        self,
        relation: Tensor,
        top_k: Optional[int] = None,
        sparsify: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("relation", relation.float())
        if top_k is not None:
            _warn_top_k_deprecated(self)
        self.sparsify = sparsify

    @property
    def num_worlds(self) -> int:
        return int(self.relation.shape[0])

    def forward(self) -> Tensor:
        """Returns the accessibility matrix ``(|W|, |W|)``."""
        A = self.relation
        if self.sparsify is not None:
            A = top_k_mask(A, self.sparsify)
        return A

    def extra_repr(self) -> str:
        return (
            f"num_worlds={self.num_worlds}, "
            f"sparsify={self.sparsify}"
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
        sparsify: If set, keep only the ``sparsify`` largest entries of
            each row after the sigmoid and make every other world
            inaccessible (a sparser Kripke frame — a modelling choice, see
            :func:`top_k_mask`). Default ``None``.
        top_k: **Deprecated and ignored.** Masking ``A`` before
            aggregation was unsound; pass ``top_k`` to
            :class:`torchmodal.nn.Necessity` / :class:`~torchmodal.nn.Possibility`
            instead, or use ``sparsify`` for a sparsified relation.

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
        sparsify: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._num_worlds = num_worlds
        self.reflexive = reflexive
        if top_k is not None:
            _warn_top_k_deprecated(self)
        self.sparsify = sparsify

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

        if self.sparsify is not None:
            A = top_k_mask(A, self.sparsify)

        return A

    def extra_repr(self) -> str:
        return (
            f"num_worlds={self._num_worlds}, "
            f"reflexive={self.reflexive}, "
            f"sparsify={self.sparsify}"
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
        sparsify: If set, keep only the ``sparsify`` largest entries of
            each row and make every other world inaccessible (a sparser
            Kripke frame — a modelling choice, see :func:`top_k_mask`).
            Default ``None``.
        top_k: **Deprecated and ignored.** Masking ``A`` before
            aggregation was unsound; pass ``top_k`` to
            :class:`torchmodal.nn.Necessity` / :class:`~torchmodal.nn.Possibility`
            instead, or use ``sparsify`` for a sparsified relation.

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
        sparsify: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._num_worlds = num_worlds
        self.embed_dim = embed_dim
        self.reflexive = reflexive
        if top_k is not None:
            _warn_top_k_deprecated(self)
        self.sparsify = sparsify

        # Exactly one of these is populated; annotate before the branch so
        # each is declared once.
        self.encoder: Optional[nn.Sequential]
        self.embeddings: Optional[nn.Parameter]
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

        if self.sparsify is not None:
            A = top_k_mask(A, self.sparsify)

        return A

    def extra_repr(self) -> str:
        return (
            f"num_worlds={self._num_worlds}, "
            f"embed_dim={self.embed_dim}, "
            f"reflexive={self.reflexive}, "
            f"sparsify={self.sparsify}"
        )


class AttentionAccessibility(nn.Module):
    r"""Attention-based accessibility relation.

    Scores every ordered pair of worlds with multi-head query/key attention
    over their feature vectors and squashes the score with a sigmoid:

    .. math::
        A_{ij} = \sigma\Bigl(\frac{1}{h}\sum_{\text{heads}}
            \frac{(W_q x_i)^\top (W_k x_j)}{\sqrt{d_h}} + b\Bigr)

    Unlike :class:`MetricAccessibility` (a symmetric inner-product kernel),
    separate query and key projections make the relation **asymmetric**:
    world *i* can see *j* without *j* seeing *i*. The parameter count is
    O(d²), independent of |W|, so it suits worlds with rich feature
    representations (sentence embeddings, state encodings).

    .. note::
       **Sigmoid, not softmax.** Up to 0.8.0 this module returned the
       row-softmax weights of ``nn.MultiheadAttention`` directly, so every
       row summed to 1 and no entry could exceed ``1/|W|`` on average —
       with 50 worlds the largest off-diagonal entry was 0.07. Fed to a □
       neuron, such a relation makes nearly every world inaccessible and
       the operator close to vacuous. An accessibility relation is a set
       of *independent* graded edges, so each score is now squashed on its
       own and ``A_ij = 1`` is reachable for every pair. The state-dict
       layout changed with this (``q_proj`` / ``k_proj`` instead of
       ``attn`` / ``proj``); checkpoints from earlier releases do not load.

    Args:
        input_dim: Dimension of per-world feature vectors. Must be
            divisible by ``num_heads``.
        num_heads: Number of attention heads; the per-head scores are
            averaged. Default 4.
        reflexive: Enforce self-accessibility. Default ``True``.
        init_bias: Constant added to every score before the sigmoid, so the
            relation can start from a "prior of distrust" the way
            :class:`LearnableAccessibility` does with its ``init_bias``.
            Default 0.0 (every edge starts near 0.5).
        sparsify: If set, keep only the ``sparsify`` largest entries of
            each row and make every other world inaccessible (a sparser
            Kripke frame — a modelling choice, see :func:`top_k_mask`).
            Default ``None``.
        top_k: **Deprecated and ignored.** Masking ``A`` before
            aggregation was unsound; pass ``top_k`` to
            :class:`torchmodal.nn.Necessity` / :class:`~torchmodal.nn.Possibility`
            instead, or use ``sparsify`` for a sparsified relation.

    Example::

        >>> access = AttentionAccessibility(input_dim=384, num_heads=4)
        >>> features = torch.randn(7, 384)  # 7 worlds, 384-d features
        >>> A = access(features)  # (7, 7) accessibility matrix
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        reflexive: bool = True,
        top_k: Optional[int] = None,
        sparsify: Optional[int] = None,
        init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError(
                f"input_dim ({input_dim}) must be divisible by num_heads "
                f"({num_heads})"
            )
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.reflexive = reflexive
        self.init_bias = init_bias
        if top_k is not None:
            _warn_top_k_deprecated(self)
        self.sparsify = sparsify

        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)

    def forward(self, features: Tensor) -> Tensor:
        """Compute the accessibility matrix from world features.

        Args:
            features: Per-world features ``(|W|, input_dim)``.

        Returns:
            Accessibility matrix ``(|W|, |W|)`` in [0, 1].
        """
        n = features.shape[0]
        head_dim = self.input_dim // self.num_heads
        # (|W|, d) -> (heads, |W|, d_h)
        q = self.q_proj(features).reshape(n, self.num_heads, head_dim)
        k = self.k_proj(features).reshape(n, self.num_heads, head_dim)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(head_dim)  # (h, W, W)
        A = torch.sigmoid(scores.mean(dim=0) + self.init_bias)

        if self.reflexive:
            A = A.clone()
            A.fill_diagonal_(1.0)

        if self.sparsify is not None:
            A = top_k_mask(A, self.sparsify)

        return A

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, "
            f"num_heads={self.num_heads}, "
            f"reflexive={self.reflexive}, "
            f"init_bias={self.init_bias}, "
            f"sparsify={self.sparsify}"
        )
