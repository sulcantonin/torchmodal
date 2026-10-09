"""
torchmodal.inference
~~~~~~~~~~~~~~~~~~~~

Upward-Downward inference algorithm for MLNN (Appendix B.1).

The core inference procedure propagates truth bounds through the
logical formula graph in two passes:

1. **Upward pass** (leaves → root): Computes bounds bottom-up using
   differentiable operators following topological order.
2. **Downward pass** (root → leaves): Refines bounds top-down using
   inverse operator constraints.

Each iteration can only *tighten* bounds (increase L or decrease U),
creating a monotonic bounded sequence that converges to a unique
fixed point for acyclic formula graphs.

**Why the two passes are iterated rather than run once.** A single
upward sweep is exact for the upward system alone, and a single downward
sweep is exact for the downward system given fixed parent bounds, but the
*joint* fixed point generally needs more than one round: the downward pass
tightens a leaf that the upward pass has already consumed, so any sibling
formula sharing that leaf is stale until the next sweep. Since sharing
subformulae across asserted axioms is precisely what the downward pass is
for, :func:`upward_downward` iterates to ``convergence_threshold`` and warns
if ``max_iterations`` is exhausted first.

**Downward coverage.** ``NEGATION``, ``CONJUNCTION``, ``DISJUNCTION`` and
``IMPLICATION`` invert on both endpoints of both children (for implication:
modus ponens on ``L_b``, modus tollens on ``U_a``, and the two upper-side
rules that apply where the parent's upper bound is below 1). ``NECESSITY``
and ``POSSIBILITY``
invert on one endpoint each — a universally quantified lower bound
distributes over the neighbourhood (□) and an existential upper bound caps
every disjunct (♢), while the opposite directions constrain an aggregate
without saying which neighbour realises it and so have no canonical
per-world form. ``UNTIL`` has no downward rule at all: its backward DP
couples every time step.
"""

from __future__ import annotations

import warnings
from enum import Enum
from typing import Dict, List, Optional

import torch
from torch import Tensor

from torchmodal import functional as F

__all__ = [
    "FormulaNode",
    "FormulaType",
    "FormulaGraph",
    "upward_downward",
]


class FormulaType(Enum):
    """Types of nodes in a formula graph.

    Each type corresponds to a logical operator in the MLNN language:

    - ``ATOMIC``: Leaf proposition (no children).
    - ``NEGATION``: ¬ϕ (one child).
    - ``CONJUNCTION``: ϕ ∧ ψ (two children), Łukasiewicz t-norm.
    - ``DISJUNCTION``: ϕ ∨ ψ (two children), Łukasiewicz t-conorm.
    - ``IMPLICATION``: ϕ → ψ (two children), Łukasiewicz implication.
    - ``NECESSITY``: □ϕ (one child), aggregates over accessible worlds.
    - ``POSSIBILITY``: ♢ϕ (one child), aggregates over accessible worlds.
    - ``UNTIL``: ϕ U ψ (two children), backward DP over time steps.
    """

    ATOMIC = "atomic"
    NEGATION = "neg"
    CONJUNCTION = "and"
    DISJUNCTION = "or"
    IMPLICATION = "implies"
    NECESSITY = "box"
    POSSIBILITY = "diamond"
    UNTIL = "until"


class FormulaNode:
    """A node in the formula dependency graph.

    Args:
        name: Unique name for this formula node.
        ftype: Type of formula (atomic, connective, or modal).
        children: Names of child formula nodes.
    """

    def __init__(
        self,
        name: str,
        ftype: FormulaType,
        children: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.ftype = ftype
        self.children = children or []

    def __repr__(self) -> str:
        return (
            f"FormulaNode('{self.name}', {self.ftype.value}, "
            f"children={self.children})"
        )


class FormulaGraph:
    """Directed acyclic graph of logical formulae.

    Manages the dependency structure for upward-downward inference.
    Nodes are added in any order; topological sort is computed
    automatically.

    Example::

        >>> graph = FormulaGraph()
        >>> graph.add_atomic("p")
        >>> graph.add_atomic("q")
        >>> graph.add_conjunction("p_and_q", "p", "q")
        >>> graph.add_necessity("box_p_and_q", "p_and_q")
        >>> order = graph.topological_order()
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, FormulaNode] = {}
        self._topo_cache: Optional[List[str]] = None

    def _invalidate_cache(self) -> None:
        self._topo_cache = None

    def add_atomic(self, name: str) -> FormulaNode:
        """Add an atomic proposition node (leaf)."""
        node = FormulaNode(name, FormulaType.ATOMIC)
        self.nodes[name] = node
        self._invalidate_cache()
        return node

    def add_negation(self, name: str, child: str) -> FormulaNode:
        """Add a negation node: ¬child."""
        node = FormulaNode(name, FormulaType.NEGATION, [child])
        self.nodes[name] = node
        self._invalidate_cache()
        return node

    def add_conjunction(
        self, name: str, left: str, right: str
    ) -> FormulaNode:
        """Add a conjunction node: left ∧ right."""
        node = FormulaNode(name, FormulaType.CONJUNCTION, [left, right])
        self.nodes[name] = node
        self._invalidate_cache()
        return node

    def add_disjunction(
        self, name: str, left: str, right: str
    ) -> FormulaNode:
        """Add a disjunction node: left ∨ right."""
        node = FormulaNode(name, FormulaType.DISJUNCTION, [left, right])
        self.nodes[name] = node
        self._invalidate_cache()
        return node

    def add_implication(
        self, name: str, antecedent: str, consequent: str
    ) -> FormulaNode:
        """Add an implication node: antecedent → consequent."""
        node = FormulaNode(
            name, FormulaType.IMPLICATION, [antecedent, consequent]
        )
        self.nodes[name] = node
        self._invalidate_cache()
        return node

    def add_necessity(self, name: str, child: str) -> FormulaNode:
        """Add a necessity node: □child."""
        node = FormulaNode(name, FormulaType.NECESSITY, [child])
        self.nodes[name] = node
        self._invalidate_cache()
        return node

    def add_possibility(self, name: str, child: str) -> FormulaNode:
        """Add a possibility node: ♢child."""
        node = FormulaNode(name, FormulaType.POSSIBILITY, [child])
        self.nodes[name] = node
        self._invalidate_cache()
        return node

    def add_until(
        self, name: str, hold: str, goal: str
    ) -> FormulaNode:
        """Add an Until node: hold U goal.

        ``hold`` must remain true until ``goal`` becomes true.
        """
        node = FormulaNode(name, FormulaType.UNTIL, [hold, goal])
        self.nodes[name] = node
        self._invalidate_cache()
        return node

    def is_acyclic(self) -> bool:
        """Check whether the formula graph is a DAG.

        The upward-downward inference algorithm requires an acyclic
        dependency graph (Theorem 2 in the paper).  Call this method
        to verify the invariant before running inference.

        Returns:
            ``True`` if the graph is acyclic, ``False`` otherwise.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {name: WHITE for name in self.nodes}

        def has_cycle(name: str) -> bool:
            color[name] = GRAY
            for child in self.nodes[name].children:
                if child not in color:
                    continue
                if color[child] == GRAY:
                    return True
                if color[child] == WHITE and has_cycle(child):
                    return True
            color[name] = BLACK
            return False

        return not any(
            has_cycle(n) for n in self.nodes if color[n] == WHITE
        )

    def topological_order(self) -> List[str]:
        """Compute topological order (leaves first)."""
        if self._topo_cache is not None:
            return self._topo_cache

        visited = set()
        order = []

        def dfs(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            node = self.nodes[name]
            for child in node.children:
                dfs(child)
            order.append(name)

        for name in self.nodes:
            dfs(name)

        self._topo_cache = order
        return order


def upward_downward(
    graph: FormulaGraph,
    bounds: Dict[str, Tensor],
    accessibility: Tensor,
    tau: float = 0.1,
    max_iterations: int = 10,
    convergence_threshold: float = 1e-6,
    top_k: Optional[int] = None,
) -> Dict[str, Tensor]:
    """Run upward-downward inference on a formula graph.

    Iteratively tightens truth bounds until convergence or maximum
    iterations. Each iteration consists of:

    1. Upward pass: propagate from leaves to root in topological order.
    2. Downward pass: propagate constraints from root to leaves.

    Args:
        graph: The formula dependency graph.
        bounds: Dict mapping formula names to initial bounds ``(|W|, 2)``.
        accessibility: Accessibility matrix ``(|W|, |W|)`` in [0, 1].
        tau: Temperature for modal operators. Default 0.1.
        max_iterations: Maximum inference iterations. Default 10.
        convergence_threshold: Stop if max bound change < threshold.
        top_k: Passed to :func:`torchmodal.functional.necessity` /
            :func:`~torchmodal.functional.possibility` in the upward pass:
            each modal endpoint aggregates only its ``top_k`` extreme
            terms. The downward modal rules are unaffected — they are
            sound for any sound parent bound, which the top-k upward
            bounds are. Default ``None`` (full aggregation).

    Returns:
        A **new** dict mapping formula names to tightened bounds ``(|W|, 2)``.
        The ``bounds`` argument is not modified (up to 0.8.0 it was updated
        in place as well as returned, so a second call on the same dict
        started from the previous fixed point).
    """
    order = graph.topological_order()  # leaves first
    # Shallow copy: entries are replaced, never written in place, so this is
    # enough to leave the caller's dict untouched.
    bounds = dict(bounds)

    for iteration in range(max_iterations):
        max_change = 0.0

        # --- Upward pass (leaves → root) ---
        for name in order:
            node = graph.nodes[name]
            old_bounds = bounds[name].clone()

            if node.ftype == FormulaType.ATOMIC:
                continue  # keep current bounds

            elif node.ftype == FormulaType.NEGATION:
                child_b = bounds[node.children[0]]
                neg = F.negation(child_b)
                new_b = neg.flip(-1)  # swap L and U

            elif node.ftype == FormulaType.CONJUNCTION:
                a_b = bounds[node.children[0]]
                b_b = bounds[node.children[1]]
                L = F.conjunction(a_b[..., 0], b_b[..., 0])
                U = torch.min(a_b[..., 1], b_b[..., 1])
                new_b = torch.stack([L, U], dim=-1)

            elif node.ftype == FormulaType.DISJUNCTION:
                a_b = bounds[node.children[0]]
                b_b = bounds[node.children[1]]
                L = torch.max(a_b[..., 0], b_b[..., 0])
                U = F.disjunction(a_b[..., 1], b_b[..., 1])
                new_b = torch.stack([L, U], dim=-1)

            elif node.ftype == FormulaType.IMPLICATION:
                a_b = bounds[node.children[0]]
                b_b = bounds[node.children[1]]
                L = torch.clamp(
                    1.0 - a_b[..., 1] + b_b[..., 0], min=0.0, max=1.0
                )
                U = torch.clamp(
                    1.0 - a_b[..., 0] + b_b[..., 1], min=0.0, max=1.0
                )
                new_b = torch.stack([L, U], dim=-1)

            elif node.ftype == FormulaType.NECESSITY:
                child_b = bounds[node.children[0]]
                new_b = F.necessity(
                    child_b, accessibility, tau=tau, top_k=top_k
                )

            elif node.ftype == FormulaType.POSSIBILITY:
                child_b = bounds[node.children[0]]
                new_b = F.possibility(
                    child_b, accessibility, tau=tau, top_k=top_k
                )

            elif node.ftype == FormulaType.UNTIL:
                hold_b = bounds[node.children[0]]
                goal_b = bounds[node.children[1]]
                # `tau` is deliberately not forwarded: F.until ignores it
                # (its backward DP has no smooth aggregation) and passing
                # it now raises a DeprecationWarning.
                new_b = F.until(hold_b, goal_b, accessibility)

            else:
                continue

            # Tighten: only increase L, decrease U
            tightened_L = torch.max(bounds[name][..., 0], new_b[..., 0])
            tightened_U = torch.min(bounds[name][..., 1], new_b[..., 1])
            bounds[name] = torch.stack([tightened_L, tightened_U], dim=-1)

            change = (bounds[name] - old_bounds).abs().max().item()
            max_change = max(max_change, change)

        # --- Downward pass (root → leaves) ---
        for name in reversed(order):
            node = graph.nodes[name]
            parent_b = bounds[name]

            if node.ftype == FormulaType.ATOMIC:
                continue

            elif node.ftype == FormulaType.NEGATION:
                # ¬child = parent → child = ¬parent (swapped)
                child_name = node.children[0]
                neg_parent = F.negation(parent_b).flip(-1)
                old_child = bounds[child_name].clone()
                bounds[child_name] = torch.stack([
                    torch.max(old_child[..., 0], neg_parent[..., 0]),
                    torch.min(old_child[..., 1], neg_parent[..., 1]),
                ], dim=-1)
                change = (bounds[child_name] - old_child).abs().max().item()
                max_change = max(max_change, change)

            elif node.ftype == FormulaType.CONJUNCTION:
                # a ∧ b = parent, with parent = max(0, a + b - 1).
                # Sound Łukasiewicz inverses (and symmetrically for b):
                #   a >= parent            → L_a ← max(L_a, L_parent)
                #   a <= U_parent + 1 - b  → U_a ← min(U_a, U_parent + 1 - L_b)
                # The previous rule U_a ← min(U_a, U_parent) is the L_b = 1
                # special case and over-tightens otherwise (e.g. a = 0.9,
                # b = 0.2 gives U_parent = 0.2, which would exclude a = 0.9).
                a_name, b_name = node.children
                for child_name, sibling_name in [
                    (a_name, b_name), (b_name, a_name)
                ]:
                    old_child = bounds[child_name].clone()
                    sibling_L = bounds[sibling_name][..., 0]
                    new_L = torch.max(old_child[..., 0], parent_b[..., 0])
                    new_U = torch.min(
                        old_child[..., 1],
                        torch.clamp(
                            parent_b[..., 1] + 1.0 - sibling_L, max=1.0
                        ),
                    )
                    bounds[child_name] = torch.stack([new_L, new_U], dim=-1)
                    change = (bounds[child_name] - old_child).abs().max().item()
                    max_change = max(max_change, change)

            elif node.ftype == FormulaType.DISJUNCTION:
                # a ∨ b = parent, with parent = min(1, a + b).
                # Sound Łukasiewicz inverses (and symmetrically for b):
                #   parent >= a always (b >= 0, a <= 1)  → U_a ← min(U_a, U_parent)
                #   a + b >= L_parent                    →
                #       L_a ← max(L_a, L_parent - U_b)
                # The lower rule holds whether or not the clamp is active:
                # min(1, a + b) >= L_parent implies a + b >= L_parent for every
                # L_parent <= 1.
                a_name, b_name = node.children
                for child_name, sibling_name in [
                    (a_name, b_name), (b_name, a_name)
                ]:
                    old_child = bounds[child_name].clone()
                    sibling_U = bounds[sibling_name][..., 1]
                    new_L = torch.max(
                        old_child[..., 0],
                        torch.clamp(parent_b[..., 0] - sibling_U, min=0.0),
                    )
                    new_U = torch.min(old_child[..., 1], parent_b[..., 1])
                    bounds[child_name] = torch.stack([new_L, new_U], dim=-1)
                    change = (bounds[child_name] - old_child).abs().max().item()
                    max_change = max(max_change, change)

            elif node.ftype == FormulaType.IMPLICATION:
                # a → b = parent, with parent = min(1, 1 - a + b).
                # Sound Łukasiewicz inverses, from the two sides of the
                # parent interval:
                #   parent >= L_p holds whether or not the clamp is active
                #   (if it is, 1 - a + b >= 1 >= L_p), so
                #     b >= L_p + a - 1 → L_b ← max(L_b, L_p + L_a - 1)  (MP)
                #     a <= 1 - L_p + b → U_a ← min(U_a, 1 - L_p + U_b)  (MT)
                #   parent <= U_p constrains 1 - a + b only where U_p < 1
                #   (at U_p = 1 the clamp may be active and says nothing):
                #     b <= U_p + a - 1 → U_b ← min(U_b, U_p + U_a - 1)
                #     a >= 1 + b - U_p → L_a ← max(L_a, 1 + L_b - U_p)
                # Up to 0.8.0 only the first rule ran, so `a → b` true with
                # `b` false left `a` at [0, 1] instead of [0, 0].
                a_name, b_name = node.children
                old_a = bounds[a_name].clone()
                old_b = bounds[b_name].clone()
                L_p, U_p = parent_b[..., 0], parent_b[..., 1]
                upper_informative = U_p < 1.0

                new_L_b = torch.clamp(L_p + old_a[..., 0] - 1.0, min=0.0)
                new_U_a = torch.clamp(1.0 - L_p + old_b[..., 1], max=1.0)
                new_U_b = torch.where(
                    upper_informative,
                    torch.clamp(U_p + old_a[..., 1] - 1.0, min=0.0),
                    torch.ones_like(U_p),
                )
                new_L_a = torch.where(
                    upper_informative,
                    torch.clamp(1.0 + old_b[..., 0] - U_p, max=1.0),
                    torch.zeros_like(U_p),
                )
                bounds[a_name] = torch.stack([
                    torch.max(old_a[..., 0], new_L_a),
                    torch.min(old_a[..., 1], new_U_a),
                ], dim=-1)
                bounds[b_name] = torch.stack([
                    torch.max(old_b[..., 0], new_L_b),
                    torch.min(old_b[..., 1], new_U_b),
                ], dim=-1)
                change = max(
                    (bounds[a_name] - old_a).abs().max().item(),
                    (bounds[b_name] - old_b).abs().max().item(),
                )
                max_change = max(max_change, change)

            elif node.ftype == FormulaType.NECESSITY:
                # □ϕ = parent. Only the LOWER bound factorises per world:
                #   L_parent[w] <= min_{w'} [(1 - A[w,w']) + ϕ[w']]
                #               <= (1 - A[w,w']) + ϕ[w']        for every w'
                #   =>  ϕ[w'] >= L_parent[w] - 1 + A[w,w']      for every w
                # A universally quantified lower bound distributes over the
                # neighbourhood, so the constraint is canonical. Inaccessible
                # pairs (A = 0) contribute L_parent - 1 <= 0 and so are inert.
                # The rule is sound for any sound L_parent, including the
                # top-k upward bound (which keeps the true minimum in its
                # selected terms and so never exceeds the full-row minimum),
                # so it always runs over the full accessibility matrix.
                # The upper direction does NOT factorise: U_parent[w] bounds a
                # *minimum*, i.e. it says some neighbour is small without saying
                # which, and many valuation profiles realise the same minimum.
                child_name = node.children[0]
                old_child = bounds[child_name].clone()
                cand = parent_b[..., 0].unsqueeze(1) - 1.0 + accessibility
                new_L = torch.max(
                    old_child[..., 0],
                    torch.clamp(cand.max(dim=0).values, min=0.0, max=1.0),
                )
                bounds[child_name] = torch.stack([
                    new_L, old_child[..., 1]
                ], dim=-1)
                change = (bounds[child_name] - old_child).abs().max().item()
                max_change = max(max_change, change)

            elif node.ftype == FormulaType.POSSIBILITY:
                # ♢ϕ = parent. Dual of the above: only the UPPER bound
                # factorises, because an existential upper bound caps every
                # disjunct.
                #   U_parent[w] >= max_{w'} [A[w,w'] + ϕ[w'] - 1]
                #               >= A[w,w'] + ϕ[w'] - 1          for every w'
                #   =>  ϕ[w'] <= U_parent[w] + 1 - A[w,w']      for every w
                child_name = node.children[0]
                old_child = bounds[child_name].clone()
                cand = parent_b[..., 1].unsqueeze(1) + 1.0 - accessibility
                new_U = torch.min(
                    old_child[..., 1],
                    torch.clamp(cand.min(dim=0).values, min=0.0, max=1.0),
                )
                bounds[child_name] = torch.stack([
                    old_child[..., 0], new_U
                ], dim=-1)
                change = (bounds[child_name] - old_child).abs().max().item()
                max_change = max(max_change, change)

            # UNTIL has no downward rule: its backward DP couples every time
            # step, so neither endpoint factorises into a per-step constraint.

        if max_change < convergence_threshold:
            break
    else:
        if max_change >= convergence_threshold:
            warnings.warn(
                f"upward_downward did not converge in {max_iterations} "
                f"iterations (last max bound change {max_change:.2e} >= "
                f"threshold {convergence_threshold:.2e}); returned bounds are "
                f"sound but may not be the fixed point. Raise max_iterations.",
                RuntimeWarning,
                stacklevel=2,
            )

    return bounds
