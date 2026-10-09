"""Muddy children — a classic epistemic puzzle, recovered exactly.

Three children are playing; all three get mud on their foreheads. Each can
see the others but not themselves. Their father announces:

    "At least one of you is muddy."

He then asks, repeatedly, "does anyone know whether they are muddy?" The
children answer truthfully and simultaneously. Nobody knows after the
first question, nor after the second — but after the third, *all three*
know they are muddy.

The puzzle is a standard test for epistemic logic because the information
that moves the children is entirely negative: nothing new is observed,
only the fact that nobody else could answer. This script encodes it as a
Kripke model and evaluates ``K_a(muddy_a)`` with torchmodal's necessity
neuron, printing the learned interval bounds beside the crisp answer.

The bounds must *bracket* the crisp value at every round — that is the
soundness property, and the script asserts it. As ``tau -> 0`` the
interval collapses onto the crisp answer.

Run::

    python examples/muddy_children.py
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import torch

# Always use the checkout, never a previously installed PyPI release.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torchmodal import functional as F  # noqa: E402

N_CHILDREN = 3
TAU = 0.05

# The 8 possible worlds: one bit per child, 1 = muddy.
WORLDS = list(product([0, 1], repeat=N_CHILDREN))
ACTUAL = (1, 1, 1)  # every child really is muddy


def indistinguishable(agent: int) -> torch.Tensor:
    """Agent `agent` sees everyone but itself.

    Two worlds are indistinguishable to the agent when they agree on every
    *other* child, so the relation is an equivalence — the standard S5
    frame for knowledge.
    """
    n = len(WORLDS)
    A = torch.zeros(n, n)
    for i, w in enumerate(WORLDS):
        for j, v in enumerate(WORLDS):
            if all(w[k] == v[k] for k in range(N_CHILDREN) if k != agent):
                A[i, j] = 1.0
    return A


def surviving(round_idx: int) -> torch.Tensor:
    """Worlds still considered possible after `round_idx` announcements.

    Round 0 is the father's "at least one is muddy", which kills the
    all-clean world. Each later round is the public fact that nobody knew
    yet, which kills every world with exactly that many muddy children.
    """
    keep = torch.ones(len(WORLDS))
    for i, w in enumerate(WORLDS):
        if sum(w) < round_idx + 1:
            keep[i] = 0.0
    return keep


def knows_own_state(agent: int, round_idx: int) -> torch.Tensor:
    """Bounds for K_a(muddy_a) at every world, after `round_idx` rounds.

    An announcement removes worlds, which is applied here by zeroing the
    accessibility *into* the removed worlds — the agent stops considering
    them possible.
    """
    keep = surviving(round_idx)
    A = indistinguishable(agent) * keep.unsqueeze(0)

    muddy = torch.tensor(
        [float(w[agent]) for w in WORLDS]
    ).unsqueeze(-1).expand(-1, 2)

    return F.necessity(muddy, A, tau=TAU)


def crisp_knows(agent: int, round_idx: int) -> int:
    """The textbook answer: 1 if the agent knows, computed exactly."""
    keep = surviving(round_idx)
    considered = [
        v
        for j, v in enumerate(WORLDS)
        if keep[j] > 0
        and all(ACTUAL[k] == v[k] for k in range(N_CHILDREN) if k != agent)
    ]
    return int(all(v[agent] == 1 for v in considered)) if considered else 1


def main() -> None:
    here = WORLDS.index(ACTUAL)
    print("Muddy children — all 3 are muddy; each sees the others only.\n")
    print(f"{'after round':>12}  {'child':>5}  {'crisp':>5}   learned bounds")
    print("-" * 56)

    for round_idx in range(N_CHILDREN):
        for agent in range(N_CHILDREN):
            bounds = knows_own_state(agent, round_idx)[here]
            crisp = crisp_knows(agent, round_idx)
            lo, hi = bounds[0].item(), bounds[1].item()

            # Soundness: the interval must contain the crisp value.
            assert lo - 1e-4 <= crisp <= hi + 1e-4, (
                f"unsound at round {round_idx}, child {agent}: "
                f"[{lo:.4f}, {hi:.4f}] does not contain {crisp}"
            )

            print(
                f"{round_idx + 1:>12}  {agent:>5}  {crisp:>5}   "
                f"[{lo:.4f}, {hi:.4f}]"
            )
        print()

    print("Every interval bracketed the crisp answer (asserted above).")
    print(
        "Nobody knows after rounds 1 and 2; after round 3 all three do — "
        "the textbook result,\nrecovered from the modal operators alone."
    )


if __name__ == "__main__":
    main()
