"""Tests for the dynamic-epistemic-logic update operators.

The central property is the pointwise sandwich ``A_hi <= A_crisp <= A_lo``,
which is what makes :func:`necessity_after` interval-sound. The muddy-children
model is used as the worked case because its crisp answers are known
independently of anything in this library.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from torchmodal.functional import announce, group_announce, necessity_after

WORLDS = [
    torch.tensor(w, dtype=torch.float)
    for w in itertools.product([0, 1], repeat=3)
]
N_WORLDS = len(WORLDS)


def agent_relation(i: int) -> torch.Tensor:
    """Agent *i* cannot tell worlds apart that differ only in its own state."""
    A = torch.zeros(N_WORLDS, N_WORLDS)
    for a, wa in enumerate(WORLDS):
        for b, wb in enumerate(WORLDS):
            if all(wa[k] == wb[k] for k in range(3) if k != i):
                A[a, b] = 1.0
    return A


def bounds_of(values: torch.Tensor) -> torch.Tensor:
    return torch.stack([values, values], dim=-1)


AT_LEAST_ONE = torch.tensor([1.0 if w.sum() >= 1 else 0.0 for w in WORLDS])
CHILD0_MUDDY = torch.tensor([float(w[0]) for w in WORLDS])


def crisp_box(
    A: torch.Tensor, V: torch.Tensor, surviving: torch.Tensor
) -> torch.Tensor:
    """Crisp box over the surviving submodel; a dead end is vacuously true."""
    out = torch.zeros(N_WORLDS)
    for w in range(N_WORLDS):
        reachable = [
            w2 for w2 in range(N_WORLDS) if A[w, w2] > 0.5 and surviving[w2] > 0.5
        ]
        out[w] = 1.0 if not reachable else min(float(V[w2]) for w2 in reachable)
    return out


class TestAnnounceSandwich:
    def test_hi_le_crisp_le_lo_on_random_graded_inputs(self):
        """The invariant every soundness claim downstream rests on."""
        torch.manual_seed(0)
        A = agent_relation(0)
        for _ in range(1000):
            lower = torch.rand(N_WORLDS)
            upper = torch.clamp(lower + torch.rand(N_WORLDS) * 0.5, max=1.0)
            crisp_value = lower + (upper - lower) * torch.rand(N_WORLDS)
            lo, hi = announce(A, torch.stack([lower, upper], dim=-1))
            crisp_relation, _ = announce(A, bounds_of(crisp_value))
            assert (hi <= crisp_relation + 1e-6).all()
            assert (crisp_relation <= lo + 1e-6).all()

    def test_zero_trust_is_the_identity(self):
        A = agent_relation(1)
        lo, hi = announce(A, bounds_of(AT_LEAST_ONE), trust=0.0)
        assert torch.equal(lo, A)
        assert torch.equal(hi, A)

    def test_full_trust_on_crisp_psi_is_pal_relativisation(self):
        A = agent_relation(2)
        lo, hi = announce(A, bounds_of(AT_LEAST_ONE), trust=1.0)
        expected = A * AT_LEAST_ONE.unsqueeze(0)
        assert torch.allclose(lo, expected)
        assert torch.allclose(hi, expected)

    def test_partial_trust_interpolates(self):
        A = agent_relation(0)
        lo_half, _ = announce(A, bounds_of(AT_LEAST_ONE), trust=0.5)
        lo_full, _ = announce(A, bounds_of(AT_LEAST_ONE), trust=1.0)
        assert (lo_half >= lo_full - 1e-6).all()
        assert (lo_half <= A + 1e-6).all()

    def test_rejects_unknown_tnorm(self):
        with pytest.raises(ValueError, match="tnorm"):
            announce(agent_relation(0), bounds_of(AT_LEAST_ONE), tnorm="nope")


class TestNecessityAfterSoundness:
    """P1: the announced box brackets crisp PAL on the surviving worlds."""

    @pytest.mark.parametrize("tau", [0.3, 0.1, 0.03, 0.01])
    def test_brackets_crisp_pal_on_survivors(self, tau):
        A = agent_relation(0)
        crisp = crisp_box(A, CHILD0_MUDDY, AT_LEAST_ONE)
        out = necessity_after(
            bounds_of(CHILD0_MUDDY), A, bounds_of(AT_LEAST_ONE), tau=tau
        )
        survivors = AT_LEAST_ONE > 0.5
        assert (out[survivors, 0] <= crisp[survivors] + 1e-9).all()
        assert (out[survivors, 1] >= crisp[survivors] - 1e-9).all()

    def test_interval_tightens_as_tau_falls(self):
        A = agent_relation(0)
        survivors = AT_LEAST_ONE > 0.5
        widths = []
        for tau in (0.3, 0.1, 0.03, 0.01):
            out = necessity_after(
                bounds_of(CHILD0_MUDDY), A, bounds_of(AT_LEAST_ONE), tau=tau
            )
            widths.append(float((out[survivors, 1] - out[survivors, 0]).max()))
        assert widths == sorted(widths, reverse=True)
        assert widths[-1] < 0.02

    def test_cut_worlds_are_dead_ends_not_deletions(self):
        """Documented divergence from crisp PAL: a cut world reports ~[1, 1]."""
        A = agent_relation(0)
        out = necessity_after(
            bounds_of(CHILD0_MUDDY), A, bounds_of(AT_LEAST_ONE), tau=0.01
        )
        cut = (AT_LEAST_ONE < 0.5).nonzero().flatten()
        assert cut.numel() == 1  # only the all-clean world
        assert out[cut[0], 0] > 0.95
        assert out[cut[0], 1] == pytest.approx(1.0, abs=1e-3)

    def test_gradient_reaches_the_relation(self):
        A = agent_relation(0).requires_grad_(True)
        out = necessity_after(
            bounds_of(CHILD0_MUDDY), A, bounds_of(AT_LEAST_ONE), tau=0.1
        )
        grad = torch.autograd.grad(out[..., 0].sum(), A)[0]
        assert grad.abs().max() > 0


class TestGroupAnnounce:
    def test_non_recipients_keep_their_relation(self):
        stack = torch.stack([agent_relation(i) for i in range(3)])
        recipients = torch.tensor([1.0, 0.0, 1.0])
        lo, hi = group_announce(stack, bounds_of(AT_LEAST_ONE), recipients)
        assert torch.allclose(lo[1], stack[1])
        assert torch.allclose(hi[1], stack[1])

    def test_recipients_are_updated(self):
        stack = torch.stack([agent_relation(i) for i in range(3)])
        recipients = torch.tensor([1.0, 0.0, 1.0])
        lo, _ = group_announce(stack, bounds_of(AT_LEAST_ONE), recipients)
        expected = stack[0] * AT_LEAST_ONE.unsqueeze(0)
        assert torch.allclose(lo[0], expected)

    def test_requires_a_stack_of_relations(self):
        with pytest.raises(ValueError, match="group_announce"):
            group_announce(
                agent_relation(0), bounds_of(AT_LEAST_ONE), torch.ones(3)
            )
