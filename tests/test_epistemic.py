"""Tests for torchmodal.epistemic — group knowledge and the frame audit.

Several of these pin *negative* behaviour: a Łukasiewicz fold flooring an
iterated tower, and the common-knowledge fixpoint returning a vacuous interval.
Those are real properties of graded modal logic, not defects, and they decide
which operator may appear in a loss — so they are regression-tested to stop
anyone silently reintroducing them.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from torchmodal import functional as F
from torchmodal.epistemic import (
    and_bounds,
    common_knowledge,
    distributed_knowledge,
    everybody_knows,
    frame_audit,
    mutual_knowledge,
    pooled_accessibility,
)

N_AGENTS = 6


def dense_relation(value: float = 0.9, n: int = N_AGENTS) -> torch.Tensor:
    A = torch.full((n, n), value)
    A.fill_diagonal_(1.0)
    return A


def broadcast(A: torch.Tensor, n: int = N_AGENTS) -> torch.Tensor:
    return A.unsqueeze(0).expand(n, -1, -1).contiguous()


class TestAndBounds:
    def test_crisp_inputs_are_exact_for_every_tnorm(self):
        for bits in itertools.product([0.0, 1.0], repeat=3):
            stack = torch.tensor([[b, b] for b in bits])
            expected = min(bits)
            for tnorm in ("godel", "product", "luk"):
                got = and_bounds(stack, dim=0, tnorm=tnorm)
                assert got[0].item() == pytest.approx(expected)

    def test_soundness_ordering_luk_le_prod_le_godel(self):
        torch.manual_seed(0)
        for _ in range(200):
            stack = torch.rand(5, 2)
            stack[:, 1] = torch.clamp(stack[:, 0] + torch.rand(5) * 0.3, max=1.0)
            luk = and_bounds(stack, tnorm="luk")[0]
            prod = and_bounds(stack, tnorm="product")[0]
            god = and_bounds(stack, tnorm="godel")[0]
            assert luk <= prod + 1e-6 <= god + 1e-6

    def test_godel_is_idempotent_and_luk_is_not(self):
        stack = torch.tensor([[0.8, 1.0], [0.8, 1.0]])
        assert and_bounds(stack, tnorm="godel")[0].item() == pytest.approx(0.8)
        assert and_bounds(stack, tnorm="luk")[0].item() == pytest.approx(0.6)

    def test_rejects_unknown_tnorm(self):
        with pytest.raises(ValueError, match="tnorm"):
            and_bounds(torch.rand(3, 2), tnorm="nope")


class TestEverybodyKnows:
    def test_matches_single_necessity_when_relations_are_identical(self):
        A = dense_relation()
        phi = torch.stack([torch.full((N_AGENTS,), 0.7), torch.ones(N_AGENTS)], -1)
        eg = everybody_knows(phi, broadcast(A), tau=0.1)
        box = F.necessity(phi, A, tau=0.1)
        assert torch.allclose(eg, box, atol=1e-6)

    def test_accepts_a_single_matrix(self):
        A = dense_relation()
        phi = torch.full((N_AGENTS, 2), 0.6)
        assert torch.allclose(
            everybody_knows(phi, A, tau=0.1),
            everybody_knows(phi, broadcast(A), tau=0.1),
        )

    def test_crisp_limit_recovers_classical_conjunction_of_boxes(self):
        A = (torch.rand(N_AGENTS, N_AGENTS) > 0.4).float()
        A.fill_diagonal_(1.0)
        V = (torch.rand(N_AGENTS) > 0.5).float()
        phi = torch.stack([V, V], dim=-1)
        eg = everybody_knows(phi, broadcast(A), tau=1e-4)
        crisp = min(
            min(
                (V[w2].item() for w2 in range(N_AGENTS) if A[w, w2] > 0.5),
                default=1.0,
            )
            for w in range(N_AGENTS)
        )
        assert eg[..., 0].min().item() == pytest.approx(crisp, abs=2e-3)


class TestMutualKnowledgeTower:
    """The tower is the trainable gauge; the fold decides whether it survives."""

    REFERENCE_GODEL = [0.896, 0.791, 0.687, 0.583, 0.478]

    def test_godel_tower_degrades_gracefully_and_keeps_gradient(self):
        A = dense_relation().requires_grad_(True)
        phi = torch.ones(N_AGENTS, 2)
        for depth, expected in enumerate(self.REFERENCE_GODEL, start=1):
            out = mutual_knowledge(phi, broadcast(A), depth=depth, tau=0.1)
            assert out[0, 0].item() == pytest.approx(expected, abs=2e-3)
            grad = torch.autograd.grad(out[0, 0], A, retain_graph=True)[0]
            assert grad.abs().max() > 0, f"gradient vanished at depth {depth}"

    def test_luk_fold_floors_the_tower_at_depth_two(self):
        """Pinned trap: the loosest sound fold destroys the gauge."""
        A = dense_relation().requires_grad_(True)
        phi = torch.ones(N_AGENTS, 2)
        first = mutual_knowledge(phi, broadcast(A), depth=1, tau=0.1, tnorm="luk")
        assert first[0, 0].item() == pytest.approx(0.374, abs=2e-3)
        second = mutual_knowledge(phi, broadcast(A), depth=2, tau=0.1, tnorm="luk")
        assert second[0, 0].item() == 0.0
        grad = torch.autograd.grad(second[0, 0], A, allow_unused=True)[0]
        assert grad is None or grad.abs().max() == 0.0

    def test_depth_one_equals_everybody_knows(self):
        A = dense_relation()
        phi = torch.full((N_AGENTS, 2), 0.8)
        assert torch.allclose(
            mutual_knowledge(phi, broadcast(A), depth=1, tau=0.1),
            everybody_knows(phi, broadcast(A), tau=0.1),
        )

    def test_tower_is_monotone_non_increasing_in_depth(self):
        A = dense_relation()
        phi = torch.ones(N_AGENTS, 2)
        prev = 1.0
        for depth in range(1, 6):
            cur = mutual_knowledge(phi, broadcast(A), depth=depth, tau=0.1)[0, 0].item()
            assert cur <= prev + 1e-6
            prev = cur

    def test_tau_schedule_bounds_the_accumulated_slack(self):
        A = dense_relation()
        phi = torch.ones(N_AGENTS, 2)
        fixed = mutual_knowledge(phi, broadcast(A), depth=5, tau=0.1)[0, 0]
        annealed = mutual_knowledge(
            phi, broadcast(A), depth=5, tau=0.1, tau_schedule=0.5
        )[0, 0]
        assert annealed > fixed

    def test_rejects_zero_depth(self):
        with pytest.raises(ValueError, match="depth"):
            mutual_knowledge(torch.rand(N_AGENTS, 2), dense_relation(), depth=0)


class TestCommonKnowledge:
    def test_unannealed_lower_bound_is_vacuous_and_has_no_gradient(self):
        """Pinned trap: without annealing the gfp can only settle on the floor.

        This is the greatest-fixpoint cliff — iterating down from the top
        through a smooth diamond loses a little each sweep, and on a sub-unit
        relation there is nothing above zero to land on. It is why
        ``tau_decay`` now defaults to 0.5 rather than None; ``None`` is kept
        so the failure stays reproducible, and this test holds it in place.
        """
        for phi_lower in (0.7, 0.9, 0.95, 1.0):
            A = dense_relation().requires_grad_(True)
            phi = torch.stack(
                [torch.full((N_AGENTS,), phi_lower), torch.ones(N_AGENTS)], dim=-1
            )
            out = common_knowledge(
                phi, broadcast(A), tau=0.1, tau_decay=None
            )
            assert out[0, 0].item() == 0.0
            grad = torch.autograd.grad(out[0, 0], A, allow_unused=True)[0]
            assert grad is None or grad.abs().max() == 0.0

    def test_default_lower_bound_is_informative_and_trainable(self):
        """The shipped default must be usable without extra arguments."""
        for phi_lower in (0.7, 0.9, 1.0):
            A = dense_relation().requires_grad_(True)
            phi = torch.stack(
                [torch.full((N_AGENTS,), phi_lower), torch.ones(N_AGENTS)], dim=-1
            )
            out = common_knowledge(phi, broadcast(A), tau=0.1)
            assert out[0, 0].item() > 0.0
            grad = torch.autograd.grad(out[0, 0], A)[0]
            assert grad.abs().max() > 0.0

    def test_tau_decay_makes_the_lower_bound_informative(self):
        A = dense_relation().requires_grad_(True)
        phi = torch.stack(
            [torch.full((N_AGENTS,), 0.9), torch.ones(N_AGENTS)], dim=-1
        )
        out = common_knowledge(phi, broadcast(A), tau=0.1, tau_decay=0.5)
        assert out[0, 0].item() > 0.4
        grad = torch.autograd.grad(out[0, 0], A)[0]
        assert grad.abs().max() > 0

    def test_max_depth_also_escapes_the_floor(self):
        A = dense_relation()
        phi = torch.stack(
            [torch.full((N_AGENTS,), 0.9), torch.ones(N_AGENTS)], dim=-1
        )
        assert common_knowledge(phi, broadcast(A), tau=0.1, max_depth=2)[0, 0] > 0.4

    def test_upper_bound_stays_informative_by_default(self):
        """The upper endpoint keeps discriminating where the lower cannot.

        It is a sound *upper* bound, so it sits above ``U_phi`` rather than on
        it; what matters is that it still moves with the input and stays below
        the vacuous 1.0, which is why it is the right default read-out.
        """
        A = dense_relation()
        seen = []
        for upper in (0.5, 0.7, 0.9):
            phi = torch.stack(
                [torch.zeros(N_AGENTS), torch.full((N_AGENTS,), upper)], dim=-1
            )
            value = common_knowledge(phi, broadcast(A), tau=0.1)[0, 1].item()
            # The annealed default makes the upper endpoint tight enough to
            # land on U_phi to within float error, so compare with tolerance.
            assert value >= upper - 1e-6
            assert value < 1.0
            seen.append(value)
        assert seen == sorted(seen), "upper bound must track U_phi monotonically"


class TestDistributedKnowledge:
    def test_pooled_relation_is_the_elementwise_min(self):
        stack = torch.rand(3, 4, 4)
        assert torch.allclose(pooled_accessibility(stack), stack.min(dim=0).values)

    def test_distributed_dominates_everybody_knows(self):
        torch.manual_seed(1)
        stack = torch.rand(4, 5, 5)
        phi = torch.rand(5, 2)
        phi[:, 1] = torch.clamp(phi[:, 0] + 0.2, max=1.0)
        d_g = distributed_knowledge(phi, stack, tau=0.01)
        e_g = everybody_knows(phi, stack, tau=0.01)
        assert (d_g[:, 0] >= e_g[:, 0] - 1e-5).all()


class TestFrameAudit:
    def test_reports_four_numbers_for_implication_axioms(self):
        report = frame_audit(torch.rand(5, 5), shuffles=5)
        for axiom in ("transitive", "euclidean"):
            entry = report[axiom]
            assert set(entry) == {"score", "coverage", "score_non_vacuous", "null"}
            assert entry["coverage"] is not None
            assert entry["null"] is not None

    def test_zero_coverage_yields_none_not_a_meaningless_one(self):
        """The C-MAPSS failure mode: every weight tiny, so no triple tests 4."""
        A = torch.full((5, 5), 0.14)
        report = frame_audit(A, shuffles=3)
        assert report["transitive"]["coverage"] == 0.0
        assert report["transitive"]["score_non_vacuous"] is None
        assert report["transitive"]["score"] == pytest.approx(1.0)
        assert not report["transitive"].credited

    def test_reflexivity_and_seriality_are_plain_averages(self):
        A = torch.eye(4)
        report = frame_audit(A, shuffles=3)
        assert report["reflexive"]["score"] == pytest.approx(1.0)
        assert report["reflexive"]["null"] is None
        assert report["serial"]["score"] == pytest.approx(1.0)

    def test_symmetric_graph_beats_its_shuffled_null(self):
        torch.manual_seed(0)
        A = (torch.rand(8, 8) > 0.5).float()
        A = ((A + A.T) > 0).float()
        report = frame_audit(A, shuffles=30)
        assert report["symmetric"]["score"] == pytest.approx(1.0)
        assert report["symmetric"]["null"] < 1.0
        assert report["symmetric"].credited
