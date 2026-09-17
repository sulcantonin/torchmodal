"""Tests for modal transition systems and torchmodal.verify.

The load-bearing test is :meth:`TestModalTransitionSystems.
test_interval_brackets_every_concretisation`: an MTS interval is only
meaningful if it encloses the value on *every* Kripke frame lying between the
must and may relations, and that is checked by sampling concretisations rather
than by construction.
"""

import re

import pytest
import torch

from torchmodal import fixpoint as fp
from torchmodal import functional as F
from torchmodal import verify as V


class TestModalTransitionSystems:
    @pytest.mark.parametrize(
        "pair", [(F.necessity_mts, F.necessity), (F.possibility_mts, F.possibility)]
    )
    def test_must_equals_may_reduces_exactly(self, pair):
        mts, plain = pair
        torch.manual_seed(0)
        for _ in range(20):
            A = torch.rand(6, 6)
            bounds = torch.rand(6, 2).sort(dim=1).values
            assert torch.equal(mts(bounds, A, A), plain(bounds, A))

    @pytest.mark.parametrize("mode", ["soft", "exact"])
    def test_must_equals_may_reduces_in_both_modes(self, mode):
        A = torch.rand(5, 5)
        bounds = torch.rand(5, 2).sort(dim=1).values
        assert torch.equal(
            F.necessity_mts(bounds, A, A, mode=mode),
            F.necessity(bounds, A, mode=mode),
        )

    def test_interval_brackets_every_concretisation(self):
        """The property that makes an MTS interval mean anything."""
        torch.manual_seed(0)
        bounds = torch.rand(5, 2).sort(dim=1).values
        must = torch.rand(5, 5) * 0.3
        may = must + torch.rand(5, 5) * 0.7

        for op_mts, op in (
            (F.necessity_mts, F.necessity),
            (F.possibility_mts, F.possibility),
        ):
            envelope = op_mts(bounds, must, may, mode="exact")
            for _ in range(100):
                lam = torch.rand(5, 5)
                concrete = must + lam * (may - must)
                got = op(bounds, concrete, mode="exact")
                assert (got[:, 0] >= envelope[:, 0] - 1e-6).all()
                assert (got[:, 1] <= envelope[:, 1] + 1e-6).all()

    def test_universal_and_existential_use_opposite_relations(self):
        """Box widens with may; diamond widens with must. They are duals."""
        bounds = torch.full((4, 2), 0.5)
        must = torch.zeros(4, 4)
        may = torch.ones(4, 4)
        box = F.necessity_mts(bounds, must, may, mode="exact")
        dia = F.possibility_mts(bounds, must, may, mode="exact")
        # both are genuinely uncertain here, so both intervals are wide
        assert (box[:, 1] - box[:, 0]).min() > 0
        assert (dia[:, 1] - dia[:, 0]).min() > 0

    def test_rejects_an_ill_formed_system(self):
        bounds = torch.rand(4, 2).sort(dim=1).values
        with pytest.raises(ValueError, match="ill-formed"):
            F.necessity_mts(bounds, torch.ones(4, 4), torch.zeros(4, 4))

    def test_rejects_mismatched_shapes(self):
        bounds = torch.rand(4, 2).sort(dim=1).values
        with pytest.raises(ValueError, match="same shape"):
            F.necessity_mts(bounds, torch.ones(4, 4), torch.ones(5, 5))


class TestCertify:
    def test_the_three_verdicts(self):
        bounds = torch.tensor([[0.9, 1.0], [0.0, 0.1], [0.2, 0.8]])
        assert V.certify(bounds) == [
            V.Verdict.PROVEN, V.Verdict.REFUTED, V.Verdict.UNDECIDED
        ]

    def test_undecided_is_returned_rather_than_rounded(self):
        """An interval straddling the boundary must not be forced."""
        assert V.certify(torch.tensor([[0.49, 0.51]])) == [
            V.Verdict.UNDECIDED
        ]

    def test_threshold_is_respected(self):
        bounds = torch.tensor([[0.7, 0.8]])
        assert V.certify(bounds, threshold=0.6) == [V.Verdict.PROVEN]
        assert V.certify(bounds, threshold=0.9) == [V.Verdict.REFUTED]


class TestRoundingMargin:
    def test_boundary_is_zero_and_extremes_are_half(self):
        m = V.rounding_margin(
            torch.tensor([[0.5, 0.5], [0.0, 0.0], [1.0, 1.0]])
        )
        assert m.tolist() == [0.0, 0.5, 0.5]

    def test_is_never_negative(self):
        torch.manual_seed(0)
        b = torch.rand(20, 2).sort(dim=1).values
        assert (V.rounding_margin(b) >= 0).all()


class TestCertificateGap:
    def test_identical_inputs_give_zero(self):
        torch.manual_seed(0)
        b = torch.rand(8, 2).sort(dim=1).values
        assert V.certificate_gap(b, b) == 0.0

    def test_fully_opposed_gives_one(self):
        lo = torch.zeros(4, 2)
        hi = torch.ones(4, 2)
        assert V.certificate_gap(lo, hi) == 1.0

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            V.certificate_gap(torch.zeros(3, 2), torch.zeros(4, 2))

    def test_soft_agrees_with_exact_on_a_clean_frame(self):
        A = torch.zeros(5, 5)
        A[torch.arange(4), torch.arange(1, 5)] = 1.0
        phi = torch.tensor([0.0, 0, 0, 0, 1.0])
        soft = fp.ef(phi, A, tau=0.1).bounds
        exact = fp.ef(phi, A, mode="exact").bounds
        assert V.certificate_gap(soft, exact) == 0.0


class TestWitnessPath:
    def test_finds_a_shortest_path(self):
        A = torch.zeros(5, 5)
        A[torch.arange(4), torch.arange(1, 5)] = 1.0
        goal = torch.tensor([0.0, 0, 0, 0, 1.0])
        assert V.witness_path(A, 0, goal) == [0, 1, 2, 3, 4]

    def test_start_already_satisfies_the_goal(self):
        A = torch.zeros(3, 3)
        assert V.witness_path(A, 1, torch.tensor([0.0, 1.0, 0.0])) == [1]

    def test_returns_none_when_unreachable(self):
        A = torch.zeros(4, 4)
        A[0, 1] = A[1, 0] = 1.0
        goal = torch.tensor([0.0, 0.0, 0.0, 1.0])
        assert V.witness_path(A, 0, goal) is None

    def test_accepts_bounds_shaped_goal(self):
        A = torch.zeros(3, 3)
        A[0, 1] = A[1, 2] = 1.0
        goal = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        assert V.witness_path(A, 0, goal) == [0, 1, 2]


class TestRoundAndCertify:
    def test_certifies_reachability_on_a_learned_relation(self):
        torch.manual_seed(0)
        A = torch.rand(5, 5) * 0.4
        A[0, 1], A[1, 2], A[2, 3] = 0.95, 0.92, 0.88
        phi = torch.tensor([0.0, 0, 0, 1.0, 0])
        r = V.round_and_certify(phi, A, operator="ef")
        assert r.verdicts[0] == V.Verdict.PROVEN
        assert V.witness_path(r.relation, 0, phi) is not None

    def test_reports_the_frame_it_certified(self):
        A = torch.rand(4, 4)
        phi = torch.rand(4)
        r = V.round_and_certify(phi, A, operator="ef")
        assert set(r.relation.unique().tolist()) <= {0.0, 1.0}
        assert r.n_flipped >= 0

    def test_margin_is_reported_per_world(self):
        A = torch.rand(4, 4)
        phi = torch.rand(4)
        r = V.round_and_certify(phi, A, operator="ef")
        assert r.margin.shape == (4,)

    def test_rejects_an_unknown_operator(self):
        with pytest.raises(ValueError, match="operator"):
            V.round_and_certify(
                torch.rand(3), torch.rand(3, 3), operator="until"
            )


class TestSmvExport:
    def _frame(self):
        A = torch.zeros(3, 3)
        A[0, 1] = A[1, 2] = A[2, 0] = 1.0
        return A

    def test_structure(self):
        smv = V.to_smv(self._frame(), {"p": [1.0, 0.0, 0.0]})
        assert "MODULE main" in smv
        assert "state : 0 .. 2;" in smv
        assert "init(state) := 0;" in smv
        assert "esac;" in smv

    def test_every_world_has_a_transition_case(self):
        smv = V.to_smv(self._frame(), {})
        for i in range(3):
            assert re.search(rf"state = {i} : \{{", smv)

    def test_labels_become_defines(self):
        smv = V.to_smv(self._frame(), {"goal": [0.0, 0.0, 1.0]})
        assert "DEFINE" in smv
        assert "goal := state = 2;" in smv

    def test_an_empty_label_is_false_not_malformed(self):
        smv = V.to_smv(self._frame(), {"never": [0.0, 0.0, 0.0]})
        assert "never := FALSE;" in smv

    def test_spec_is_emitted(self):
        smv = V.to_smv(self._frame(), {"p": [1.0, 0, 0]}, spec="AG (p)")
        assert "CTLSPEC AG (p);" in smv

    def test_dead_end_is_rejected_with_a_pointer_to_serialize(self):
        A = torch.zeros(3, 3)
        A[0, 1] = 1.0
        with pytest.raises(ValueError, match="serialize"):
            V.to_smv(A, {})

    def test_serialize_makes_it_exportable(self):
        A = torch.zeros(3, 3)
        A[0, 1] = 1.0
        smv = V.to_smv(F.serialize(A), {})
        assert "MODULE main" in smv

    def test_rejects_a_label_of_the_wrong_length(self):
        with pytest.raises(ValueError, match="worlds"):
            V.to_smv(self._frame(), {"p": [1.0, 0.0]})

    def test_module_name_is_configurable(self):
        assert "MODULE frame" in V.to_smv(
            self._frame(), {}, module_name="frame"
        )


class TestRoundRelation:
    def test_produces_only_zero_and_one(self):
        torch.manual_seed(0)
        out = V.round_relation(torch.rand(6, 6))
        assert set(out.unique().tolist()) <= {0.0, 1.0}

    def test_threshold_is_respected(self):
        A = torch.tensor([[0.4, 0.6]])
        assert V.round_relation(A).tolist() == [[0.0, 1.0]]
        assert V.round_relation(A, threshold=0.3).tolist() == [[1.0, 1.0]]

    def test_preserves_dtype(self):
        A = torch.rand(4, 4, dtype=torch.float64)
        assert V.round_relation(A).dtype == torch.float64
