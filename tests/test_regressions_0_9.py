"""Regression tests for the 0.9.0 audit fixes.

Each test pins a behaviour that was wrong up to 0.8.0 and names the symptom
it guards against, so that a future change which reintroduces the defect
fails here with a message that says what broke.
"""

from __future__ import annotations

import itertools
import warnings

import pytest
import torch

import torchmodal
from torchmodal import FormulaGraph, upward_downward, utils, verify
from torchmodal import fixpoint as fp
from torchmodal import functional as F
from torchmodal import nn as mnn
from torchmodal.epistemic import common_knowledge, frame_audit, mutual_knowledge
from torchmodal.epistemic.operators import _tau_at
from torchmodal.kripke import KripkeModel, Proposition
from torchmodal.losses import AxiomRegularization, ContradictionLoss, SemanticLoss

# ---------------------------------------------------------------------------
# 1. learnable_tau received no gradient (tau.item() detached it)
# ---------------------------------------------------------------------------


class TestLearnableTau:
    @pytest.mark.parametrize("cls", [mnn.Necessity, mnn.Possibility])
    def test_modal_modules_pass_gradient_to_tau(self, cls):
        torch.manual_seed(0)
        m = cls(tau=0.1, learnable_tau=True)
        A = torch.rand(4, 4)
        b = torch.rand(4, 2).sort(-1).values
        out = m(b, A)
        assert out.requires_grad, "output must be connected to tau"
        out.sum().backward()
        assert m.tau.grad is not None and torch.isfinite(m.tau.grad)
        assert m.tau.grad.abs() > 0

    @pytest.mark.parametrize("cls", [mnn.SmoothMin, mnn.SmoothMax, mnn.ConvPool])
    def test_aggregator_modules_pass_gradient_to_tau(self, cls):
        torch.manual_seed(0)
        m = cls(tau=0.1, learnable=True)
        m(torch.rand(5)).sum().backward()
        assert m.tau.grad is not None and m.tau.grad.abs() > 0

    @pytest.mark.parametrize("cls", [mnn.Necessity, mnn.Possibility])
    def test_set_tau_works_on_the_parameter(self, cls):
        m = cls(tau=0.1, learnable_tau=True)
        m.set_tau(0.05)  # in-place on a leaf requiring grad: needs no_grad
        assert m.tau.item() == pytest.approx(0.05)

    def test_tensor_tau_matches_float_tau(self):
        torch.manual_seed(0)
        A = torch.rand(6, 6)
        b = torch.rand(6, 2).sort(-1).values
        assert torch.equal(
            F.necessity(b, A, tau=0.1), F.necessity(b, A, tau=torch.tensor(0.1))
        )
        assert torch.equal(
            F.possibility(b, A, tau=0.1),
            F.possibility(b, A, tau=torch.tensor(0.1)),
        )


# ---------------------------------------------------------------------------
# 2. group_announce crashed with per-world trust
# ---------------------------------------------------------------------------


class TestGroupAnnounceTrustShapes:
    @pytest.mark.parametrize("trust_shape", [(), (3,), (3, 4)])
    def test_accepts_every_documented_trust_shape(self, trust_shape):
        torch.manual_seed(0)
        Ag = torch.rand(3, 4, 4)
        psi = torch.rand(4, 2).sort(-1).values
        trust = torch.rand(trust_shape) if trust_shape else 0.7
        recipients = torch.tensor([1.0, 0.0, 1.0])
        lo, hi = F.group_announce(Ag, psi, recipients=recipients, trust=trust)
        assert lo.shape == hi.shape == (3, 4, 4)
        # The non-recipient keeps its relation untouched.
        assert torch.equal(lo[1], Ag[1]) and torch.equal(hi[1], Ag[1])
        # Recipients are cut: hi <= lo <= A.
        assert bool((hi <= lo + 1e-6).all()) and bool((lo <= Ag + 1e-6).all())


# ---------------------------------------------------------------------------
# 3. Transitivity regulariser used the matrix product
# ---------------------------------------------------------------------------


class TestTransitivityIsMaxMinComposition:
    def test_godel_transitive_graded_relation_is_not_penalised(self):
        # max_j min(A[i,j], A[j,k]) <= A[i,k] holds, yet the matrix product
        # gave 0.0625 here because (A @ A)[0, 1] = 1.0 + 0.5 > 0.5.
        A = torch.tensor([[1.0, 0.5], [0.0, 1.0]])
        assert AxiomRegularization(transitivity=1.0)(A).item() == 0.0

    def test_reflexive_relation_with_one_graded_edge_is_transitive(self):
        A = torch.eye(3)
        A[0, 1] = 0.3
        assert AxiomRegularization(transitivity=1.0)(A).item() == 0.0

    def test_crisp_transitive_relation_scores_zero(self):
        A = torch.tensor([[1.0, 1, 1], [0, 1, 1], [0, 0, 1]])
        assert AxiomRegularization(transitivity=1.0)(A).item() == 0.0

    def test_crisp_non_transitive_relation_is_penalised(self):
        A = torch.tensor([[1.0, 1, 0], [0, 1, 1], [0, 0, 1]])  # 0->1->2, no 0->2
        assert AxiomRegularization(transitivity=1.0)(A).item() > 0.0

    def test_matches_explicit_max_min_definition(self):
        torch.manual_seed(1)
        A = torch.rand(5, 5)
        comp = torch.minimum(A.unsqueeze(2), A.unsqueeze(0)).max(dim=1).values
        expected = torch.relu(comp - A).pow(2).mean()
        got = AxiomRegularization(transitivity=1.0)(A)
        assert torch.allclose(got, expected)


# ---------------------------------------------------------------------------
# 4. KripkeModel.contradiction_loss() was identically zero
# ---------------------------------------------------------------------------


class TestKripkeContradictionLoss:
    def _model(self):
        m = KripkeModel(2, mnn.FixedAccessibility(torch.eye(2)))
        q = m.add_proposition("q", learnable=False)
        q.set_bounds(torch.tensor([[0.0, 0.2], [0.0, 0.2]]))
        return m

    def test_no_argument_form_warns_when_everything_is_learnable(self):
        m = KripkeModel(3, mnn.LearnableAccessibility(3))
        m.add_proposition("p")
        with pytest.warns(UserWarning, match="learnable"):
            assert m.contradiction_loss().item() == 0.0

    def test_no_argument_form_is_silent_with_a_fixed_proposition(self):
        m = self._model()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert m.contradiction_loss().item() == 0.0

    def test_derived_bounds_produce_the_crossing(self):
        m = self._model()
        derived = {"q": torch.tensor([[0.9, 1.0], [0.9, 1.0]])}
        # intersection is [0.9, 0.2] per world -> 0.7 each
        assert m.contradiction_loss(derived).item() == pytest.approx(1.4)

    def test_derived_form_is_differentiable_in_the_relation(self):
        m = KripkeModel(3, mnn.LearnableAccessibility(3, reflexive=False))
        p = m.add_proposition("p", learnable=False)
        p.set_bounds(torch.tensor([[0.0, 0.1], [0.9, 1.0], [0.9, 1.0]]))
        A = m.get_accessibility()
        # □p at world 0 bounds p at world 0 (reflexive reading); p is low
        # there while its neighbours are high, so the crossing depends on A.
        loss = m.contradiction_loss({"p": m.necessity("p", A)})
        assert loss.requires_grad
        loss.backward()
        assert m.accessibility.logits.grad is not None

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            self._model().contradiction_loss({"nope": torch.zeros(2, 2)})


# ---------------------------------------------------------------------------
# 5. Downward implication did modus ponens only, and mutated the input dict
# ---------------------------------------------------------------------------


class TestImplicationDownward:
    def _graph(self):
        g = FormulaGraph()
        g.add_atomic("a")
        g.add_atomic("b")
        g.add_implication("i", "a", "b")
        return g

    def test_modus_tollens(self):
        bounds = {
            "a": torch.tensor([[0.0, 1.0]]),
            "b": torch.tensor([[0.0, 0.0]]),
            "i": torch.tensor([[1.0, 1.0]]),
        }
        out = upward_downward(self._graph(), bounds, torch.eye(1))
        assert out["a"].tolist() == [[0.0, 0.0]]

    def test_upper_side_rules_apply_only_below_one(self):
        # a -> b <= 0.3 with a <= 1: b <= 0.3 + a - 1 <= 0.3, a >= 1 + b - 0.3 >= 0.7
        bounds = {
            "a": torch.tensor([[0.0, 1.0]]),
            "b": torch.tensor([[0.0, 1.0]]),
            "i": torch.tensor([[0.0, 0.3]]),
        }
        out = upward_downward(self._graph(), bounds, torch.eye(1))
        assert out["b"][0, 1].item() == pytest.approx(0.3)
        assert out["a"][0, 0].item() == pytest.approx(0.7)
        # With U_parent = 1 the clamp may be active: nothing is said.
        bounds["i"] = torch.tensor([[0.0, 1.0]])
        out = upward_downward(self._graph(), bounds, torch.eye(1))
        assert out["a"].tolist() == [[0.0, 1.0]]
        assert out["b"].tolist() == [[0.0, 1.0]]

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")  # one sweep only
    def test_inverse_rules_are_sound_on_a_grid(self):
        """Every (a, b) consistent with the parent interval survives."""
        torch.manual_seed(0)
        grid = torch.linspace(0, 1, 11).tolist()
        for _ in range(150):
            La, Ua = sorted(torch.rand(2).tolist())
            Lb, Ub = sorted(torch.rand(2).tolist())
            Lp, Up = sorted(torch.rand(2).tolist())
            if torch.rand(1).item() < 0.3:
                Up = 1.0
            bounds = {
                "a": torch.tensor([[La, Ua]]),
                "b": torch.tensor([[Lb, Ub]]),
                "i": torch.tensor([[Lp, Up]]),
            }
            out = upward_downward(
                self._graph(), bounds, torch.eye(1), max_iterations=1
            )
            la, ua = out["a"][0].tolist()
            lb, ub = out["b"][0].tolist()
            for a, b in itertools.product(grid, grid):
                if not (La <= a <= Ua and Lb <= b <= Ub):
                    continue
                if not (Lp - 1e-6 <= min(1.0, 1 - a + b) <= Up + 1e-6):
                    continue
                assert la - 1e-5 <= a <= ua + 1e-5, (bounds, out)
                assert lb - 1e-5 <= b <= ub + 1e-5, (bounds, out)

    def test_input_dict_is_not_mutated(self):
        bounds = {
            "a": torch.tensor([[0.0, 1.0]]),
            "b": torch.tensor([[1.0, 1.0]]),
            "i": torch.tensor([[0.0, 1.0]]),
        }
        snapshot = {k: v.clone() for k, v in bounds.items()}
        out = upward_downward(self._graph(), bounds, torch.eye(1))
        assert out is not bounds
        assert out["i"].tolist() == [[1.0, 1.0]]
        for k in bounds:
            assert torch.equal(bounds[k], snapshot[k]), k


# ---------------------------------------------------------------------------
# 6. AttentionAccessibility returned row-softmax weights
# ---------------------------------------------------------------------------


class TestAttentionAccessibility:
    def test_rows_are_not_a_distribution(self):
        torch.manual_seed(0)
        access = mnn.AttentionAccessibility(input_dim=16, reflexive=False)
        A = access(torch.randn(50, 16))
        assert A.shape == (50, 50)
        assert not torch.allclose(A.sum(-1), torch.ones(50), atol=1e-3)
        # Up to 0.8.0 the largest entry at |W| = 50 was 0.07.
        assert A.max().item() > 0.3

    def test_init_bias_moves_the_whole_relation(self):
        torch.manual_seed(0)
        x = torch.randn(6, 8)
        A_lo = mnn.AttentionAccessibility(8, num_heads=2, init_bias=-4.0)(x)
        A_hi = mnn.AttentionAccessibility(8, num_heads=2, init_bias=4.0)(x)
        off = ~torch.eye(6, dtype=torch.bool)
        assert A_lo[off].max() < 0.5 < A_hi[off].min()

    def test_asymmetric_and_trainable(self):
        torch.manual_seed(0)
        access = mnn.AttentionAccessibility(input_dim=16, num_heads=4)
        A = access(torch.randn(5, 16))
        off = ~torch.eye(5, dtype=torch.bool)
        assert (A - A.t())[off].abs().max() > 0
        A.sum().backward()
        assert access.q_proj.weight.grad is not None
        assert access.k_proj.weight.grad is not None

    def test_rejects_indivisible_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            mnn.AttentionAccessibility(input_dim=10, num_heads=4)


# ---------------------------------------------------------------------------
# 7. Annealed temperatures underflowed to NaN
# ---------------------------------------------------------------------------


class TestTemperatureFloor:
    def test_tau_at_is_floored(self):
        assert _tau_at(0.5, 0.1, 500) == F.TAU_MIN
        assert _tau_at(lambda k: 0.0, 0.1, 3) == F.TAU_MIN
        assert _tau_at(None, 0.1, 500) == 0.1

    def test_common_knowledge_deep_tower_is_finite(self):
        torch.manual_seed(0)
        Ag = torch.rand(2, 6, 6)
        phi = torch.rand(6, 2).sort(-1).values
        out = common_knowledge(phi, Ag, max_depth=200)  # NaN in 0.8.0
        assert torch.isfinite(out).all()
        out = mutual_knowledge(phi, Ag, depth=200, tau_schedule=0.5)
        assert torch.isfinite(out).all()

    def test_gfp_long_run_is_finite(self):
        torch.manual_seed(0)
        A = torch.rand(6, 6)
        phi = torch.rand(6, 2).sort(-1).values
        r = fp.eg(phi, A, round_stable=False, max_iter=400, tol=0.0)
        assert torch.isfinite(r.bounds).all()

    def test_until_graph_long_run_is_finite(self):
        torch.manual_seed(0)
        T = 6
        A = torch.zeros(T, T)
        A[torch.arange(T - 1), torch.arange(1, T)] = 1.0
        phi = torch.rand(T, 2).sort(-1).values
        psi = torch.rand(T, 2).sort(-1).values
        out = F.until_graph(phi, psi, A, max_iter=400, tol=0.0)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 8. Proposition(init=0 or 1) had infinite logits and no gradient
# ---------------------------------------------------------------------------


class TestPropositionInitClamp:
    @pytest.mark.parametrize("init", [0.0, 1.0])
    def test_extreme_init_still_trains(self, init):
        p = Proposition("p", 3, learnable=True, init=init)
        assert torch.isfinite(p._logits).all()
        assert torch.allclose(p.bounds, torch.full((3, 2), init), atol=1e-3)
        (p.bounds - 0.5).pow(2).sum().backward()
        assert bool((p._logits.grad != 0).all())

    def test_interior_init_is_unchanged(self):
        p = Proposition("p", 2, learnable=True, init=0.3)
        assert torch.allclose(p.bounds, torch.full((2, 2), 0.3), atol=1e-6)


# ---------------------------------------------------------------------------
# 9. bounds_to_labels read the midpoint and called "impossible" indeterminate
# ---------------------------------------------------------------------------


class TestBoundsToLabels:
    def test_reads_the_endpoints(self):
        b = torch.tensor([
            [0.95, 1.00],  # necessary
            [0.30, 0.90],  # wide: possible, not necessary -> indeterminate
            [0.00, 0.05],  # impossible
            [0.90, 1.00],  # vacuous-ish but L == threshold: not necessary
        ])
        nec, pos, ind = utils.bounds_to_labels(b)
        assert nec.tolist() == [True, False, False, False]
        assert pos.tolist() == [True, True, False, True]
        assert ind.tolist() == [False, True, False, True]
        assert bool((nec <= pos).all()), "necessary implies possible"
        assert bool((ind == (pos & ~nec)).all())

    def test_single_bound(self):
        nec, pos, ind = utils.bounds_to_labels(torch.tensor([0.95, 1.0]))
        assert nec.shape == (1,) and nec.item() is True


# ---------------------------------------------------------------------------
# 10. anneal_temperature clamped heating schedules to tau_end
# ---------------------------------------------------------------------------


class TestAnnealHeating:
    def test_heating_schedule_rises(self):
        taus = [
            utils.anneal_temperature(e, 5, tau_start=0.1, tau_end=2.0)
            for e in range(5)
        ]
        assert taus[0] == pytest.approx(0.1) and taus[-1] == pytest.approx(2.0)
        assert all(a < b for a, b in zip(taus, taus[1:]))

    def test_cooling_schedule_unchanged(self):
        taus = [
            utils.anneal_temperature(e, 5, tau_start=2.0, tau_end=0.1)
            for e in range(5)
        ]
        assert taus[0] == pytest.approx(2.0) and taus[-1] == pytest.approx(0.1)
        assert all(a > b for a, b in zip(taus, taus[1:]))


# ---------------------------------------------------------------------------
# 11. Connective modules misread 2-world point tensors as bounds
# ---------------------------------------------------------------------------


class TestConnectiveBoundsFlag:
    def test_explicit_point_values_with_two_worlds(self):
        x = torch.tensor([0.2, 0.9])
        assert mnn.Negation(bounds=False)(x).tolist() == pytest.approx([0.8, 0.1])
        # Default inference still reads a trailing 2 as a bound pair.
        assert mnn.Negation()(x).tolist() == pytest.approx([0.1, 0.8])

    def test_explicit_bounds(self):
        a = torch.tensor([[0.2, 0.9]])
        b = torch.tensor([[0.5, 0.6]])
        out = mnn.Conjunction(bounds=True)(a, b)
        assert torch.allclose(out, torch.tensor([[0.0, 0.6]]))
        out = mnn.Implication(bounds=False)(torch.tensor(0.9), torch.tensor(0.2))
        assert out.item() == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# 12. frame_audit and AxiomRegularization disagreed on seriality
# ---------------------------------------------------------------------------


class TestSerialityConsistency:
    def test_identity_is_not_serial_in_either_place(self):
        eye = torch.eye(3)
        assert frame_audit(eye, shuffles=1)["serial"]["score"] == 0.0
        assert AxiomRegularization(seriality=1.0)(eye).item() > 0.0

    def test_ring_is_serial_in_both(self):
        ring = torchmodal.build_ring_accessibility(4)
        assert frame_audit(ring, shuffles=1)["serial"]["score"] == 1.0
        assert AxiomRegularization(seriality=1.0)(ring).item() == 0.0


# ---------------------------------------------------------------------------
# 13. MultiAgentKripke ignored `features`
# ---------------------------------------------------------------------------


class TestMultiAgentKripkeFeatures:
    def test_features_reach_a_metric_epistemic_relation(self):
        torch.manual_seed(0)
        mk = torchmodal.MultiAgentKripke(
            num_agents=4,
            num_steps=2,
            epistemic_accessibility=mnn.MetricAccessibility(
                4, embed_dim=8, input_dim=5
            ),
        )
        f1, f2 = torch.randn(4, 5), torch.randn(4, 5)
        A1 = mk.get_epistemic_accessibility(f1)
        A2 = mk.get_epistemic_accessibility(f2)
        assert A1.shape == (4, 4) and not torch.allclose(A1, A2)
        prop = torch.rand(8, 2).sort(-1).values
        assert mk.K_G(prop, features=f1).shape == (8, 2)

    def test_default_still_learnable(self):
        mk = torchmodal.MultiAgentKripke(num_agents=3)
        assert isinstance(mk.epistemic_access, mnn.LearnableAccessibility)
        assert mk.get_epistemic_accessibility(torch.randn(3, 2)).shape == (3, 3)


# ---------------------------------------------------------------------------
# 14. round_and_certify promised a witness path and did not return one
# ---------------------------------------------------------------------------


class TestCertificateWitnesses:
    def _chain(self):
        A = torch.zeros(4, 4)
        A[0, 1] = A[1, 2] = A[2, 3] = 0.9
        A[3, 3] = 0.9
        return A

    def test_ef_witness_is_a_path_to_the_goal(self):
        phi = torch.tensor([0.0, 0.0, 0.0, 1.0])
        r = verify.round_and_certify(phi, self._chain(), operator="ef")
        assert r.verdicts[0] == verify.Verdict.PROVEN
        assert r.witnesses is not None
        assert r.witnesses[0] == [0, 1, 2, 3]
        assert r.witnesses[3] == [3]

    def test_ex_witness_is_one_step(self):
        phi = torch.tensor([0.0, 0.0, 0.0, 1.0])
        r = verify.round_and_certify(phi, self._chain(), operator="ex")
        assert r.verdicts[2] == verify.Verdict.PROVEN
        assert r.witnesses is not None and r.witnesses[2] == [2, 3]
        assert r.witnesses[0] is None

    def test_other_operators_report_none(self):
        phi = torch.tensor([1.0, 1.0, 1.0, 1.0])
        r = verify.round_and_certify(phi, self._chain(), operator="ag")
        assert r.witnesses is None


# ---------------------------------------------------------------------------
# 15. Validation by `assert` is stripped under python -O
# ---------------------------------------------------------------------------


class TestValidationRaises:
    def test_world_names_length(self):
        with pytest.raises(ValueError, match="world_names"):
            KripkeModel(3, mnn.FixedAccessibility(torch.eye(3)), world_names=["a"])

    def test_reduction(self):
        with pytest.raises(ValueError, match="reduction"):
            ContradictionLoss(reduction="max")
        with pytest.raises(ValueError, match="reduction"):
            SemanticLoss(reduction="max")
