"""Tests for the precision, batching and guardrail additions.

Covers:

- :func:`torchmodal.functional.auto_tau` and the ``precision=`` keyword —
  stating a bracket width instead of a temperature.
- Batched evaluation of the modal operators, which must be *bit-identical*
  to looping over the batch.
- :func:`torchmodal.diagnostics.vacuity_report` — satisfied because true, or
  satisfied because empty?
- The monotonicity table, checked empirically rather than asserted.
- Axioms D (serial) and 5 (Euclidean) in ``AxiomRegularization``.
"""

import math

import pytest
import torch

from torchmodal import AxiomRegularization
from torchmodal import functional as F
from torchmodal.diagnostics import (
    MONOTONICITY,
    monotone_in_accessibility,
    vacuity_report,
)


class TestAutoTau:
    """The inverse of the bracket: ask for a width, get a temperature."""

    @pytest.mark.parametrize("target", [0.01, 0.05, 0.2])
    def test_closed_form_is_safe_for_any_proposition(self, target):
        """tau = eps / log n must hold for bounds it never saw."""
        torch.manual_seed(0)
        for _ in range(20):
            n = int(torch.randint(2, 12, (1,)).item())
            A = torch.rand(n, n)
            tau = F.auto_tau(A, target)
            bounds = torch.rand(n, 2).sort(dim=1).values
            width = F.box_width_entropy(A, bounds, tau=tau).max().item()
            assert width <= target + 1e-6

    @pytest.mark.parametrize("target", [0.02, 0.05, 0.1])
    def test_exact_mode_is_safe(self, target):
        torch.manual_seed(1)
        A = torch.rand(12, 12)
        bounds = torch.rand(12, 2).sort(dim=1).values
        tau = F.auto_tau(A, target, prop_bounds=bounds)
        width = F.box_width_entropy(A, bounds, tau=tau).max().item()
        assert width <= target + 1e-6

    def test_exact_mode_is_tighter_than_closed_form(self):
        """A larger tau for the same guarantee — better-conditioned grads."""
        torch.manual_seed(1)
        A = torch.rand(12, 12)
        bounds = torch.rand(12, 2).sort(dim=1).values
        for target in (0.02, 0.05, 0.1):
            cf = F.auto_tau(A, target)
            ex = F.auto_tau(A, target, prop_bounds=bounds)
            assert ex > cf

    def test_width_is_monotone_in_tau(self):
        """The property that makes the bisection well posed."""
        torch.manual_seed(2)
        A = torch.rand(9, 9)
        bounds = torch.rand(9, 2).sort(dim=1).values
        taus = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        widths = [
            F.box_width_entropy(A, bounds, tau=t).max().item() for t in taus
        ]
        assert widths == sorted(widths)

    def test_matches_the_paper_rule(self):
        A = torch.ones(16, 16)
        assert F.auto_tau(A, 0.1) == pytest.approx(0.1 / math.log(16))

    def test_respects_top_k(self):
        A = torch.ones(50, 50)
        assert F.auto_tau(A, 0.1, top_k=4) == pytest.approx(
            0.1 / math.log(4)
        )

    def test_single_term_is_finite(self):
        A = torch.ones(5, 5)
        assert math.isfinite(F.auto_tau(A, 0.1, top_k=1))

    @pytest.mark.parametrize("bad", [0.0, -0.1])
    def test_rejects_non_positive_target(self, bad):
        with pytest.raises(ValueError, match="target_width"):
            F.auto_tau(torch.ones(4, 4), bad)


class TestPrecisionKeyword:
    @pytest.mark.parametrize("op", [F.necessity, F.possibility])
    def test_precision_achieves_the_width(self, op):
        A = torch.ones(8, 8)
        bounds = torch.full((8, 2), 0.5)
        out = op(bounds, A, precision=0.05)
        assert (out[:, 1] - out[:, 0]).max().item() <= 0.05 + 1e-6

    def test_precision_overrides_tau(self):
        A = torch.ones(8, 8)
        bounds = torch.full((8, 2), 0.5)
        loose = F.necessity(bounds, A, tau=1.0)
        tight = F.necessity(bounds, A, tau=1.0, precision=0.02)
        assert (tight[:, 1] - tight[:, 0]).max() < (
            loose[:, 1] - loose[:, 0]
        ).max()

    def test_omitting_precision_is_unchanged(self):
        A = torch.rand(6, 6)
        bounds = torch.rand(6, 2).sort(dim=1).values
        assert torch.equal(
            F.necessity(bounds, A, tau=0.1),
            F.necessity(bounds, A, tau=0.1, precision=None),
        )


class TestBatching:
    """Batched evaluation must be bit-identical to looping."""

    @pytest.mark.parametrize("op", [F.necessity, F.possibility])
    @pytest.mark.parametrize("shape", [(4, 6), (3, 9), (2, 2)])
    def test_matches_loop(self, op, shape):
        B, N = shape
        torch.manual_seed(0)
        bounds = torch.rand(B, N, 2).sort(dim=-1).values
        A = torch.rand(B, N, N)
        batched = op(bounds, A, tau=0.1)
        looped = torch.stack([op(bounds[b], A[b], tau=0.1) for b in range(B)])
        assert batched.shape == (B, N, 2)
        assert torch.equal(batched, looped)

    def test_box_width_entropy_matches_loop(self):
        torch.manual_seed(1)
        B, N = 4, 7
        bounds = torch.rand(B, N, 2).sort(dim=-1).values
        A = torch.rand(B, N, N)
        assert torch.equal(
            F.box_width_entropy(A, bounds, tau=0.1),
            torch.stack(
                [
                    F.box_width_entropy(A[b], bounds[b], tau=0.1)
                    for b in range(B)
                ]
            ),
        )

    def test_top_k_matches_loop(self):
        torch.manual_seed(2)
        B, N = 5, 7
        bounds = torch.rand(B, N, 2).sort(dim=-1).values
        A = torch.rand(B, N, N)
        assert torch.equal(
            F.necessity(bounds, A, tau=0.1, top_k=3),
            torch.stack(
                [
                    F.necessity(bounds[b], A[b], tau=0.1, top_k=3)
                    for b in range(B)
                ]
            ),
        )

    def test_batched_point_valued(self):
        torch.manual_seed(3)
        B, N = 5, 7
        pts = torch.rand(B, N)
        A = torch.rand(B, N, N)
        out = F.necessity(pts, A, tau=0.1)
        assert out.shape == (B, N)
        assert torch.equal(
            out, torch.stack([F.necessity(pts[b], A[b], tau=0.1) for b in range(B)])
        )

    def test_two_worlds_is_not_ambiguous(self):
        """|W| == 2 is where a shape-only rule would guess wrong.

        ``(2, 2)`` is bounds for two worlds, ``(2,)`` is point values — the
        rule keys on rank relative to the relation, not on the trailing
        extent.
        """
        A = torch.rand(2, 2)
        assert F.necessity(torch.rand(2, 2), A, tau=0.1).shape == (2, 2)
        assert F.necessity(torch.rand(2), A, tau=0.1).shape == (2,)

    def test_extra_leading_dims(self):
        torch.manual_seed(4)
        bounds = torch.rand(2, 3, 5, 2).sort(dim=-1).values
        A = torch.rand(2, 3, 5, 5)
        assert F.necessity(bounds, A, tau=0.1).shape == (2, 3, 5, 2)

    def test_gradients_flow_through_the_batch(self):
        B, N = 3, 5
        A = torch.rand(B, N, N, requires_grad=True)
        bounds = torch.rand(B, N, 2).sort(dim=-1).values
        F.necessity(bounds, A, tau=0.1)[..., 0].sum().backward()
        assert A.grad is not None
        # every batch element must receive gradient, not just the first
        assert (A.grad.flatten(1).abs().sum(dim=1) > 0).all()


class TestBoxWidthEntropyNaN:
    """Regression: the entropy went NaN at small tau in float32."""

    @pytest.mark.parametrize("tau", [1.0, 0.1, 0.01, 0.008, 1e-3, 1e-4])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_finite_at_every_temperature(self, tau, dtype):
        torch.manual_seed(0)
        A = torch.rand(12, 12, dtype=dtype)
        bounds = torch.rand(12, 2, dtype=dtype).sort(dim=1).values
        width = F.box_width_entropy(A, bounds, tau=tau)
        assert torch.isfinite(width).all()
        assert (width >= 0).all()

    def test_identity_still_holds_after_the_fix(self):
        torch.manual_seed(1)
        A = torch.rand(10, 10, dtype=torch.float64)
        bounds = torch.rand(10, 2, dtype=torch.float64).sort(dim=1).values
        terms = (1.0 - A) + bounds[:, 1].unsqueeze(0)
        gap = F.conv_pool(terms, -terms, tau=0.1, dim=1) - F.smooth_min(
            terms, tau=0.1, dim=1
        )
        assert torch.allclose(
            F.box_width_entropy(A, bounds, tau=0.1), gap, atol=1e-12
        )


class TestVacuityReport:
    """Satisfied because TRUE, or satisfied because EMPTY?"""

    def test_box_on_an_unsupported_proposition_is_vacuous(self):
        torch.manual_seed(0)
        A = torch.rand(6, 6)
        phi = torch.zeros(6, 2)
        r = vacuity_report(lambda a: F.necessity(phi, a)[:, 0], A)
        assert r["vacuous"]
        assert r["direction"] == "maximal_when_empty"
        assert r["vacuous_value"] > r["observed_value"]

    def test_diamond_is_maximal_when_full(self):
        torch.manual_seed(0)
        A = torch.rand(6, 6)
        phi = torch.ones(6, 2)
        r = vacuity_report(lambda a: F.possibility(phi, a)[:, 1], A)
        assert r["direction"] == "maximal_when_full"
        assert not r["vacuous"]

    def test_a_genuinely_informative_term_is_not_vacuous(self):
        A = torch.ones(5, 5)
        phi = torch.ones(5, 2)
        r = vacuity_report(lambda a: F.possibility(phi, a)[:, 1], A)
        assert r["margin_over_vacuous"] > 0

    def test_does_not_disturb_gradients(self):
        A = torch.rand(5, 5, requires_grad=True)
        phi = torch.zeros(5, 2)
        vacuity_report(lambda a: F.necessity(phi, a)[:, 0], A)
        assert A.grad is None


class TestMonotonicity:
    def test_table_matches_measurement(self):
        """The table is checked, not asserted."""
        torch.manual_seed(0)
        N, tau, trials = 6, 0.1, 60
        observed = {}
        for op_name, op in (
            ("necessity", F.necessity),
            ("possibility", F.possibility),
        ):
            for idx, end in ((0, "L"), (1, "U")):
                violations = 0
                for _ in range(trials):
                    A = torch.rand(N, N)
                    bounds = torch.rand(N, 2).sort(dim=1).values
                    base = op(bounds, A, tau=tau)[:, idx]
                    # raise one entry: a monotone endpoint cannot decrease
                    i, j = (
                        int(torch.randint(N, (1,))),
                        int(torch.randint(N, (1,))),
                    )
                    A2 = A.clone()
                    A2[i, j] = min(1.0, A[i, j].item() + 0.3)
                    bumped = op(bounds, A2, tau=tau)[:, idx]
                    if op_name == "necessity":
                        # more access lowers (1-A)+L, so L_box may only fall
                        if (bumped - base).max() > 1e-6:
                            violations += 1
                    else:
                        if (bumped - base).min() < -1e-6:
                            violations += 1
                observed[f"{op_name}.{end}"] = violations

        for key, entry in MONOTONICITY.items():
            if entry["monotone"]:
                assert observed[key] == 0, (key, observed[key])

    def test_lookup_and_aliases(self):
        assert monotone_in_accessibility("necessity", "L") is True
        assert monotone_in_accessibility("box", "l") is True
        assert monotone_in_accessibility("necessity", "U") is False
        assert monotone_in_accessibility("diamond", "U") is True
        assert monotone_in_accessibility("possibility", "L") is False

    def test_rejects_unknown(self):
        with pytest.raises(KeyError):
            monotone_in_accessibility("until", "L")


class TestNewAxioms:
    def test_identity_fails_hollow_seriality(self):
        """The trap: the identity has no dead ends but coordinates nothing."""
        eye = torch.eye(5)
        naive = AxiomRegularization(seriality=1.0, serial_hollow=False)(eye)
        hollow = AxiomRegularization(seriality=1.0, serial_hollow=True)(eye)
        assert naive.item() == pytest.approx(0.0, abs=1e-8)
        assert hollow.item() == pytest.approx(1.0, abs=1e-6)

    def test_serial_relation_passes(self):
        N = 5
        ring = torch.zeros(N, N)
        ring[torch.arange(N), (torch.arange(N) + 1) % N] = 1.0
        for hollow in (True, False):
            loss = AxiomRegularization(
                seriality=1.0, serial_hollow=hollow
            )(ring)
            assert loss.item() == pytest.approx(0.0, abs=1e-8)

    def test_dead_end_is_penalised(self):
        N = 5
        ring = torch.zeros(N, N)
        ring[torch.arange(N), (torch.arange(N) + 1) % N] = 1.0
        ring[2] = 0.0
        assert AxiomRegularization(seriality=1.0)(ring).item() > 0

    def test_seriality_is_trainable(self):
        """A few hundred steps must give the dead-end row a successor."""
        N = 5
        logits = torch.full((N, N), -3.0, requires_grad=True)
        opt = torch.optim.Adam([logits], lr=0.1)
        reg = AxiomRegularization(seriality=1.0)
        for _ in range(400):
            opt.zero_grad()
            reg(torch.sigmoid(logits)).backward()
            opt.step()
        A = torch.sigmoid(logits).detach()
        eye = torch.eye(N)
        assert (A * (1 - eye)).max(dim=-1).values.min() > 0.9

    def test_euclidean_complete_relation_passes(self):
        assert AxiomRegularization(euclidean=1.0)(
            torch.ones(5, 5)
        ).item() == pytest.approx(0.0, abs=1e-8)

    def test_euclidean_penalises_a_ring(self):
        N = 5
        ring = torch.zeros(N, N)
        ring[torch.arange(N), (torch.arange(N) + 1) % N] = 1.0
        assert AxiomRegularization(euclidean=1.0)(ring).item() > 0

    def test_euclidean_antecedent_is_not_vacuous_when_sparse(self):
        """Godel antecedent, not Lukasiewicz.

        With Lukasiewicz the antecedent min(1, a+b-1) collapses to 0 whenever
        the two access values sum below 1, which would make the axiom
        vacuously satisfied on exactly the sparse relations it should catch.
        """
        A = torch.zeros(3, 3)
        A[0, 1] = A[0, 2] = 0.4      # sums to 0.8 < 1
        assert AxiomRegularization(euclidean=1.0)(A).item() > 0

    def test_defaults_are_unchanged(self):
        torch.manual_seed(0)
        A = torch.rand(6, 6)
        assert AxiomRegularization()(A).item() == 0.0
        old = AxiomRegularization(
            reflexivity=1.0, transitivity=0.5, symmetry=0.2
        )(A)
        new = AxiomRegularization(
            reflexivity=1.0,
            transitivity=0.5,
            symmetry=0.2,
            seriality=0.0,
            euclidean=0.0,
        )(A)
        assert torch.equal(old, new)

    def test_repr_mentions_the_new_axioms(self):
        r = repr(AxiomRegularization(seriality=1.0, euclidean=2.0))
        assert "seriality" in r and "euclidean" in r
