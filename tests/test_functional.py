"""Tests for torchmodal.functional."""

import torch
import pytest

from torchmodal import functional as F


class TestSoftmin:
    def test_below_true_min(self):
        """softmin must be a lower bound on min."""
        x = torch.tensor([0.3, 0.7, 0.5])
        result = F.softmin(x, tau=0.1)
        assert result.item() <= x.min().item() + 1e-6

    def test_converges_to_min(self):
        """As tau -> 0, softmin -> min."""
        x = torch.tensor([0.3, 0.7, 0.5])
        result = F.softmin(x, tau=0.001)
        assert abs(result.item() - 0.3) < 0.01

    def test_differentiable(self):
        x = torch.tensor([0.3, 0.7, 0.5], requires_grad=True)
        loss = F.softmin(x, tau=0.1)
        loss.backward()
        assert x.grad is not None


class TestSoftmax:
    def test_above_true_max(self):
        """softmax must be an upper bound on max."""
        x = torch.tensor([0.3, 0.7, 0.5])
        result = F.softmax(x, tau=0.1)
        assert result.item() >= x.max().item() - 1e-6

    def test_converges_to_max(self):
        x = torch.tensor([0.3, 0.7, 0.5])
        result = F.softmax(x, tau=0.001)
        assert abs(result.item() - 0.7) < 0.01


class TestConvPool:
    def test_with_positive_z_lower_bound_max(self):
        """conv_pool(x, x) is a lower bound on max."""
        x = torch.tensor([0.2, 0.8, 0.5])
        result = F.conv_pool(x, x, tau=0.1)
        assert result.item() <= x.max().item() + 0.01
        assert result.item() >= x.min().item() - 0.01


class TestConnectives:
    def test_negation(self):
        x = torch.tensor([0.0, 0.5, 1.0])
        result = F.negation(x)
        expected = torch.tensor([1.0, 0.5, 0.0])
        assert torch.allclose(result, expected)

    def test_conjunction(self):
        a = torch.tensor([1.0, 0.7, 0.3])
        b = torch.tensor([1.0, 0.5, 0.2])
        result = F.conjunction(a, b)
        # Łukasiewicz: max(0, a+b-1)
        expected = torch.tensor([1.0, 0.2, 0.0])
        assert torch.allclose(result, expected, atol=1e-6)

    def test_disjunction(self):
        a = torch.tensor([0.0, 0.7, 0.8])
        b = torch.tensor([0.0, 0.5, 0.6])
        result = F.disjunction(a, b)
        expected = torch.tensor([0.0, 1.0, 1.0])
        assert torch.allclose(result, expected, atol=1e-6)

    def test_implication(self):
        # a=1, b=0 → 0 (false)
        result = F.implication(torch.tensor(1.0), torch.tensor(0.0))
        assert abs(result.item()) < 1e-6
        # a=0, b=anything → 1 (vacuously true)
        result = F.implication(torch.tensor(0.0), torch.tensor(0.3))
        assert abs(result.item() - 1.0) < 1e-6


class TestNecessity:
    def test_all_true_reflexive(self):
        """□ϕ should be high if ϕ is true in all accessible worlds."""
        prop = torch.tensor([[0.9, 1.0], [0.9, 1.0], [0.9, 1.0]])
        A = torch.eye(3)
        result = F.necessity(prop, A, tau=0.1)
        # With reflexive-only access, □ϕ ≈ ϕ per world
        assert result[:, 0].min().item() > 0.5

    def test_one_false_lowers_box(self):
        """If one accessible world has false ϕ, □ϕ drops."""
        prop = torch.tensor([[0.9, 1.0], [0.1, 0.2], [0.9, 1.0]])
        A = torch.ones(3, 3)  # full access
        result = F.necessity(prop, A, tau=0.1)
        # Lower bound should be low because world 1 is false
        assert result[0, 0].item() < 0.5

    def test_point_valued(self):
        """Test with point-valued (1D) input."""
        prop = torch.tensor([0.9, 0.1, 0.8])
        A = torch.eye(3)
        result = F.necessity(prop, A, tau=0.1)
        assert result.shape == (3,)


class TestPossibility:
    def test_one_true_is_enough(self):
        """♢ϕ should be high if at least one accessible world has ϕ true."""
        prop = torch.tensor([[0.1, 0.2], [0.9, 1.0], [0.1, 0.2]])
        A = torch.ones(3, 3)
        result = F.possibility(prop, A, tau=0.1)
        # Upper bound should be high due to world 1
        assert result[0, 1].item() > 0.5

    def test_duality_upper_bound(self):
        """♢ϕ upper ≡ ¬□¬ϕ upper — modal duality via logsumexp identity.

        The identity softmax(x) = 1 - softmin(1-x) ensures exact duality
        for the logsumexp-based bounds (U_diamond and L_box). The conv_pool
        bounds (L_diamond and U_box) are intentionally different sound
        approximations and need not match exactly.
        """
        prop = torch.tensor([[0.7, 0.9], [0.3, 0.5]])
        A = torch.ones(2, 2) * 0.8
        tau = 0.01
        dia = F.possibility(prop, A, tau=tau)
        # ¬ϕ: swap and negate bounds
        neg_prop = torch.stack([1.0 - prop[:, 1], 1.0 - prop[:, 0]], dim=-1)
        box_neg = F.necessity(neg_prop, A, tau=tau)
        neg_box_neg = torch.stack(
            [1.0 - box_neg[:, 1], 1.0 - box_neg[:, 0]], dim=-1
        )
        # Upper bounds of ♢ and ¬□¬ both use logsumexp — should match
        assert torch.allclose(dia[:, 1], neg_box_neg[:, 1], atol=0.05)


class TestContradiction:
    def test_no_contradiction(self):
        bounds = torch.tensor([[0.3, 0.7], [0.5, 0.9]])
        assert F.contradiction(bounds).item() == 0.0

    def test_has_contradiction(self):
        bounds = torch.tensor([[0.8, 0.3], [0.5, 0.9]])
        assert F.contradiction(bounds).item() > 0.0
        assert abs(F.contradiction(bounds).item() - 0.5) < 1e-6
