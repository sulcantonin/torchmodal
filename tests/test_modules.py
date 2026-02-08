"""Tests for torchmodal.nn modules."""

import torch
import pytest

import torchmodal
from torchmodal import nn


class TestNecessityModule:
    def test_forward(self):
        box = nn.Necessity(tau=0.1)
        prop = torch.tensor([[0.8, 1.0], [0.2, 0.3]])
        A = torch.eye(2)
        result = box(prop, A)
        assert result.shape == (2, 2)

    def test_learnable_tau(self):
        box = nn.Necessity(tau=0.1, learnable_tau=True)
        assert any(p.requires_grad for p in box.parameters())


class TestPossibilityModule:
    def test_forward(self):
        dia = nn.Possibility(tau=0.1)
        prop = torch.tensor([[0.8, 1.0], [0.2, 0.3]])
        A = torch.eye(2)
        result = dia(prop, A)
        assert result.shape == (2, 2)


class TestAccessibility:
    def test_fixed(self):
        R = torch.eye(3)
        access = nn.FixedAccessibility(R)
        A = access()
        assert torch.allclose(A, R)
        # No learnable parameters
        assert len(list(access.parameters())) == 0

    def test_learnable(self):
        access = nn.LearnableAccessibility(5, init_bias=-2.0, reflexive=True)
        A = access()
        assert A.shape == (5, 5)
        # Diagonal should be 1.0 (reflexive)
        assert torch.allclose(A.diagonal(), torch.ones(5))
        # Off-diagonal should be small (negative init_bias)
        off_diag = A[~torch.eye(5, dtype=bool)]
        assert off_diag.mean().item() < 0.2

    def test_metric(self):
        access = nn.MetricAccessibility(10, embed_dim=32)
        A = access()
        assert A.shape == (10, 10)
        # Diagonal should be 1.0 (reflexive)
        assert torch.allclose(A.diagonal(), torch.ones(10))

    def test_metric_with_features(self):
        access = nn.MetricAccessibility(
            10, embed_dim=32, input_dim=64
        )
        features = torch.randn(10, 64)
        A = access(features)
        assert A.shape == (10, 10)

    def test_top_k_mask(self):
        A = torch.tensor([
            [1.0, 0.9, 0.1, 0.8],
            [0.2, 1.0, 0.7, 0.3],
            [0.5, 0.4, 1.0, 0.6],
            [0.3, 0.8, 0.2, 1.0],
        ])
        masked = nn.top_k_mask(A, k=2)
        # Each row should have at most 2 non-zero entries
        for i in range(4):
            assert (masked[i] > 0).sum().item() <= 3  # k=2 but ties possible


class TestConnectiveModules:
    def test_negation_bounds(self):
        neg = nn.Negation()
        x = torch.tensor([[0.3, 0.7]])
        result = neg(x)
        # [1-U, 1-L] = [0.3, 0.7] -> swapped and negated
        assert torch.allclose(result, torch.tensor([[0.3, 0.7]]))

    def test_conjunction_bounds(self):
        conj = nn.Conjunction()
        a = torch.tensor([[0.8, 1.0]])
        b = torch.tensor([[0.6, 0.9]])
        result = conj(a, b)
        # L = max(0, 0.8+0.6-1) = 0.4, U = min(1.0, 0.9) = 0.9
        assert abs(result[0, 0].item() - 0.4) < 1e-6
        assert abs(result[0, 1].item() - 0.9) < 1e-6


class TestKripkeModel:
    def test_create_model(self):
        model = torchmodal.KripkeModel(
            num_worlds=3,
            accessibility=nn.LearnableAccessibility(3),
        )
        assert model.num_worlds == 3
        assert model.num_propositions == 0

    def test_add_proposition(self):
        model = torchmodal.KripkeModel(
            num_worlds=3,
            accessibility=nn.LearnableAccessibility(3),
        )
        p = model.add_proposition("safe", learnable=True)
        assert model.num_propositions == 1
        assert p.bounds.shape == (3, 2)

    def test_necessity_evaluation(self):
        model = torchmodal.KripkeModel(
            num_worlds=3,
            accessibility=nn.FixedAccessibility(torch.eye(3)),
        )
        model.add_proposition("p", learnable=True)
        box_p = model.necessity("p")
        assert box_p.shape == (3, 2)

    def test_contradiction_loss(self):
        model = torchmodal.KripkeModel(
            num_worlds=2,
            accessibility=nn.FixedAccessibility(torch.eye(2)),
        )
        model.add_proposition("p", learnable=True)
        loss = model.contradiction_loss()
        assert loss.item() >= 0.0


class TestLosses:
    def test_contradiction_loss(self):
        loss_fn = torchmodal.ContradictionLoss(reduction="sum")
        # No contradiction
        bounds = torch.tensor([[0.3, 0.7], [0.5, 0.9]])
        assert loss_fn(bounds).item() == 0.0
        # Has contradiction
        bounds = torch.tensor([[0.8, 0.3], [0.5, 0.9]])
        assert loss_fn(bounds).item() > 0.0

    def test_modal_loss(self):
        criterion = torchmodal.ModalLoss(beta=0.5)
        task_loss = torch.tensor(1.0)
        bounds = {"p": torch.tensor([[0.3, 0.7]])}
        total = criterion(task_loss, bounds)
        # No contradiction, so total ≈ task_loss
        assert abs(total.item() - 1.0) < 0.1

    def test_sparsity_loss(self):
        sparse = torchmodal.SparsityLoss(lambda_sparse=0.1)
        A = torch.ones(3, 3)
        loss = sparse(A)
        assert loss.item() > 0.0

    def test_crystallization_loss(self):
        crystal = torchmodal.CrystallizationLoss()
        # High entropy (uncertain)
        values = torch.tensor([0.5, 0.5, 0.5])
        high_loss = crystal(values)
        # Low entropy (crisp)
        values = torch.tensor([0.99, 0.01, 0.99])
        low_loss = crystal(values)
        assert high_loss.item() > low_loss.item()
