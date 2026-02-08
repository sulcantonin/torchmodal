"""Tests for torchmodal.systems (higher-level modal logics)."""

import torch
import pytest

import torchmodal


class TestEpistemicOperator:
    def test_knowledge_true(self):
        """K_a(ϕ) should be high if ϕ is true in all accessible worlds."""
        K = torchmodal.EpistemicOperator(tau=0.1)
        prop = torch.tensor([[0.9, 1.0], [0.85, 0.95], [0.8, 0.9]])
        # Agent 0 can access all worlds
        agent_access = torch.tensor([1.0, 1.0, 1.0])
        result = K(prop, agent_access)
        assert result[0].item() > 0.5  # lower bound > 0.5

    def test_knowledge_false_if_one_world_false(self):
        K = torchmodal.EpistemicOperator(tau=0.1)
        prop = torch.tensor([[0.9, 1.0], [0.05, 0.1], [0.9, 1.0]])
        agent_access = torch.tensor([1.0, 1.0, 1.0])
        result = K(prop, agent_access)
        # Should be low because world 1 has ϕ ≈ false
        assert result[0].item() < 0.5


class TestTemporalOperator:
    def test_globally(self):
        temporal = torchmodal.TemporalOperator(num_steps=3)
        A = temporal.build_forward_accessibility()
        prop = torch.tensor([[0.9, 1.0], [0.85, 0.95], [0.1, 0.2]])
        result = temporal.globally(prop, A)
        # G(ϕ) at t=0 should be low (ϕ is false at t=2)
        assert result[0, 0].item() < 0.5

    def test_finally(self):
        temporal = torchmodal.TemporalOperator(num_steps=3)
        A = temporal.build_forward_accessibility()
        prop = torch.tensor([[0.1, 0.2], [0.1, 0.2], [0.9, 1.0]])
        result = temporal.finally_(prop, A)
        # F(ϕ) at t=0 should be high (ϕ becomes true at t=2)
        assert result[0, 1].item() > 0.3


class TestMultiAgentKripke:
    def test_creation(self):
        mk = torchmodal.MultiAgentKripke(
            num_agents=3, num_steps=2
        )
        assert mk.num_states == 6

    def test_epistemic_accessibility(self):
        mk = torchmodal.MultiAgentKripke(
            num_agents=3, num_steps=1
        )
        A = mk.get_epistemic_accessibility()
        assert A.shape == (3, 3)
        # Diagonal should be 1 (reflexive)
        assert torch.allclose(A.diagonal(), torch.ones(3))

    def test_K_operator(self):
        mk = torchmodal.MultiAgentKripke(
            num_agents=3, num_steps=1, tau=0.1
        )
        prop = torch.tensor([[0.8, 1.0], [0.7, 0.9], [0.6, 0.8]])
        K_phi = mk.K(prop)
        assert K_phi.shape == (3, 2)


class TestUtils:
    def test_anneal_temperature(self):
        tau = torchmodal.anneal_temperature(
            epoch=0, total_epochs=100, tau_start=2.0, tau_end=0.1
        )
        assert abs(tau - 2.0) < 0.01

        tau = torchmodal.anneal_temperature(
            epoch=99, total_epochs=100, tau_start=2.0, tau_end=0.1
        )
        assert abs(tau - 0.1) < 0.01

    def test_build_sudoku_accessibility(self):
        A = torchmodal.build_sudoku_accessibility(3)
        assert A.shape == (81, 81)
        # Each cell should have 20 neighbors (8 row + 8 col + 4 box unique)
        assert A[0].sum().item() == 20.0

    def test_build_ring_accessibility(self):
        A = torchmodal.build_ring_accessibility(5)
        assert A.shape == (5, 5)
        # Each node: self + one forward neighbor
        for i in range(5):
            assert A[i, i].item() == 1.0
            assert A[i, (i + 1) % 5].item() == 1.0
