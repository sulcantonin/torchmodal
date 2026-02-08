"""
Scalability Ring — Accessibility Learning with Box & Diamond
=============================================================

Reproduces the structural ablation from ablation_final.py and
MLNN_AccesbilityScalabilityAblation.ipynb.

Tests learning a ring-structured accessibility relation using
consistency (□) and expansion (♢) losses. Runs ablations over
temperature, top-k masking, and learnable vs fixed structure.

Uses:
  - torchmodal.nn.LearnableAccessibility
  - torchmodal.functional.softmin, softmax
  - torchmodal.build_ring_accessibility
"""

import torch
import torch.optim as optim
import numpy as np

import torchmodal
from torchmodal import functional as F


def generate_ring_data(num_agents=20, num_props=100):
    """Generate observations with a ring-correlated truth structure plus beacons."""
    # Smooth ring of correlated facts
    V = torch.rand(num_props, num_agents)
    for _ in range(5):
        for i in range(num_agents):
            neighbor = (i + 1) % num_agents
            V[:, i] = 0.8 * V[:, i] + 0.2 * V[:, neighbor]
    V = V.round()

    # Beacons: agent i needs item from agent (i+1) % N
    beacons = torch.zeros(num_agents, num_agents)
    for i in range(num_agents):
        target = (i + 1) % num_agents
        beacons[i, target] = 1.0

    V_full = torch.cat([V, beacons], dim=0)
    beacon_start = num_props
    return V_full, beacon_start


class RingMLNN(torch.nn.Module):
    """MLNN for learning ring structure from box + diamond losses."""

    def __init__(self, observations, num_agents, tau=0.1, top_k=None,
                 learnable=True):
        super().__init__()
        self.num_agents = num_agents
        self.tau = tau

        # Learnable accessibility
        self.access = torchmodal.nn.LearnableAccessibility(
            num_agents, init_bias=0.0, reflexive=False, top_k=top_k
        )
        if not learnable:
            self.access.logits.requires_grad = False

        self.register_buffer("beliefs", observations)

    def forward(self, beacon_start):
        A = self.access()
        B = self.beliefs

        # 1. Consistency Loss (Box): don't trust contradicting neighbors
        general_facts = B[:beacon_start, :]
        disagreement = torch.cdist(
            general_facts.T, general_facts.T, p=1
        ) / beacon_start
        loss_box = torch.mean(A * disagreement)

        # 2. Expansion Loss (Diamond): find someone with the beacon
        beacon_indices = torch.arange(self.num_agents) + beacon_start
        target_facts = B[beacon_indices, :]
        weighted_evidence = A * target_facts

        max_evidence = self.tau * torch.logsumexp(
            weighted_evidence / self.tau, dim=1
        )
        loss_diamond = torch.mean((1.0 - max_evidence) ** 2)

        return loss_box + loss_diamond, A


def train_and_evaluate(config, seeds=3):
    """Train and evaluate a single configuration."""
    num_agents = 20
    A_gt = torchmodal.build_ring_accessibility(num_agents)
    mask_off_diag = ~torch.eye(num_agents, dtype=torch.bool)

    mses = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        V, beacon_idx = generate_ring_data(num_agents)

        model = RingMLNN(
            V, num_agents,
            tau=config["tau"],
            top_k=config["top_k"],
            learnable=config["learnable"],
        )

        params = [p for p in model.parameters() if p.requires_grad]
        if params:
            optimizer = optim.Adam(params, lr=0.05)
            for _ in range(200):
                optimizer.zero_grad()
                loss, _ = model(beacon_idx)
                loss.backward()
                optimizer.step()

        _, A_pred = model(beacon_idx)
        A_final = A_pred.detach()
        mse = torch.mean((A_final[mask_off_diag] - A_gt[mask_off_diag]) ** 2).item()
        mses.append(mse)

    return np.mean(mses), np.std(mses)


def main():
    print("Scalability Ring Ablation (ablation_final.py)")
    print(f"{'Setting':<20} | {'Param':<10} | {'MSE (Mean ± Std)'}")
    print("-" * 55)

    # Temperature sweep
    for tau in [0.05, 0.1, 0.2]:
        m, s = train_and_evaluate({"tau": tau, "top_k": 8, "learnable": True})
        print(f"{'Temperature':<20} | {tau:<10} | {m:.4f} ± {s:.4f}")
    print("-" * 55)

    # Top-k sweep
    for k in [4, 8, 16]:
        m, s = train_and_evaluate({"tau": 0.1, "top_k": k, "learnable": True})
        print(f"{'Top-k Mask':<20} | {k:<10} | {m:.4f} ± {s:.4f}")
    print("-" * 55)

    # Learnable vs fixed
    for learnable, name in [(False, "Fixed R"), (True, "Learned A")]:
        m, s = train_and_evaluate(
            {"tau": 0.1, "top_k": 8, "learnable": learnable}
        )
        print(f"{'Relation':<20} | {name:<10} | {m:.4f} ± {s:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
