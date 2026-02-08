"""
Doxastic Belief Calibration
============================

Reproduces agent_debug_doxastic.ipynb.

Demonstrates doxastic modal logic for belief calibration and
hallucination detection. Agents have beliefs (B_a(ϕ)) that may
differ from reality. The model learns per-agent calibration
parameters to detect when belief strength exceeds warranted
confidence (hallucination).

Uses:
  - torchmodal.DoxasticOperator
  - torchmodal.functional.conjunction, negation
  - Per-agent calibration parameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torchmodal
from torchmodal import functional as F

SEED = 42


class DoxasticMLNN(nn.Module):
    """Doxastic MLNN for belief calibration.

    Learns per-agent calibration parameters. Detects hallucinations
    where B_a(ϕ) is high but ϕ is actually false.
    """

    def __init__(self, num_agents):
        super().__init__()
        self.num_agents = num_agents
        # Per-agent calibration: multiplicative factor on belief
        self.calibration_logits = nn.Parameter(torch.zeros(num_agents))

    @property
    def calibration(self):
        # Calibration in (0, 2) — values < 1 reduce belief, > 1 amplify
        return torch.sigmoid(self.calibration_logits) * 2.0

    def forward(self, belief_strength, ground_truth, agent_ids):
        """Compute doxastic loss for a batch.

        Args:
            belief_strength: (batch,) raw belief strength in [0, 1]
            ground_truth: (batch,) actual truth in {0, 1}
            agent_ids: (batch,) integer agent indices

        Returns:
            Scalar doxastic loss.
        """
        cal = self.calibration[agent_ids]
        calibrated_belief = torch.clamp(belief_strength * cal, 0.0, 1.0)

        # Hallucination detection: belief is high but reality is low
        # B_a(ϕ) ∧ ¬ϕ → contradiction
        reality_gap = F.negation(ground_truth)
        hallucination_loss = F.conjunction(calibrated_belief, reality_gap)

        # Correct confidence: if ground truth is high, belief should be high
        correct_confidence_loss = F.conjunction(
            F.negation(calibrated_belief), ground_truth
        )

        # Calibration regularization: prefer calibration near 1.0
        cal_reg = torch.abs(cal - 1.0).mean()

        loss = (
            hallucination_loss.mean()
            + 0.5 * correct_confidence_loss.mean()
            + 0.1 * cal_reg
        )
        return loss, calibrated_belief


def generate_agent_interactions(
    num_agents=4, num_interactions=300, seed=42
):
    """Generate synthetic QA interactions with varying agent accuracy."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Agent profiles:
    # Agent 0: Accurate & calibrated (high belief → usually correct)
    # Agent 1: Overconfident (high belief → often wrong = hallucinator)
    # Agent 2: Underconfident (low belief → usually correct)
    # Agent 3: Random
    accuracy = np.array([0.90, 0.30, 0.85, 0.50])[:num_agents]
    confidence = np.array([0.85, 0.95, 0.40, 0.60])[:num_agents]

    beliefs = []
    truths = []
    agents = []

    for _ in range(num_interactions):
        agent_id = np.random.randint(num_agents)
        # Belief strength is drawn near agent's confidence level
        belief = np.clip(
            confidence[agent_id] + np.random.normal(0, 0.1), 0.05, 0.95
        )
        # Ground truth depends on accuracy
        truth = 1.0 if np.random.random() < accuracy[agent_id] else 0.0

        beliefs.append(belief)
        truths.append(truth)
        agents.append(agent_id)

    return (
        torch.tensor(beliefs, dtype=torch.float32),
        torch.tensor(truths, dtype=torch.float32),
        torch.tensor(agents, dtype=torch.long),
        accuracy,
        confidence,
    )


def main():
    print("=" * 60)
    print("  Doxastic Belief Calibration & Hallucination Detection")
    print("=" * 60)

    NUM_AGENTS = 4
    beliefs, truths, agent_ids, true_acc, true_conf = (
        generate_agent_interactions(NUM_AGENTS)
    )

    profiles = [
        "Accurate & Calibrated",
        "HALLUCINATOR (overconf)",
        "Underconfident",
        "Random",
    ]

    print(f"\nAgent profiles:")
    for i in range(NUM_AGENTS):
        print(
            f"  Agent {i}: accuracy={true_acc[i]:.0%}, "
            f"confidence={true_conf[i]:.0%} [{profiles[i]}]"
        )

    model = DoxasticMLNN(NUM_AGENTS)
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    print(f"\n{'Epoch':<8} | {'Loss':<10} | Calibration Factors")
    print("-" * 60)

    for epoch in range(301):
        optimizer.zero_grad()
        loss, _ = model(beliefs, truths, agent_ids)
        loss.backward()
        optimizer.step()

        if epoch % 75 == 0:
            cal = model.calibration.detach().numpy()
            cal_str = ", ".join(f"{c:.3f}" for c in cal)
            print(f"{epoch:<8} | {loss.item():<10.4f} | [{cal_str}]")

    print("\n" + "=" * 60)
    print(f"{'Agent':<8} | {'Profile':<25} | {'Calibration':<13} | {'Effect'}")
    print("-" * 65)
    cal_final = model.calibration.detach().numpy()
    for i in range(NUM_AGENTS):
        if cal_final[i] < 0.8:
            effect = "SUPPRESSED (hallucinator)"
        elif cal_final[i] > 1.2:
            effect = "AMPLIFIED (underconfident)"
        else:
            effect = "~Unchanged (calibrated)"
        print(
            f"Agent {i:<3} | {profiles[i]:<25} | {cal_final[i]:<13.3f} | {effect}"
        )

    print("\nThe model learned to SUPPRESS the overconfident agent's beliefs")
    print("(hallucination detection) and AMPLIFY the underconfident agent.")


if __name__ == "__main__":
    main()
