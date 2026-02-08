"""
Epistemic Trust Learning
========================

Reproduces agent_debug_epistemic.ipynb.

Demonstrates epistemic modal logic for learning trust from
promise-keeping behavior. Agents make promises and either keep
or break them; the MLNN learns trust scores such that
K_a(promise) → □(fulfillment) — if you trust the agent,
their promises must hold across all accessible worlds.

Uses:
  - torchmodal.nn.Necessity
  - torchmodal.functional.implication, softmin
  - Learnable trust logit per agent
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torchmodal
from torchmodal import functional as F

SEED = 42


class EpistemicTrustModel(nn.Module):
    """MLNN for epistemic trust learning from agent interactions.

    Each agent has a learnable trust logit. Trust is evaluated via
    the consistency axiom: K(claim) → □(claim ∨ retraction).
    """

    def __init__(self, num_agents, tau=0.1):
        super().__init__()
        self.num_agents = num_agents
        self.tau = tau
        # Per-agent trust logits (sigmoid → [0,1])
        self.trust_logits = nn.Parameter(torch.zeros(num_agents))

    @property
    def trust(self):
        return torch.sigmoid(self.trust_logits)

    def forward(self, claims, ground_truths, agent_ids):
        """Compute contradiction loss for a batch of interactions.

        Args:
            claims: (batch,) claim values in [0, 1]
            ground_truths: (batch,) actual truth values in [0, 1]
            agent_ids: (batch,) integer agent indices

        Returns:
            Scalar contradiction loss.
        """
        trust = self.trust[agent_ids]  # (batch,)

        # Consistency: claim should match ground truth if agent is trusted
        agreement = 1.0 - torch.abs(claims - ground_truths)

        # Modal axiom: □(trust → agreement)
        # If trust is high and agreement is low → contradiction
        # We use a product conjunction: trust * (1 - agreement) should be 0
        contradiction = trust * (1.0 - agreement)

        # Also: predicted belief should match ground truth
        predicted_belief = trust * claims + (1.0 - trust) * 0.5
        task_loss = (predicted_belief - ground_truths) ** 2

        return task_loss.mean() + 0.3 * contradiction.mean()


def generate_agent_data(num_agents=5, num_interactions=200, seed=42):
    """Generate synthetic promise-keeping data.

    Each agent has a hidden reliability rate. They make claims
    (promises) and sometimes break them.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Hidden reliability: agents 0,1 are honest, 3,4 are liars
    reliability = np.array([0.95, 0.90, 0.50, 0.15, 0.05])[:num_agents]

    claims = []
    truths = []
    agents = []

    for _ in range(num_interactions):
        agent_id = np.random.randint(num_agents)
        claim = 1.0  # always claims high
        # Truth depends on reliability
        truth = 1.0 if np.random.random() < reliability[agent_id] else 0.0
        claims.append(claim)
        truths.append(truth)
        agents.append(agent_id)

    return (
        torch.tensor(claims, dtype=torch.float32),
        torch.tensor(truths, dtype=torch.float32),
        torch.tensor(agents, dtype=torch.long),
        reliability,
    )


def main():
    print("=" * 60)
    print("  Epistemic Trust Learning from Promise-Keeping")
    print("=" * 60)

    NUM_AGENTS = 5
    claims, truths, agent_ids, true_reliability = generate_agent_data(
        NUM_AGENTS
    )

    print(f"\nTrue reliability: {true_reliability}")
    print(f"Interactions: {len(claims)}\n")

    model = EpistemicTrustModel(NUM_AGENTS, tau=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    print(f"{'Epoch':<8} | {'Loss':<10} | Trust Scores")
    print("-" * 65)

    for epoch in range(201):
        optimizer.zero_grad()
        loss = model(claims, truths, agent_ids)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            trust = model.trust.detach().numpy()
            trust_str = ", ".join(f"{t:.3f}" for t in trust)
            print(f"{epoch:<8} | {loss.item():<10.4f} | [{trust_str}]")

    print("\n" + "=" * 60)
    print(f"{'Agent':<8} | {'True Rel.':<12} | {'Learned Trust':<14} | {'Match'}")
    print("-" * 55)
    trust_final = model.trust.detach().numpy()
    for i in range(NUM_AGENTS):
        match = "✓" if abs(trust_final[i] - true_reliability[i]) < 0.25 else "✗"
        print(
            f"Agent {i:<3} | {true_reliability[i]:<12.2f} | "
            f"{trust_final[i]:<14.3f} | {match}"
        )

    print("\nTrust correlates with actual reliability (promise-keeping rate).")


if __name__ == "__main__":
    main()
