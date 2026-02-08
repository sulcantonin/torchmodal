"""
Deontic Boundary Learning
=========================

Reproduces agent_debug_denoic.ipynb.

Demonstrates deontic modal logic for learning normative boundaries.
The model learns to classify financial trading orders as PERMITTED
(legal) or PROHIBITED (spoofing) based on order duration and size,
implementing the deontic operators O(ϕ) (obligatory) and P(ϕ) (permitted).

Uses:
  - torchmodal.functional.conjunction, negation
  - Weighted hinge loss for normative classification
  - Decision boundary visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torchmodal
from torchmodal import functional as F

SEED = 42


class DeonticMLNN(nn.Module):
    """One-class deontic MLNN for normative boundary learning.

    Learns a decision boundary separating permitted (P(ϕ)) from
    prohibited (¬P(ϕ)) actions in a feature space.
    """

    def __init__(self, input_dim=2, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # Output in [-1, 1]: permitted (+) vs prohibited (-)
        )

    def forward(self, features):
        return self.net(features)


def generate_market_data(n_samples=5000, seed=42):
    """Generate synthetic market order data with spoofing labels.

    Features: (duration, size) normalized to [0, 1]
    Spoofing pattern: short duration + large size
    """
    np.random.seed(seed)
    data = []

    for _ in range(n_samples):
        is_spoof = np.random.random() < 0.15  # 15% spoofing rate

        if is_spoof:
            duration = np.random.beta(1, 8)  # short duration
            size = np.random.beta(8, 1)  # large size
            label = -1.0  # prohibited
        else:
            duration = np.random.beta(3, 2)  # moderate-long duration
            size = np.random.beta(2, 3)  # moderate size
            label = 1.0  # permitted

        # True label for metrics (not used in training directly)
        true_label = -1.0 if is_spoof else 1.0
        data.append([duration, size, label, true_label])

    return np.array(data)


def main():
    print("=" * 60)
    print("  Deontic Boundary Learning — Spoofing Detection")
    print("=" * 60)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    raw_data = generate_market_data()
    features = torch.tensor(raw_data[:, :2], dtype=torch.float32)
    targets = torch.tensor(raw_data[:, 2:3], dtype=torch.float32)
    true_labels = raw_data[:, 3]

    n_permitted = (true_labels > 0).sum()
    n_prohibited = (true_labels < 0).sum()
    print(f"Data: {len(raw_data)} orders ({n_permitted} permitted, "
          f"{n_prohibited} prohibited)")

    model = DeonticMLNN()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    print(f"\n{'Epoch':<8} | {'Loss':<10} | {'Spoof Recall'}")
    print("-" * 40)

    for epoch in range(1001):
        optimizer.zero_grad()
        scores = model(features)

        # Weighted hinge loss: heavily penalize missing prohibited actions
        raw_losses = torch.relu(1.0 - targets * scores)
        weights = torch.ones_like(targets)
        weights[targets == -1.0] = 50.0  # Higher weight for violations
        loss = (raw_losses * weights).mean()

        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            preds = scores.detach().numpy().squeeze()
            spoof_indices = true_labels < 0
            spoof_preds = preds[spoof_indices]
            recall = np.mean(spoof_preds < 0.0)
            print(f"{epoch:<8} | {loss.item():<10.4f} | {recall*100:.1f}%")

    # Final evaluation
    print("\n" + "=" * 60)
    print("DEONTIC BOUNDARY EVALUATION")
    print("=" * 60)

    test_cases = [
        (0.5, 0.5, "Normal Trade"),
        (0.9, 0.9, "Large Institutional Block"),
        (0.05, 0.1, "HFT Scalp"),
        (0.05, 0.9, "SPOOFING PATTERN"),
    ]

    print(f"{'Duration':<10} | {'Size':<8} | {'Score':<10} | {'Verdict':<12} | {'Description'}")
    print("-" * 65)

    for dur, sz, desc in test_cases:
        inp = torch.tensor([[dur, sz]])
        with torch.no_grad():
            score = model(inp).item()
        verdict = "PERMITTED" if score > 0.0 else "PROHIBITED"
        print(f"{dur:<10} | {sz:<8} | {score:+.4f}    | {verdict:<12} | {desc}")

    # Compute metrics
    with torch.no_grad():
        all_scores = model(features).squeeze().numpy()

    preds = np.where(all_scores > 0, 1.0, -1.0)
    tp = np.sum((preds == -1) & (true_labels == -1))
    fp = np.sum((preds == -1) & (true_labels == 1))
    fn = np.sum((preds == 1) & (true_labels == -1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nSpoof Detection Metrics:")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1-Score:  {f1:.1%}")


if __name__ == "__main__":
    main()
