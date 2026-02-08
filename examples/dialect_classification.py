"""
Dialect Classification with Modal Logic
========================================

Reproduces MLNN_DIALECTS_MLNN_CP.ipynb (simplified, self-contained).

Demonstrates using necessity (□) and possibility (♢) operators for
dialect classification (American vs British vs Neutral English).
The modal reasoner uses threshold-based Kripke worlds to detect
logical indeterminacy — inputs that belong to neither class.

Standard classifiers force a choice (closed-world assumption);
the MLNN can explicitly abstain via the rule:
  (¬♢HasAmE ∧ ¬♢HasBrE) ∨ (♢HasAmE ∧ ♢HasBrE) → IsNeutral

Uses:
  - torchmodal.functional.negation, conjunction, disjunction
  - torchmodal.bounds_to_labels
  - Threshold-based Kripke worlds (Real, Skeptical, Credulous)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torchmodal
from torchmodal import functional as F


# Feature word sets
AME_FEATURES = {"color", "favorite", "realize", "organize", "truck",
                "apartment", "center", "gray"}
BRE_FEATURES = {"colour", "favourite", "realise", "organise", "lorry",
                "flat", "centre", "grey"}


def generate_sentences(n=3000, seed=42):
    """Generate synthetic sentences with American/British/Neutral labels.

    Each sentence is represented as (has_ame_score, has_bre_score, label).
    """
    np.random.seed(seed)
    data = []

    for _ in range(n):
        category = np.random.choice(["AmE", "BrE", "Neutral"], p=[0.35, 0.35, 0.30])

        if category == "AmE":
            ame_score = np.random.beta(8, 2)  # high AmE signal
            bre_score = np.random.beta(1, 8)  # low BrE signal
        elif category == "BrE":
            ame_score = np.random.beta(1, 8)
            bre_score = np.random.beta(8, 2)
        else:  # Neutral
            ame_score = np.random.beta(1, 5)  # low signals for both
            bre_score = np.random.beta(1, 5)

        data.append((ame_score, bre_score, category))

    return data


class PropositionPredictor(nn.Module):
    """Simulates the LSTM proposition predictor from the paper.

    Takes feature scores and outputs proposition truth values
    for HasAmE and HasBrE.
    """

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )

    def forward(self, features):
        """
        Args:
            features: (batch, 2) — [ame_score, bre_score]

        Returns:
            (batch, 2) — [P(HasAmE), P(HasBrE)]
        """
        return self.net(features)


class ModalDialectReasoner(nn.Module):
    """Modal MLNN reasoner for 3-class dialect classification.

    Simulates a 3-world Kripke structure via thresholds:
    - □P if score > 0.9 (necessarily true in all worlds)
    - ♢P if score > 0.1 (possibly true in some world)

    Classification rules:
    - AmE: □HasAmE ∧ ¬♢HasBrE
    - BrE: □HasBrE ∧ ¬♢HasAmE
    - Neutral: (¬♢HasAmE ∧ ¬♢HasBrE) ∨ (♢HasAmE ∧ ♢HasBrE)
    """

    def __init__(self, box_threshold=0.9, diamond_threshold=0.1):
        super().__init__()
        self.box_threshold = box_threshold
        self.diamond_threshold = diamond_threshold

    def forward(self, prop_scores):
        """
        Args:
            prop_scores: (batch, 2) — [P(HasAmE), P(HasBrE)]

        Returns:
            (batch, 3) — truth values for [AmE, BrE, Neutral]
        """
        ame_scores = prop_scores[:, 0]
        bre_scores = prop_scores[:, 1]

        # Modal operators via thresholds
        box_ame = (ame_scores > self.box_threshold).float()
        box_bre = (bre_scores > self.box_threshold).float()
        dia_ame = (ame_scores > self.diamond_threshold).float()
        dia_bre = (bre_scores > self.diamond_threshold).float()

        # Classification logic
        is_ame = box_ame * F.negation(dia_bre)
        is_bre = box_bre * F.negation(dia_ame)

        # Neutral: no features OR mixed features
        no_features = F.negation(dia_ame) * F.negation(dia_bre)
        mixed = dia_ame * dia_bre
        is_neutral = F.disjunction(no_features, mixed)

        return torch.stack([is_ame, is_bre, is_neutral], dim=1)


def main():
    print("=" * 60)
    print("  Dialect Classification with Modal Logic")
    print("=" * 60)

    torch.manual_seed(42)
    data = generate_sentences()
    label_map = {"AmE": 0, "BrE": 1, "Neutral": 2}

    features = torch.tensor(
        [[d[0], d[1]] for d in data], dtype=torch.float32
    )
    labels = torch.tensor(
        [label_map[d[2]] for d in data], dtype=torch.long
    )

    # Split: train on AmE/BrE only, test on all 3
    train_mask = labels != 2  # exclude Neutral from training
    train_features = features[train_mask]
    train_labels = labels[train_mask]

    print(f"Data: {len(data)} sentences")
    print(f"  AmE: {(labels==0).sum()}, BrE: {(labels==1).sum()}, "
          f"Neutral: {(labels==2).sum()}")
    print(f"Training on {train_mask.sum()} AmE/BrE sentences only")

    # 1. Train proposition predictor (on binary task)
    predictor = PropositionPredictor()
    optimizer = optim.Adam(predictor.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Training targets: one-hot for the propositions
    train_targets = torch.zeros(len(train_features), 2)
    train_targets[train_labels == 0, 0] = 1.0  # HasAmE
    train_targets[train_labels == 1, 1] = 1.0  # HasBrE

    print(f"\n--- Training Proposition Predictor ---")
    for epoch in range(100):
        optimizer.zero_grad()
        preds = predictor(train_features)
        loss = loss_fn(preds, train_targets)
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.4f}")

    # 2. Apply Modal Reasoner to ALL data (including unseen Neutral)
    reasoner = ModalDialectReasoner()

    print(f"\n--- Evaluating Modal Reasoner on Full 3-Class Test ---")
    predictor.eval()
    with torch.no_grad():
        prop_scores = predictor(features)
        class_truths = reasoner(prop_scores)
        predictions = class_truths.argmax(dim=1)

    # Metrics per class
    class_names = ["AmE", "BrE", "Neutral"]
    print(f"\n{'Class':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    print("-" * 45)

    for c, name in enumerate(class_names):
        tp = ((predictions == c) & (labels == c)).sum().float()
        fp = ((predictions == c) & (labels != c)).sum().float()
        fn = ((predictions != c) & (labels == c)).sum().float()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        print(
            f"{name:<10} | {precision:<10.2f} | {recall:<10.2f} | {f1:<10.2f}"
        )

    overall_acc = (predictions == labels).float().mean()
    print(f"\nOverall Accuracy: {overall_acc:.1%}")

    # Highlight: Neutral class detection
    neutral_mask = labels == 2
    neutral_recall = (predictions[neutral_mask] == 2).float().mean()
    print(f"\nNeutral Class Recall: {neutral_recall:.1%}")
    print("(Trained ONLY on AmE/BrE, yet detects Neutral via modal logic!)")
    print("\nThis demonstrates axiomatic detection of the unknown:")
    print("  (¬♢HasAmE ∧ ¬♢HasBrE) → IsNeutral")


if __name__ == "__main__":
    main()
