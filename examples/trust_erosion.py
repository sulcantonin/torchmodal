"""
Combined Trust Erosion
======================

Reproduces agent_debug_combined_trust_erosion.ipynb.

Demonstrates combined modal logic for trust erosion using temporal
(retrospective reasoning), epistemic (knowledge updates), and deontic
(violations vs lies) logic. When proof emerges later, earlier ambiguous
events are re-evaluated, and lies cause larger trust drops than violations.

Uses:
  - torchmodal.nn.LearnableAccessibility
  - torchmodal.functional.implication, conjunction
  - Learnable trust logit optimized via contradiction minimization
"""

import torch
import torch.optim as optim

import torchmodal
from torchmodal import functional as F

EVENTS = {
    0: "Contract_Signed",
    1: "Latency_Spike",        # Ambiguous Event
    2: "Explanation: 'Noise'",  # The Lie
    3: "Normal_Ops",
    4: "LOGS_FOUND(Mining)",    # The Smoking Gun
    5: "Sanction",
}


class NeurosymbolicMonitor(torch.nn.Module):
    """Trust monitor using modal logic over event history.

    Trust is a learnable logit, and contradiction loss is computed
    from temporal retrospection over the event timeline.
    """

    def __init__(self):
        super().__init__()
        # Trust logit: high initial trust (3.0 → sigmoid ≈ 0.95)
        self.trust_logit = torch.nn.Parameter(torch.tensor([3.0]))
        self.history = []

    def log_event(self, t, event_id):
        self.history.append({"t": t, "id": event_id})

    def forward(self):
        contradiction = torch.zeros(1)
        trust = torch.sigmoid(self.trust_logit)

        # RETROSPECTIVE SCAN (Temporal Logic)
        # Check if proof (Event 4) exists in history
        has_proof = any(e["id"] == 4 for e in self.history)

        for record in self.history:
            eid = record["id"]

            # If we have proof, the "Latency Spike" (1) retroactively
            # becomes a violation (temporal reinterpretation)
            is_violation = eid == 1 and has_proof

            # If we have proof, the "Explanation" (2) retroactively
            # becomes a LIE (deontic: claimed 'Noise', reality was 'Mining')
            is_lie = eid == 2 and has_proof

            # Axiom 1: Competence (faults hurt trust slightly)
            if is_violation:
                # □(violation → ¬trust): contradiction with trust
                contradiction = contradiction + F.conjunction(
                    trust, torch.tensor([0.2])
                )

            # Axiom 2: Veracity (lies hurt trust MASSIVELY)
            if is_lie:
                # The "Double Fault": the lie hurts more than the crime
                # □(lie → ¬¬trust): strong contradiction
                contradiction = contradiction + F.conjunction(
                    trust, torch.tensor([1.5])
                )

        return contradiction


def main():
    print("=" * 70)
    print("  Combined Trust Erosion: Temporal + Epistemic + Deontic Logic")
    print("=" * 70)

    model = NeurosymbolicMonitor()
    optimizer = optim.Adam([model.trust_logit], lr=0.1)

    print(
        f"{'Time':<5} | {'Event':<25} | {'Trust':<8} | "
        f"{'Interpretation'}"
    )
    print("-" * 75)

    timeline = [
        (1, 0),   # Contract Signed
        (5, 1),   # Latency Spike (ambiguous)
        (8, 2),   # Explanation: "Noise" (the lie)
        (10, 3),  # Normal Operations (calm before the storm)
        (15, 4),  # LOGS FOUND: Mining (proof!)
        (16, 5),  # Sanction
    ]

    for t, eid in timeline:
        model.log_event(t, eid)

        # Reasoning loop: model reflects on ALL history given the new event
        for _ in range(15):
            optimizer.zero_grad()
            loss = model()
            if loss.item() > 0.0:
                loss.backward()
                optimizer.step()

        trust = torch.sigmoid(model.trust_logit).item()

        # Interpret
        if eid == 0:
            status = "Stable (initial trust)"
        elif eid == 1:
            status = "Suspicious (ambiguous event)"
        elif eid == 2:
            status = "Recovering (accepted excuse)"
        elif eid == 3:
            status = "Normal operations"
        elif eid == 4:
            status = "CRITICAL: Retroactive lie detected!"
        elif eid == 5:
            status = "COLLAPSED" if trust < 0.1 else "Low trust"
        else:
            status = ""

        print(f"t={t:<4} | {EVENTS[eid]:<25} | {trust:.4f}  | {status}")

    print("\n" + "=" * 70)
    print("Key Insight:")
    print("  At t=15, the proof (LOGS_FOUND) causes RETROACTIVE reinterpretation:")
    print("  - The Latency Spike (t=5) is now recognized as a VIOLATION")
    print("  - The Explanation (t=8) is now recognized as a LIE")
    print("  - The lie causes a MASSIVE trust drop (Veracity axiom)")
    print("  This demonstrates temporal + deontic modal reasoning.")
    print("=" * 70)


if __name__ == "__main__":
    main()
