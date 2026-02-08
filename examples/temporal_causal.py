"""
Temporal Causal Reasoning
=========================

Reproduces agent_debug_temporal_causal.ipynb.

Demonstrates temporal modal logic for root cause analysis in
event traces. Given a sequence of system events, the model
learns causal weights that identify which event types are
causally linked to system crashes via attention-based temporal
aggregation.

Uses:
  - torchmodal.functional.softmin (temporal aggregation)
  - torchmodal.nn.Necessity (causal necessity)
  - Learnable causality logits per event type
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torchmodal
from torchmodal import functional as F

SEED = 42

# Event types in a system monitoring scenario
EVENT_TYPES = [
    "CPU_Spike",
    "Memory_Leak",
    "Disk_IO",
    "Network_Timeout",
    "Cache_Miss",
    "Normal_Op",
]

# True causal structure: Memory_Leak (1) and Network_Timeout (3) cause crashes
TRUE_CAUSES = {1, 3}


class TemporalCausalMLNN(nn.Module):
    """Temporal causal reasoning model.

    Learns which event types are causally linked to system crashes
    by enforcing temporal necessity: □(cause → crash).
    """

    def __init__(self, num_event_types, embed_dim=16, tau=0.1):
        super().__init__()
        self.num_event_types = num_event_types
        self.tau = tau

        # Event type embeddings
        self.event_embed = nn.Embedding(num_event_types, embed_dim)

        # Temporal attention: which events in the trace are important
        self.attn_proj = nn.Linear(embed_dim, 1)

        # Causal weight per event type (learnable logits)
        self.causality_logits = nn.Parameter(
            torch.zeros(num_event_types)
        )

    @property
    def causality_probs(self):
        return torch.sigmoid(self.causality_logits)

    def forward(self, event_trace, is_crash):
        """
        Args:
            event_trace: (seq_len,) integer event type indices
            is_crash: scalar 0 or 1

        Returns:
            Scalar contradiction loss.
        """
        embeds = self.event_embed(event_trace)  # (seq_len, embed_dim)
        attn_logits = self.attn_proj(embeds).squeeze(-1)  # (seq_len,)
        attn_weights = torch.softmax(attn_logits / self.tau, dim=0)

        # Event types present in trace
        event_types_in_trace = event_trace.unique()
        causality = self.causality_probs[event_types_in_trace]

        # Weighted causal evidence from attention
        type_attention = torch.zeros(self.num_event_types)
        for i, etype in enumerate(event_trace):
            type_attention[etype] += attn_weights[i]

        # Explained cause: sum of (attention * causality) for events in trace
        explained = torch.sum(
            type_attention[event_types_in_trace] * causality
        )
        explained = torch.clamp(explained, 0.0, 1.0)

        # Modal axiom: □(crash → ∃cause)
        if is_crash > 0.5:
            # For crash traces: there MUST be an explained cause
            contradiction = torch.relu(1.0 - explained)
        else:
            # For non-crash traces: no causal explanation expected
            contradiction = torch.relu(explained)

        return contradiction


def generate_traces(num_traces=500, seq_len=10, seed=42):
    """Generate synthetic event traces with known causal structure."""
    np.random.seed(seed)
    traces = []

    for _ in range(num_traces):
        # Decide if this trace leads to a crash
        is_crash = np.random.random() < 0.4

        trace = []
        for _ in range(seq_len):
            if is_crash and np.random.random() < 0.3:
                # Insert a causal event (with some symptom dropout)
                cause = np.random.choice(list(TRUE_CAUSES))
                trace.append(cause)
            else:
                # Insert a non-causal event
                non_causes = [
                    i
                    for i in range(len(EVENT_TYPES))
                    if i not in TRUE_CAUSES
                ]
                trace.append(np.random.choice(non_causes))

        # Ensure crash traces actually contain at least one cause
        if is_crash and not any(t in TRUE_CAUSES for t in trace):
            trace[np.random.randint(seq_len)] = np.random.choice(
                list(TRUE_CAUSES)
            )

        traces.append(
            (torch.tensor(trace, dtype=torch.long), float(is_crash))
        )

    return traces


def main():
    print("=" * 60)
    print("  Temporal Causal Reasoning — Root Cause Analysis")
    print("=" * 60)
    print(f"Event types: {EVENT_TYPES}")
    print(f"True causes of crashes: {[EVENT_TYPES[i] for i in TRUE_CAUSES]}")

    traces = generate_traces()
    num_crash = sum(1 for _, c in traces if c > 0.5)
    print(f"Traces: {len(traces)} ({num_crash} crash, {len(traces)-num_crash} normal)")

    model = TemporalCausalMLNN(len(EVENT_TYPES))
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print(f"\n{'Epoch':<8} | {'Loss':<10} | Causality Weights")
    print("-" * 70)

    for epoch in range(301):
        total_loss = 0.0
        np.random.shuffle(traces)

        for trace, is_crash in traces:
            optimizer.zero_grad()
            loss = model(trace, is_crash)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 75 == 0:
            avg_loss = total_loss / len(traces)
            probs = model.causality_probs.detach().numpy()
            probs_str = ", ".join(
                f"{EVENT_TYPES[i][:6]}={probs[i]:.3f}"
                for i in range(len(EVENT_TYPES))
            )
            print(f"{epoch:<8} | {avg_loss:<10.4f} | {probs_str}")

    print("\n" + "=" * 60)
    print("Learned Causal Weights:")
    print("-" * 45)
    probs = model.causality_probs.detach().numpy()
    for i, name in enumerate(EVENT_TYPES):
        is_true_cause = "← TRUE CAUSE" if i in TRUE_CAUSES else ""
        bar = "█" * int(probs[i] * 30)
        print(f"  {name:<18} | {probs[i]:.3f} {bar} {is_true_cause}")

    print("\nThe model should assign higher causal weights to Memory_Leak")
    print("and Network_Timeout, matching the true causal structure.")


if __name__ == "__main__":
    main()
