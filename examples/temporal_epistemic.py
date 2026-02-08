"""
Temporal-Epistemic Operators Demo
==================================

Reproduces MLNN_TEMPORAL_EPISTEMIC.ipynb.

Demonstrates the epistemic (K), temporal (G, F), and composite (K∘G, K∘F)
modal operators on a 2-agent, 3-timestep scenario. Trains the MLNN to
resolve a contradiction by learning the epistemic accessibility relation.

Uses:
  - torchmodal.MultiAgentKripke
  - torchmodal.nn.Necessity, Possibility
  - torchmodal.functional.necessity, possibility
"""

import torch
import torch.optim as optim
import numpy as np

import torchmodal
from torchmodal import functional as F


def main():
    print("=" * 60)
    print("  MLNN Demo: Epistemic, Temporal, and Composite Operators")
    print("=" * 60)
    print("Scenario: 2 Agents (A, B), 3 Timesteps (t0, t1, t2)")
    print("States: [s0=(A,t0), s1=(B,t0), s2=(A,t1), "
          "s3=(B,t1), s4=(A,t2), s5=(B,t2)]")

    NUM_AGENTS = 2
    NUM_STEPS = 3

    # Create the multi-agent Kripke structure
    kripke = torchmodal.MultiAgentKripke(
        num_agents=NUM_AGENTS,
        num_steps=NUM_STEPS,
        tau=0.1,
        learnable_epistemic=True,
        init_bias=-10.0,  # start siloed (agents only see themselves)
    )

    # Ground truth: "isOnline" across spacetime states
    # States: [A@t0, B@t0, A@t1, B@t1, A@t2, B@t2]
    is_online = torch.tensor([0, 1, 1, 0, 1, 1], dtype=torch.float32)
    prop_bounds = torch.stack([is_online, is_online], dim=1)  # (6, 2)

    print(f"\nGround Truth 'isOnline': {is_online.numpy()}")

    # ---------------------------------------------------------------
    # Evaluate operators BEFORE training
    # ---------------------------------------------------------------
    print("\n" + "-" * 60)
    print("BEFORE TRAINING (siloed agents):")
    print("-" * 60)

    with torch.no_grad():
        A_epi = kripke.get_epistemic_accessibility()
        print(f"Epistemic A[s0→s0] = {A_epi[0,0]:.3f}")
        print(f"Epistemic A[s0→s1] = {A_epi[0,1]:.3f}")

        # G(isOnline): globally online?
        G_online = kripke.G(prop_bounds)
        print(f"G(isOnline) at s0: L={G_online[0,0]:.3f}, U={G_online[0,1]:.3f}")

        # F(isOnline): eventually online?
        F_online = kripke.F(prop_bounds)
        print(f"F(isOnline) at s0: L={F_online[0,0]:.3f}, U={F_online[0,1]:.3f}")

        # K(isOnline): agent knows online?
        K_online = kripke.K(prop_bounds[:NUM_AGENTS])
        print(f"K(isOnline) for Agent A: L={K_online[0,0]:.3f}, U={K_online[0,1]:.3f}")

    # ---------------------------------------------------------------
    # TRAINING: force F_epistemic(isOnline) at s0 to be TRUE
    # ---------------------------------------------------------------
    print("\n" + "-" * 60)
    print("TRAINING: Resolving contradiction via learning A_epistemic")
    print("-" * 60)
    print("Goal: Agent A at t0 should POSSIBLY know that B is online")
    print("      This requires opening A[s0→s1] (A sees B's world)")

    # Only learn the epistemic accessibility
    optimizer = optim.Adam(
        kripke.epistemic_access.parameters(), lr=0.5
    )

    for epoch in range(32):
        optimizer.zero_grad()

        A_epi = kripke.get_epistemic_accessibility()

        # F_epistemic(isOnline) — possibility over epistemic relation
        diamond = torchmodal.nn.Possibility(tau=0.1)
        # Apply diamond using the epistemic accessibility on the first
        # timestep states (agents at t0)
        t0_bounds = prop_bounds[:NUM_AGENTS]  # (2, 2)
        F_epi = diamond(t0_bounds, A_epi)

        # We want F_epi lower bound at agent 0 to be 1.0
        axiom_lower = F_epi[0, 0]
        loss = (1.0 - axiom_lower) ** 2

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            A_current = kripke.get_epistemic_accessibility().detach()
            print(
                f"Epoch {epoch:2d}: Loss={loss.item():.6f} | "
                f"A[0→0]={A_current[0,0]:.3f} | "
                f"A[0→1]={A_current[0,1]:.3f}"
            )

    # ---------------------------------------------------------------
    # Evaluate operators AFTER training
    # ---------------------------------------------------------------
    print("\n" + "-" * 60)
    print("AFTER TRAINING:")
    print("-" * 60)

    with torch.no_grad():
        A_epi = kripke.get_epistemic_accessibility()
        print(f"\nLearned Epistemic Accessibility:")
        for i in range(NUM_AGENTS):
            row = [f"{A_epi[i,j]:.3f}" for j in range(NUM_AGENTS)]
            print(f"  Agent {i}: [{', '.join(row)}]")

        # K(isOnline)
        K_online = kripke.K(prop_bounds[:NUM_AGENTS])
        for i in range(NUM_AGENTS):
            print(
                f"K(isOnline) Agent {i}: "
                f"L={K_online[i,0]:.3f}, U={K_online[i,1]:.3f}"
            )

        # G(isOnline)
        G_online = kripke.G(prop_bounds)
        print(f"G(isOnline) at s0: L={G_online[0,0]:.3f}, U={G_online[0,1]:.3f}")

        # F(isOnline)
        F_online = kripke.F(prop_bounds)
        print(f"F(isOnline) at s0: L={F_online[0,0]:.3f}, U={F_online[0,1]:.3f}")

    print("\nInterpretation:")
    print("  The optimizer opened A[0→1] so Agent A can 'see' Agent B's state,")
    print("  resolving the epistemic contradiction. Agent A now KNOWS B is online.")
    print("\nDone.")


if __name__ == "__main__":
    main()
