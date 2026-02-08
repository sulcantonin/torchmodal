"""
Axiom Ablation Study
====================

Reproduces the axiom ablation from MLNN_AxiomsLogicAblation.ipynb.

Tests how enforcing different modal logic axioms (Reflexivity T,
Transitivity 4, Symmetry B) affects the learned accessibility
relation when recovering a ground-truth ring structure.

Uses:
  - torchmodal.nn.LearnableAccessibility
  - torchmodal.functional.softmin, necessity
  - torchmodal.AxiomRegularization
  - torchmodal.build_ring_accessibility
"""

import torch
import torch.optim as optim
import numpy as np

import torchmodal
from torchmodal import functional as F

NUM_WORLDS = 10
EPOCHS = 200
LR = 0.01


def run_sweep(axiom_name, lambda_values):
    """Run a sweep over regularization strengths for a given axiom."""
    # Ground truth: directed ring
    A_gt = torchmodal.build_ring_accessibility(NUM_WORLDS, bidirectional=False)

    print(f"\n{'='*60}")
    print(f"  Axiom Sweep: {axiom_name}")
    print(f"{'='*60}")
    print(f"{'Lambda':<10} | {'Axiom Err':<12} | {'Task MSE':<12}")
    print("-" * 40)

    for lam in lambda_values:
        torch.manual_seed(42)

        # Learnable accessibility
        access = torchmodal.nn.LearnableAccessibility(
            NUM_WORLDS, init_bias=-2.0, reflexive=False
        )
        # Learnable proposition bounds
        L_p = torch.nn.Parameter(torch.rand(NUM_WORLDS) * 0.3)
        U_p = torch.nn.Parameter(0.7 + torch.rand(NUM_WORLDS) * 0.3)

        # Build the regularizer
        reg_kwargs = {"reflexivity": 0.0, "transitivity": 0.0, "symmetry": 0.0}
        if axiom_name == "Reflexivity (T)":
            reg_kwargs["reflexivity"] = lam
        elif axiom_name == "Transitivity (4)":
            reg_kwargs["transitivity"] = lam
        elif axiom_name == "Symmetry (B)":
            reg_kwargs["symmetry"] = lam

        axiom_reg = torchmodal.AxiomRegularization(**reg_kwargs)

        params = list(access.parameters()) + [L_p, U_p]
        optimizer = optim.Adam(params, lr=LR)

        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            A = access()

            # Box operator: L_box = softmin((1 - A) + L_p) per world
            impl = (1.0 - A) + L_p.unsqueeze(0)
            L_box = F.softmin(impl, tau=0.1, dim=1)

            # Contradiction loss: L_box should not exceed U_p
            l_contra = torch.mean(torch.relu(L_box - U_p) ** 2)

            # Task loss: match ground truth ring
            l_task = torch.mean((A - A_gt) ** 2)

            # Axiom regularization
            l_axiom = axiom_reg(A)

            total = l_task + l_contra + l_axiom
            total.backward()
            optimizer.step()

            with torch.no_grad():
                L_p.clamp_(0, 1)
                U_p.clamp_(0, 1)

        # Evaluate
        A_final = access().detach()

        if axiom_name == "Reflexivity (T)":
            axiom_err = torch.mean(1.0 - torch.diag(A_final)).item()
        elif axiom_name == "Transitivity (4)":
            A_sq = torch.clamp(A_final @ A_final, 0, 1)
            axiom_err = torch.mean(torch.relu(A_sq - A_final)).item()
        elif axiom_name == "Symmetry (B)":
            axiom_err = torch.mean((A_final - A_final.t()).abs()).item()
        else:
            axiom_err = 0.0

        task_mse = torch.mean((A_final - A_gt) ** 2).item()

        print(f"{lam:<10.2f} | {axiom_err:<12.6f} | {task_mse:<12.6f}")


def main():
    print("Axiom Ablation Study (MLNN_AxiomsLogicAblation)")
    print("Recovering a ring structure with axiom regularization")

    lambdas = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0]

    run_sweep("Reflexivity (T)", lambdas)
    run_sweep("Transitivity (4)", lambdas)
    run_sweep("Symmetry (B)", lambdas)

    print("\n\nDone. Higher lambda enforces the axiom more strongly.")


if __name__ == "__main__":
    main()
