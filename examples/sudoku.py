r"""
Sudoku as Differentiable Modal Logic (9x9)
==========================================

Reproduces the Sudoku experiment from Section 5.4 of the paper.

Treats a standard 9x9 Sudoku grid as a Kripke model M = (W, R, V):
  |W| = 81 worlds (one per cell)
  R   = fixed accessibility (cells in same row/col/box are mutually
        accessible)
  V_theta = learnable truth values for 9 atomic propositions p_1..p_9
            ("digit k is assigned to this cell")

The core modal axiom is:
  /\__{k=1}^{9}  (p_k -> ~diamond p_k)
  "If digit k is assigned to this cell, digit k is not possibly
   assigned to any accessible cell."

This is computed with the actual modal operators from torchmodal:
  - F.possibility(p_d, R, tau)  -- differentiable diamond
  - F.negation(diamond_p_d)     -- negation
  - F.implication(p_d, neg_diamond_p_d)  -- the full axiom

The axiom satisfaction level drives the loss: values below 1
indicate constraint violations that the optimiser resolves.

After solving, a [L,U] bounds demo shows that the operators natively
support truth-bound intervals and that ContradictionLoss detects
inconsistencies in the bound lattice.

Uses:
  - torchmodal.functional.possibility / necessity
  - torchmodal.functional.negation / implication
  - torchmodal.CrystallizationLoss
  - torchmodal.ContradictionLoss
"""

# Make sure the local development torchmodal (../torchmodal) shadows any
# PyPI-installed release, which lacks newer APIs and fails silently.
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torchmodal
from torchmodal import functional as F

BLOCK_SIZE = 3
GRID_SIZE = 9
NUM_WORLDS = GRID_SIZE * GRID_SIZE  # 81
NUM_DIGITS = GRID_SIZE  # 9

# Classic Wikipedia Sudoku (30 givens, medium difficulty)
PUZZLE = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

SOLUTION = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]


def build_accessibility():
    """Fixed accessibility: cells sharing a row, column, or 3x3 box."""
    A = torch.zeros(NUM_WORLDS, NUM_WORLDS)
    for i in range(NUM_WORLDS):
        ri, ci = divmod(i, GRID_SIZE)
        bi_r, bi_c = ri // BLOCK_SIZE, ci // BLOCK_SIZE
        for j in range(NUM_WORLDS):
            if i == j:
                continue
            rj, cj = divmod(j, GRID_SIZE)
            bj_r, bj_c = rj // BLOCK_SIZE, cj // BLOCK_SIZE
            if ri == rj or ci == cj or (bi_r == bj_r and bi_c == bj_c):
                A[i, j] = 1.0
    return A


def print_grid(grid, puzzle):
    for r in range(GRID_SIZE):
        row_str = "  "
        for c in range(GRID_SIZE):
            val = grid[r][c]
            if puzzle[r][c] > 0:
                row_str += f" {val} "
            else:
                row_str += f" {val}*" if val > 0 else " . "
            if c in (2, 5):
                row_str += "|"
        print(row_str)
        if r in (2, 5):
            print("  " + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10)


def check_solution(grid):
    for i in range(GRID_SIZE):
        if len(set(grid[i])) != GRID_SIZE:
            return False
        if len(set(grid[r][i] for r in range(GRID_SIZE))) != GRID_SIZE:
            return False
    for br in range(BLOCK_SIZE):
        for bc in range(BLOCK_SIZE):
            block = [
                grid[br * 3 + dr][bc * 3 + dc]
                for dr in range(3) for dc in range(3)
            ]
            if len(set(block)) != GRID_SIZE:
                return False
    return True


def solve(seed=42, verbose=True):
    """Solve the 9x9 Sudoku using differentiable modal logic.

    Returns (grid, probs) where grid is a list-of-lists and probs is
    the final (81, 9) probability tensor.
    """
    torch.manual_seed(seed)
    R = build_accessibility()

    given_cells = {}
    free_cells = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell = r * GRID_SIZE + c
            if PUZZLE[r][c] > 0:
                given_cells[cell] = PUZZLE[r][c] - 1
            else:
                free_cells.append(cell)

    free_logits = nn.Parameter(torch.randn(len(free_cells), NUM_DIGITS) * 0.1)
    crystal_loss_fn = torchmodal.CrystallizationLoss()
    optimizer = optim.Adam([free_logits], lr=0.15)

    TOTAL_EPOCHS = 2000
    TAU_START, TAU_END = 0.5, 0.01

    for epoch in range(TOTAL_EPOCHS):
        optimizer.zero_grad()

        progress = epoch / TOTAL_EPOCHS
        tau = TAU_START * (TAU_END / TAU_START) ** progress

        # Assemble truth-value tensor for all 81 worlds
        all_probs = torch.zeros(NUM_WORLDS, NUM_DIGITS)
        for cell, digit in given_cells.items():
            all_probs[cell, digit] = 1.0
        free_probs = torch.softmax(free_logits, dim=-1)
        for idx, cell in enumerate(free_cells):
            all_probs[cell] = free_probs[idx]

        # === Core modal axiom: p_d -> ~diamond p_d ===
        axiom_loss = torch.tensor(0.0)
        for d in range(NUM_DIGITS):
            p_d = all_probs[:, d]                      # (81,)
            diamond_d = F.possibility(p_d, R, tau=tau)  # diamond p_d
            neg_dia_d = F.negation(diamond_d)           # ~diamond p_d
            axiom_d = F.implication(p_d, neg_dia_d)     # p_d -> ~diamond p_d
            axiom_loss = axiom_loss + (1.0 - axiom_d).clamp(min=0).sum()

        # Crystallization: anneal toward crisp 0/1 assignments
        crys_w = min(1.0, epoch / (TOTAL_EPOCHS * 0.3))
        crys_loss = crys_w * crystal_loss_fn(free_probs)

        total_loss = axiom_loss + crys_loss
        total_loss.backward()
        optimizer.step()

        if verbose and (epoch % 400 == 0 or epoch == TOTAL_EPOCHS - 1):
            with torch.no_grad():
                fp = torch.softmax(free_logits, dim=-1)
                n_crisp = (fp.max(dim=-1).values > 0.9).sum().item()
            print(
                f"  {epoch:<6} | loss {total_loss.item():<9.4f} "
                f"| tau {tau:.3f} "
                f"| crisp {n_crisp + len(given_cells)}/{NUM_WORLDS}"
            )

    # Extract final grid
    with torch.no_grad():
        all_final = torch.zeros(NUM_WORLDS, NUM_DIGITS)
        for cell, digit in given_cells.items():
            all_final[cell, digit] = 1.0
        fp = torch.softmax(free_logits, dim=-1)
        for idx, cell in enumerate(free_cells):
            all_final[cell] = fp[idx]

    assignments = (all_final.argmax(dim=-1) + 1).tolist()
    grid = [assignments[r * GRID_SIZE:(r + 1) * GRID_SIZE]
            for r in range(GRID_SIZE)]
    return grid, all_final


def main():
    print("=" * 60)
    print("  Sudoku as Differentiable Modal Logic (9x9)")
    print("=" * 60)
    print()
    print("  Modal axiom:  p_k -> ~(diamond p_k)")
    print("  Operators:    F.possibility, F.negation, F.implication")
    print("  Loss:         axiom violation + crystallization")

    n_given = sum(1 for row in PUZZLE for v in row if v > 0)
    print(f"\n  {NUM_WORLDS} worlds, {n_given} given, {NUM_WORLDS - n_given} to solve")

    print("\n  Puzzle:")
    print_grid(PUZZLE, PUZZLE)

    # Try multiple seeds for reliability (parallels paper's 512 instances)
    best_grid, best_probs = None, None
    for seed in range(5):
        print(f"\n  --- Attempt {seed + 1} (seed={seed}) ---")
        grid, probs = solve(seed=seed)
        if check_solution(grid):
            best_grid, best_probs = grid, probs
            print("  >> Valid solution found!")
            break
        n_ok = sum(
            1 for r in range(GRID_SIZE) for c in range(GRID_SIZE)
            if grid[r][c] == SOLUTION[r][c]
        )
        print(f"  >> {n_ok}/{NUM_WORLDS} cells correct, retrying...")

    if best_grid is None:
        best_grid, best_probs = grid, probs

    print("\n  Solved grid:")
    print_grid(best_grid, PUZZLE)
    print("  (* = solved by MLNN)")
    valid = check_solution(best_grid)
    print(f"\n  Valid solution: {'YES' if valid else 'NO'}")

    # ====================================================================
    #  [L, U] Bounds Demonstration
    # ====================================================================
    print()
    print("=" * 60)
    print("  [L, U] Bounds Verification")
    print("=" * 60)
    print()
    print("  The modal operators natively accept (|W|, 2) tensors with")
    print("  [L, U] truth-bound columns.  Below we feed bounds through")
    print("  F.necessity and verify ContradictionLoss = 0 on a valid")
    print("  solution (no L > U crossings).")
    print()

    R = build_accessibility()
    contra_loss = torchmodal.ContradictionLoss()

    with torch.no_grad():
        total_contra = 0.0
        for d in range(NUM_DIGITS):
            L_d = best_probs[:, d]
            U_d = (best_probs[:, d] + 0.05).clamp(max=1.0)

            # For given cells: no uncertainty
            for cell, digit in {
                r * GRID_SIZE + c: PUZZLE[r][c] - 1
                for r in range(GRID_SIZE) for c in range(GRID_SIZE)
                if PUZZLE[r][c] > 0
            }.items():
                if digit == d:
                    U_d[cell] = 1.0
                else:
                    U_d[cell] = 0.0

            neg_bounds = torch.stack([1 - U_d, 1 - L_d], dim=-1)
            box_neg = F.necessity(neg_bounds, R, tau=0.1)  # (81, 2)
            total_contra += contra_loss(box_neg).item()

        print(f"  Total ContradictionLoss across 9 digits: {total_contra:.6f}")
        verdict = "consistent" if total_contra < 0.01 else "contradictions detected"
        print(f"  ({verdict})")

    # Show a few cells
    print()
    with torch.no_grad():
        d = 0  # digit 1
        L_d = best_probs[:, d]
        U_d = (best_probs[:, d] + 0.05).clamp(max=1.0)
        neg_bounds = torch.stack([1 - U_d, 1 - L_d], dim=-1)
        box_neg = F.necessity(neg_bounds, R, tau=0.1)

        print("  Sample bounds for digit 1 -- Box(~p_1):")
        for cell in [0, 1, 2, 9, 10]:
            r, c = divmod(cell, GRID_SIZE)
            val = best_grid[r][c]
            lb, ub = box_neg[cell, 0].item(), box_neg[cell, 1].item()
            print(
                f"    Cell ({r},{c}) digit={val}: "
                f"p_1=[{L_d[cell]:.3f},{U_d[cell]:.3f}]  "
                f"Box(~p_1)=[{lb:.4f},{ub:.4f}]  "
                f"L<=U: {lb <= ub + 1e-7}"
            )


if __name__ == "__main__":
    main()
