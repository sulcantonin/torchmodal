"""
Sudoku as a Multi-World Constraint Satisfaction Problem
=======================================================

Reproduces the Sudoku experiment from Section 5.4 of the paper.

Treats a 4x4 Sudoku grid as a Kripke model M = ⟨W, R, V⟩ with
|W| = 16 worlds (cells). The accessibility relation R is fixed
(cells in same row/col/box are accessible). Learning is driven
purely by constraint losses with temperature annealing.

Uses a 4x4 grid (instead of 9x9 AI Escargot) for fast demonstration.

Uses:
  - torchmodal.nn.FixedAccessibility
  - torchmodal.CrystallizationLoss
  - torchmodal.anneal_temperature
"""

import torch
import torch.optim as optim
import numpy as np

import torchmodal

BLOCK_SIZE = 2  # 2x2 blocks → 4x4 grid
GRID_SIZE = BLOCK_SIZE * BLOCK_SIZE  # 4
NUM_WORLDS = GRID_SIZE * GRID_SIZE  # 16 cells
NUM_DIGITS = GRID_SIZE  # 4 digits


def build_4x4_accessibility():
    """Build accessibility for a 4x4 Sudoku grid."""
    n = GRID_SIZE
    total = n * n
    A = torch.zeros(total, total)
    for i in range(total):
        ri, ci = divmod(i, n)
        bi_r, bi_c = ri // BLOCK_SIZE, ci // BLOCK_SIZE
        for j in range(total):
            if i == j:
                continue
            rj, cj = divmod(j, n)
            bj_r, bj_c = rj // BLOCK_SIZE, cj // BLOCK_SIZE
            if ri == rj or ci == cj or (bi_r == bj_r and bi_c == bj_c):
                A[i, j] = 1.0
    return A


def main():
    print("=" * 60)
    print("  Sudoku as Multi-World CSP (4x4 Grid)")
    print("=" * 60)

    torch.manual_seed(42)

    # Fixed accessibility
    R = build_4x4_accessibility()

    # Puzzle: 4x4 Sudoku (0 = unknown)
    # Solution:
    # 1 2 | 3 4
    # 3 4 | 1 2
    # ----+----
    # 2 1 | 4 3
    # 4 3 | 2 1
    puzzle = [
        [1, 0, 0, 4],
        [0, 4, 1, 0],
        [0, 1, 4, 0],
        [4, 0, 0, 1],
    ]

    # Identify given vs free cells
    given_cells = {}
    free_cells = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell = r * GRID_SIZE + c
            if puzzle[r][c] > 0:
                given_cells[cell] = puzzle[r][c] - 1
            else:
                free_cells.append(cell)

    # Only learn logits for FREE cells
    free_logits = torch.nn.Parameter(torch.randn(len(free_cells), NUM_DIGITS) * 0.5)

    crystal_loss_fn = torchmodal.CrystallizationLoss()
    optimizer = optim.Adam([free_logits], lr=0.3)
    TOTAL_EPOCHS = 500

    print(f"\nGiven puzzle ({len(given_cells)} given, {len(free_cells)} to solve):")
    for r in range(GRID_SIZE):
        row_str = ""
        for c in range(GRID_SIZE):
            v = puzzle[r][c]
            row_str += f" {v if v > 0 else '.'}"
            if c == BLOCK_SIZE - 1 and c < GRID_SIZE - 1:
                row_str += " |"
        print(f"  {row_str}")
        if r == BLOCK_SIZE - 1 and r < GRID_SIZE - 1:
            print(f"  {'--' * GRID_SIZE}+{'--' * GRID_SIZE}")

    print(f"\nTraining ({TOTAL_EPOCHS} epochs)...")
    print(f"{'Epoch':<8} | {'Loss':<10} | {'Solved'}")
    print("-" * 35)

    for epoch in range(TOTAL_EPOCHS):
        optimizer.zero_grad()

        # Build full probability matrix
        # Given cells: one-hot. Free cells: softmax of learnable logits.
        all_probs = torch.zeros(NUM_WORLDS, NUM_DIGITS)

        # Given cells (fixed one-hot, no grad)
        for cell, digit in given_cells.items():
            all_probs[cell, digit] = 1.0

        # Free cells (learnable via softmax)
        free_probs = torch.softmax(free_logits, dim=-1)
        for idx, cell in enumerate(free_cells):
            all_probs[cell] = free_probs[idx]

        total_loss = torch.tensor(0.0)

        # Axiom 1: No same digit in accessible cells (modal contradiction)
        for d in range(NUM_DIGITS):
            p_d = all_probs[:, d]
            # Pairwise conflict: p_d[w] * p_d[w'] for accessible pairs
            conflict = torch.outer(p_d, p_d) * R
            total_loss = total_loss + conflict.sum()

        # Axiom 2: Row, column, block constraints
        probs_grid = all_probs.view(GRID_SIZE, GRID_SIZE, NUM_DIGITS)
        for d in range(NUM_DIGITS):
            # Each digit exactly once per row
            row_sums = probs_grid[:, :, d].sum(dim=1)
            total_loss = total_loss + ((row_sums - 1.0) ** 2).sum()
            # Each digit exactly once per column
            col_sums = probs_grid[:, :, d].sum(dim=0)
            total_loss = total_loss + ((col_sums - 1.0) ** 2).sum()
            # Each digit exactly once per block
            for br in range(BLOCK_SIZE):
                for bc in range(BLOCK_SIZE):
                    block = probs_grid[
                        br * BLOCK_SIZE:(br + 1) * BLOCK_SIZE,
                        bc * BLOCK_SIZE:(bc + 1) * BLOCK_SIZE, d
                    ]
                    total_loss = total_loss + (block.sum() - 1.0) ** 2

        # Axiom 3: Crystallization — push toward crisp assignment
        crystal_w = min(1.0, epoch / (TOTAL_EPOCHS * 0.3))
        total_loss = total_loss + crystal_w * crystal_loss_fn(free_probs)

        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == TOTAL_EPOCHS - 1:
            with torch.no_grad():
                fp = torch.softmax(free_logits, dim=-1)
                n_crisp = (fp.max(dim=-1).values > 0.9).sum().item()
                print(
                    f"{epoch:<8} | {total_loss.item():<10.4f} | "
                    f"{n_crisp + len(given_cells)}/{NUM_WORLDS}"
                )

    # Display solution
    print(f"\nSolved grid:")
    with torch.no_grad():
        all_probs_final = torch.zeros(NUM_WORLDS, NUM_DIGITS)
        for cell, digit in given_cells.items():
            all_probs_final[cell, digit] = 1.0
        fp = torch.softmax(free_logits, dim=-1)
        for idx, cell in enumerate(free_cells):
            all_probs_final[cell] = fp[idx]

    assignments = all_probs_final.argmax(dim=-1) + 1

    for r in range(GRID_SIZE):
        row_str = ""
        for c in range(GRID_SIZE):
            cell = r * GRID_SIZE + c
            digit = assignments[cell].item()
            given = cell in given_cells
            marker = " " if given else "*"
            row_str += f" {digit}{marker}"
            if c == BLOCK_SIZE - 1 and c < GRID_SIZE - 1:
                row_str += " |"
        print(f"  {row_str}")
        if r == BLOCK_SIZE - 1 and r < GRID_SIZE - 1:
            print(f"  {'---' * GRID_SIZE}+{'---' * GRID_SIZE}")

    print("\n  (* = solved by MLNN, space = given)")

    # Verify
    grid = assignments.reshape(GRID_SIZE, GRID_SIZE).numpy()
    valid = True
    for i in range(GRID_SIZE):
        if len(set(grid[i, :])) != GRID_SIZE:
            valid = False
        if len(set(grid[:, i])) != GRID_SIZE:
            valid = False
    for br in range(BLOCK_SIZE):
        for bc in range(BLOCK_SIZE):
            block = grid[
                br * BLOCK_SIZE:(br + 1) * BLOCK_SIZE,
                bc * BLOCK_SIZE:(bc + 1) * BLOCK_SIZE,
            ]
            if len(set(block.flatten())) != GRID_SIZE:
                valid = False

    print(f"\n  Valid solution: {'YES' if valid else 'NO'}")


if __name__ == "__main__":
    main()
