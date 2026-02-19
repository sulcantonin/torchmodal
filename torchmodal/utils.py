"""
torchmodal.utils
~~~~~~~~~~~~~~~~

Utility functions for MLNN training and evaluation.

Includes temperature annealing schedules, accessibility matrix helpers,
and visualization utilities.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

__all__ = [
    "anneal_temperature",
    "build_ring_accessibility",
    "build_sudoku_accessibility",
    "build_grid_accessibility",
    "decode_one_hot",
    "bounds_to_labels",
]


def anneal_temperature(
    epoch: int,
    total_epochs: int,
    tau_start: float = 2.0,
    tau_end: float = 0.1,
    schedule: str = "linear",
) -> float:
    """Compute annealed temperature for a given epoch.

    Temperature annealing drives the "phase transition" observed in
    satisfiability experiments (Section 5.4): high temperature allows
    exploration, low temperature forces crisp assignments.

    Args:
        epoch: Current epoch (0-indexed).
        total_epochs: Total number of training epochs.
        tau_start: Starting (high) temperature. Default 2.0.
        tau_end: Final (low) temperature. Default 0.1.
        schedule: Annealing schedule — ``"linear"``, ``"cosine"``,
            or ``"exponential"``. Default ``"linear"``.

    Returns:
        Temperature value for the current epoch.
    """
    if total_epochs <= 1:
        return tau_end

    progress = min(epoch / (total_epochs - 1), 1.0)

    if schedule == "linear":
        tau = tau_start + (tau_end - tau_start) * progress
    elif schedule == "cosine":
        tau = tau_end + 0.5 * (tau_start - tau_end) * (
            1 + math.cos(math.pi * progress)
        )
    elif schedule == "exponential":
        log_start = math.log(tau_start)
        log_end = math.log(tau_end)
        tau = math.exp(log_start + (log_end - log_start) * progress)
    else:
        raise ValueError(
            f"Unknown schedule '{schedule}'. "
            f"Choose from: linear, cosine, exponential"
        )

    return max(tau, tau_end)


def build_ring_accessibility(
    num_worlds: int,
    bidirectional: bool = False,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Build a directed ring accessibility matrix.

    Used in the synthetic Diplomacy ring scalability test (Appendix G.3).
    Agent *i* can access agent *(i+1) mod N*.

    Args:
        num_worlds: Number of worlds (agents) in the ring.
        bidirectional: If ``True``, make edges bidirectional.
        device: Target device.

    Returns:
        Binary accessibility matrix ``(num_worlds, num_worlds)``.
    """
    idx = torch.arange(num_worlds, device=device)
    next_idx = (idx + 1) % num_worlds

    # Self-access
    A = torch.eye(num_worlds, device=device)
    # Forward ring edges: i -> (i+1) mod N
    A[idx, next_idx] = 1.0
    if bidirectional:
        A[next_idx, idx] = 1.0
    return A


def build_sudoku_accessibility(
    block_size: int = 3,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Build the accessibility matrix for a Sudoku puzzle.

    Two cells (worlds) are accessible to each other if they share
    the same row, column, or sub-grid block.

    Used in the CSP experiment (Section 5.4).

    Args:
        block_size: Size of the sub-grid (3 for standard 9x9 Sudoku).
        device: Target device.

    Returns:
        Binary accessibility matrix ``(n², n²)`` where ``n = block_size²``.
    """
    n = block_size * block_size  # grid size (e.g., 9)
    total = n * n  # total cells (e.g., 81)

    idx = torch.arange(total, device=device)
    rows = idx // n
    cols = idx % n
    block_rows = rows // block_size
    block_cols = cols // block_size

    # Broadcasting: compare all (i, j) pairs simultaneously
    same_row = rows.unsqueeze(1) == rows.unsqueeze(0)
    same_col = cols.unsqueeze(1) == cols.unsqueeze(0)
    same_block = (
        (block_rows.unsqueeze(1) == block_rows.unsqueeze(0))
        & (block_cols.unsqueeze(1) == block_cols.unsqueeze(0))
    )
    not_self = ~torch.eye(total, dtype=torch.bool, device=device)

    A = ((same_row | same_col | same_block) & not_self).float()
    return A


def build_grid_accessibility(
    rows: int,
    cols: int,
    connectivity: str = "4",
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Build a grid accessibility matrix.

    Useful for spatial reasoning where worlds are arranged in a grid.

    Args:
        rows: Number of rows.
        cols: Number of columns.
        connectivity: ``"4"`` for cardinal neighbors or ``"8"`` for
            including diagonals. Default ``"4"``.
        device: Target device.

    Returns:
        Binary accessibility matrix ``(rows*cols, rows*cols)``.
    """
    total = rows * cols
    A = torch.eye(total, device=device)  # self-access

    idx = torch.arange(total, device=device)
    r = idx // cols
    c = idx % cols

    # Cardinal neighbor offsets: (delta_row, delta_col)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == "8":
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dr, dc in offsets:
        nr = r + dr
        nc = c + dc
        valid = (nr >= 0) & (nr < rows) & (nc >= 0) & (nc < cols)
        nidx = nr * cols + nc
        A[idx[valid], nidx[valid]] = 1.0

    return A


def decode_one_hot(
    bounds: Tensor,
    threshold: float = 0.5,
) -> Tensor:
    """Decode one-hot truth bounds to class labels.

    For each world, returns the proposition index with the highest
    lower bound exceeding the threshold.

    Args:
        bounds: ``(num_worlds, num_propositions, 2)`` or
            ``(num_worlds, num_propositions)`` truth values.
        threshold: Minimum confidence threshold. Default 0.5.

    Returns:
        ``(num_worlds,)`` integer labels (-1 if no proposition exceeds
        the threshold).
    """
    if bounds.dim() == 3:
        values = bounds[..., 0]  # use lower bounds
    else:
        values = bounds

    max_vals, labels = values.max(dim=-1)
    labels[max_vals < threshold] = -1
    return labels


def bounds_to_labels(
    bounds: Tensor,
    threshold_necessary: float = 0.9,
    threshold_possible: float = 0.1,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert truth bounds to modal classification labels.

    Used in the dialect classification experiment (Section 5.2) to
    assign Necessary (□), Possible (♢), or Indeterminate labels.

    Args:
        bounds: ``(batch, 2)`` or ``(2,)`` truth bounds ``[L, U]``.
        threshold_necessary: Above this → □P. Default 0.9.
        threshold_possible: Above this → ♢P. Default 0.1.

    Returns:
        Tuple of ``(is_necessary, is_possible, is_indeterminate)``
        boolean tensors.
    """
    if bounds.dim() == 1:
        bounds = bounds.unsqueeze(0)

    L = bounds[..., 0]
    U = bounds[..., 1]
    score = (L + U) / 2.0

    is_necessary = score > threshold_necessary
    is_possible = score > threshold_possible
    is_indeterminate = ~is_possible

    return is_necessary, is_possible, is_indeterminate
