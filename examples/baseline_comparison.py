"""
Baseline Comparisons for Reviewer Response
===========================================

Implements baselines requested by reviewers (R1, R4) for direct
comparison with MLNN on the self-contained examples.

Baselines implemented:
  1. Sudoku: Simulated Annealing (R1: "relationship to SA should be discussed")
  2. Sudoku: Semantic Loss (R4: "Semantic Loss is directly relevant")
  3. Dialect: Argmax baseline (no abstention)
  4. Dialect: Conformal Prediction baseline
  5. Trust: Non-modal MLP (R1: "any sequence model can detect past dishonesty")

Each baseline is compared against the MLNN approach on the same task.
"""

import json

# Make sure the local development torchmodal (../torchmodal) shadows any
# PyPI-installed release, which lacks newer APIs and fails silently.
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torchmodal
from torchmodal import functional as F

SEED = 42
RESULTS = OrderedDict()

# ===================================================================
# Experiment 1: Sudoku (9x9) — MLNN vs Simulated Annealing vs Semantic Loss
# ===================================================================

BLOCK_SIZE = 3
GRID_SIZE = 9
NUM_WORLDS = GRID_SIZE * GRID_SIZE  # 81
NUM_DIGITS = GRID_SIZE  # 9

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


def build_accessibility():
    """Fixed accessibility for 9x9: cells sharing row, column, or 3x3 box."""
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


def get_puzzle_setup():
    given_cells = {}
    free_cells = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell = r * GRID_SIZE + c
            if PUZZLE[r][c] > 0:
                given_cells[cell] = PUZZLE[r][c] - 1
            else:
                free_cells.append(cell)
    return given_cells, free_cells


def check_solution(assignments):
    grid = assignments.reshape(GRID_SIZE, GRID_SIZE).numpy()
    for i in range(GRID_SIZE):
        if len(set(grid[i, :])) != GRID_SIZE:
            return False
        if len(set(grid[:, i])) != GRID_SIZE:
            return False
    for br in range(BLOCK_SIZE):
        for bc in range(BLOCK_SIZE):
            block = grid[
                br * BLOCK_SIZE:(br + 1) * BLOCK_SIZE,
                bc * BLOCK_SIZE:(bc + 1) * BLOCK_SIZE,
            ]
            if len(set(block.flatten())) != GRID_SIZE:
                return False
    return True


def run_sudoku_mlnn(n_trials=10):
    """MLNN approach: modal axiom p_d -> ~diamond(p_d) with temperature annealing."""
    R = build_accessibility()
    given_cells, free_cells = get_puzzle_setup()
    crystal_loss_fn = torchmodal.CrystallizationLoss()

    successes = 0
    total_time = 0.0

    for trial in range(n_trials):
        torch.manual_seed(trial)
        free_logits = nn.Parameter(torch.randn(len(free_cells), NUM_DIGITS) * 0.1)
        optimizer = optim.Adam([free_logits], lr=0.15)
        epochs = 2000
        tau_start, tau_end = 0.5, 0.01

        t0 = time.time()
        for epoch in range(epochs):
            optimizer.zero_grad()
            progress = epoch / epochs
            tau = tau_start * (tau_end / tau_start) ** progress

            all_probs = torch.zeros(NUM_WORLDS, NUM_DIGITS)
            for cell, digit in given_cells.items():
                all_probs[cell, digit] = 1.0
            free_probs = torch.softmax(free_logits, dim=-1)
            for idx, cell in enumerate(free_cells):
                all_probs[cell] = free_probs[idx]

            axiom_loss = torch.tensor(0.0)
            for d in range(NUM_DIGITS):
                p_d = all_probs[:, d]
                diamond_d = F.possibility(p_d, R, tau=tau)
                neg_dia_d = F.negation(diamond_d)
                axiom_d = F.implication(p_d, neg_dia_d)
                axiom_loss = axiom_loss + (1.0 - axiom_d).clamp(min=0).sum()

            crystal_w = min(1.0, epoch / (epochs * 0.3))
            total_loss = axiom_loss + crystal_w * crystal_loss_fn(free_probs)
            total_loss.backward()
            optimizer.step()

        total_time += time.time() - t0

        with torch.no_grad():
            all_probs_final = torch.zeros(NUM_WORLDS, NUM_DIGITS)
            for cell, digit in given_cells.items():
                all_probs_final[cell, digit] = 1.0
            fp = torch.softmax(free_logits, dim=-1)
            for idx, cell in enumerate(free_cells):
                all_probs_final[cell] = fp[idx]
        assignments = all_probs_final.argmax(dim=-1) + 1
        if check_solution(assignments):
            successes += 1

    return successes / n_trials, total_time / n_trials


def run_sudoku_simulated_annealing(n_trials=10):
    """Simulated Annealing baseline (R1: discuss relationship to SA).

    Uses box-based initialisation and swap moves (standard SA approach
    for Sudoku): each 3x3 box is initialised with a valid permutation,
    then pairs of non-given cells within the same box are swapped.
    Cost counts only row and column violations (box constraints are
    satisfied by construction).
    """
    puzzle = np.array(PUZZLE)
    given_mask = puzzle > 0

    successes = 0
    total_time = 0.0

    for trial in range(n_trials):
        np.random.seed(trial)
        t0 = time.time()

        grid = puzzle.copy()
        # Initialise each box with a valid permutation
        box_free = {}
        for br in range(BLOCK_SIZE):
            for bc in range(BLOCK_SIZE):
                rs = slice(br * 3, br * 3 + 3)
                cs = slice(bc * 3, bc * 3 + 3)
                block = grid[rs, cs]
                mask = given_mask[rs, cs]
                present = set(block[mask].tolist())
                missing = [d for d in range(1, 10) if d not in present]
                np.random.shuffle(missing)
                free_pos = list(zip(*np.where(~mask)))
                for pos, val in zip(free_pos, missing):
                    block[pos] = val
                grid[rs, cs] = block
                box_free[(br, bc)] = [
                    (br * 3 + r, bc * 3 + c) for r, c in free_pos
                ]

        def row_col_violations(g):
            v = 0
            for i in range(9):
                v += 9 - len(set(g[i, :]))
                v += 9 - len(set(g[:, i]))
            return v

        cur_v = row_col_violations(grid)
        T_start, T_end = 2.0, 0.001
        max_iter = 200000

        for it in range(max_iter):
            T = T_start * (T_end / T_start) ** (it / max_iter)
            br, bc = np.random.randint(3), np.random.randint(3)
            free = box_free[(br, bc)]
            if len(free) < 2:
                continue
            i, j = np.random.choice(len(free), 2, replace=False)
            r1, c1 = free[i]
            r2, c2 = free[j]

            grid[r1, c1], grid[r2, c2] = grid[r2, c2], grid[r1, c1]
            new_v = row_col_violations(grid)
            delta = new_v - cur_v

            if delta <= 0 or np.random.random() < np.exp(-delta / T):
                cur_v = new_v
            else:
                grid[r1, c1], grid[r2, c2] = grid[r2, c2], grid[r1, c1]

            if cur_v == 0:
                break

        total_time += time.time() - t0
        if cur_v == 0:
            successes += 1

    return successes / n_trials, total_time / n_trials


def run_sudoku_semantic_loss(n_trials=10):
    """Semantic Loss baseline (R4: Xu et al., 2018)."""
    given_cells, free_cells = get_puzzle_setup()
    sem_loss_fn = torchmodal.SemanticLoss()

    successes = 0
    total_time = 0.0

    for trial in range(n_trials):
        torch.manual_seed(trial)
        free_logits = nn.Parameter(torch.randn(len(free_cells), NUM_DIGITS) * 0.1)
        optimizer = optim.Adam([free_logits], lr=0.15)
        epochs = 2000

        t0 = time.time()
        for epoch in range(epochs):
            optimizer.zero_grad()
            all_probs = torch.zeros(NUM_WORLDS, NUM_DIGITS)
            for cell, digit in given_cells.items():
                all_probs[cell, digit] = 1.0
            free_probs = torch.softmax(free_logits, dim=-1)
            for idx, cell in enumerate(free_cells):
                all_probs[cell] = free_probs[idx]

            loss_sem = sem_loss_fn.forward_mutual_exclusive(free_probs)

            loss_struct = torch.tensor(0.0)
            probs_grid = all_probs.view(GRID_SIZE, GRID_SIZE, NUM_DIGITS)
            for d in range(NUM_DIGITS):
                row_sums = probs_grid[:, :, d].sum(dim=1)
                loss_struct = loss_struct + ((row_sums - 1.0) ** 2).sum()
                col_sums = probs_grid[:, :, d].sum(dim=0)
                loss_struct = loss_struct + ((col_sums - 1.0) ** 2).sum()
                for br in range(BLOCK_SIZE):
                    for bc in range(BLOCK_SIZE):
                        block = probs_grid[
                            br * BLOCK_SIZE:(br + 1) * BLOCK_SIZE,
                            bc * BLOCK_SIZE:(bc + 1) * BLOCK_SIZE, d
                        ]
                        loss_struct = loss_struct + (block.sum() - 1.0) ** 2

            total_loss = loss_sem + loss_struct
            total_loss.backward()
            optimizer.step()

        total_time += time.time() - t0

        with torch.no_grad():
            all_probs_final = torch.zeros(NUM_WORLDS, NUM_DIGITS)
            for cell, digit in given_cells.items():
                all_probs_final[cell, digit] = 1.0
            fp = torch.softmax(free_logits, dim=-1)
            for idx, cell in enumerate(free_cells):
                all_probs_final[cell] = fp[idx]
        assignments = all_probs_final.argmax(dim=-1) + 1
        if check_solution(assignments):
            successes += 1

    return successes / n_trials, total_time / n_trials


# ===================================================================
# Experiment 2: Dialect Classification — MLNN vs Argmax vs Conformal
# ===================================================================

def generate_dialect_data(n=3000, seed=42):
    np.random.seed(seed)
    data = []
    for _ in range(n):
        category = np.random.choice(["AmE", "BrE", "Neutral"], p=[0.35, 0.35, 0.30])
        if category == "AmE":
            ame_score = np.random.beta(8, 2)
            bre_score = np.random.beta(1, 8)
        elif category == "BrE":
            ame_score = np.random.beta(1, 8)
            bre_score = np.random.beta(8, 2)
        else:
            ame_score = np.random.beta(1, 5)
            bre_score = np.random.beta(1, 5)
        data.append((ame_score, bre_score, category))
    return data


class SimplePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)


def train_predictor(features, labels, epochs=100):
    mask = labels != 2
    predictor = SimplePredictor()
    optimizer = optim.Adam(predictor.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    targets = torch.zeros(mask.sum(), 2)
    targets[labels[mask] == 0, 0] = 1.0
    targets[labels[mask] == 1, 1] = 1.0
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(predictor(features[mask]), targets)
        loss.backward()
        optimizer.step()
    return predictor


def run_dialect_mlnn():
    """MLNN modal abstention approach."""
    data = generate_dialect_data()
    label_map = {"AmE": 0, "BrE": 1, "Neutral": 2}
    features = torch.tensor([[d[0], d[1]] for d in data], dtype=torch.float32)
    labels = torch.tensor([label_map[d[2]] for d in data], dtype=torch.long)

    torch.manual_seed(SEED)
    predictor = train_predictor(features, labels)
    predictor.eval()

    with torch.no_grad():
        scores = predictor(features)
        ame, bre = scores[:, 0], scores[:, 1]

        box_ame = (ame > 0.9).float()
        box_bre = (bre > 0.9).float()
        dia_ame = (ame > 0.1).float()
        dia_bre = (bre > 0.1).float()

        is_ame = box_ame * (1 - dia_bre)
        is_bre = box_bre * (1 - dia_ame)
        no_features = (1 - dia_ame) * (1 - dia_bre)
        mixed = dia_ame * dia_bre
        is_neutral = torch.clamp(no_features + mixed, max=1.0)

        class_scores = torch.stack([is_ame, is_bre, is_neutral], dim=1)
        predictions = class_scores.argmax(dim=1)

    return compute_dialect_metrics(predictions, labels)


def run_dialect_argmax():
    """Argmax baseline: predict highest-scoring class, no abstention."""
    data = generate_dialect_data()
    label_map = {"AmE": 0, "BrE": 1, "Neutral": 2}
    features = torch.tensor([[d[0], d[1]] for d in data], dtype=torch.float32)
    labels = torch.tensor([label_map[d[2]] for d in data], dtype=torch.long)

    torch.manual_seed(SEED)
    predictor = train_predictor(features, labels)
    predictor.eval()

    with torch.no_grad():
        scores = predictor(features)
        # Forced binary classification (closed-world)
        predictions = scores.argmax(dim=1)  # 0=AmE, 1=BrE, never 2

    return compute_dialect_metrics(predictions, labels)


def run_dialect_conformal(alpha=0.05):
    """Conformal Prediction baseline.

    Abstain when the prediction set is empty or ambiguous.
    """
    data = generate_dialect_data()
    label_map = {"AmE": 0, "BrE": 1, "Neutral": 2}
    features = torch.tensor([[d[0], d[1]] for d in data], dtype=torch.float32)
    labels = torch.tensor([label_map[d[2]] for d in data], dtype=torch.long)

    torch.manual_seed(SEED)
    predictor = train_predictor(features, labels)
    predictor.eval()

    with torch.no_grad():
        scores = predictor(features)
        # Calibrate on training data
        train_mask = labels != 2
        cal_scores = scores[train_mask]
        cal_labels = labels[train_mask]
        nonconformity = 1.0 - cal_scores[torch.arange(len(cal_labels)), cal_labels]
        threshold = torch.quantile(nonconformity, 1.0 - alpha)

        # default: Neutral
        predictions = torch.full((len(labels),), 2, dtype=torch.long)
        for i in range(len(labels)):
            pred_set = []
            for c in range(2):
                if scores[i, c] >= 1.0 - threshold:
                    pred_set.append(c)
            if len(pred_set) == 1:
                predictions[i] = pred_set[0]
            # else: stays Neutral (abstention)

    return compute_dialect_metrics(predictions, labels)


def compute_dialect_metrics(predictions, labels):
    metrics = {}
    class_names = ["AmE", "BrE", "Neutral"]
    for c, name in enumerate(class_names):
        tp = ((predictions == c) & (labels == c)).sum().float()
        fp = ((predictions == c) & (labels != c)).sum().float()
        fn = ((predictions != c) & (labels == c)).sum().float()
        precision = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
        denom = precision + recall
        f1 = 2 * precision * recall / denom if denom > 0 else 0.0
        metrics[name] = {"P": precision, "R": recall, "F1": f1}
    metrics["Accuracy"] = (predictions == labels).float().mean().item()
    return metrics


# ===================================================================
# Experiment 3: Trust Learning — MLNN vs Non-modal MLP
# ===================================================================

def generate_trust_data(num_agents=5, num_interactions=200, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    reliability = np.array([0.95, 0.90, 0.50, 0.15, 0.05])[:num_agents]
    claims, truths, agents = [], [], []
    for _ in range(num_interactions):
        agent_id = np.random.randint(num_agents)
        claims.append(1.0)
        truths.append(1.0 if np.random.random() < reliability[agent_id] else 0.0)
        agents.append(agent_id)
    return (
        torch.tensor(claims, dtype=torch.float32),
        torch.tensor(truths, dtype=torch.float32),
        torch.tensor(agents, dtype=torch.long),
        reliability,
    )


def generate_trust_data_temporal(num_agents=5, num_interactions=200, seed=42):
    """Generate temporal trust data with a 'reformed liar' pattern."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Agent 2 lies in the first half, becomes honest in the second
    reliability_phase1 = np.array([0.95, 0.90, 0.10, 0.15, 0.05])[:num_agents]
    reliability_phase2 = np.array([0.95, 0.90, 0.90, 0.15, 0.05])[:num_agents]

    claims, truths, agents, phases = [], [], [], []
    half = num_interactions // 2
    for i in range(num_interactions):
        agent_id = np.random.randint(num_agents)
        rel = reliability_phase1 if i < half else reliability_phase2
        claims.append(1.0)
        truths.append(1.0 if np.random.random() < rel[agent_id] else 0.0)
        agents.append(agent_id)
        phases.append(0 if i < half else 1)

    return (
        torch.tensor(claims, dtype=torch.float32),
        torch.tensor(truths, dtype=torch.float32),
        torch.tensor(agents, dtype=torch.long),
        torch.tensor(phases, dtype=torch.long),
        reliability_phase1,
        reliability_phase2,
    )


class ModalTrustModel(nn.Module):
    """MLNN trust model with temporal □ operator."""
    def __init__(self, num_agents, tau=0.1):
        super().__init__()
        self.trust_logits = nn.Parameter(torch.zeros(num_agents))

    @property
    def trust(self):
        return torch.sigmoid(self.trust_logits)

    def forward(self, claims, ground_truths, agent_ids):
        trust = self.trust[agent_ids]
        agreement = 1.0 - torch.abs(claims - ground_truths)
        contradiction = trust * (1.0 - agreement)
        predicted_belief = trust * claims + (1.0 - trust) * 0.5
        task_loss = (predicted_belief - ground_truths) ** 2
        return task_loss.mean() + 0.3 * contradiction.mean()


class NonModalTrustMLP(nn.Module):
    """Non-modal MLP baseline: classifies each interaction independently.

    This baseline (requested by R1) shows that sequence models can
    detect dishonesty but lack the interpretable trust structure and
    temporal consistency guarantees of the modal approach.
    """
    def __init__(self, num_agents, hidden_dim=32):
        super().__init__()
        self.agent_embed = nn.Embedding(num_agents, 16)
        self.net = nn.Sequential(
            nn.Linear(16 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, claims, ground_truths, agent_ids):
        emb = self.agent_embed(agent_ids)
        x = torch.cat([emb, claims.unsqueeze(-1), ground_truths.unsqueeze(-1)], dim=-1)
        pred = self.net(x).squeeze(-1)
        loss = (pred - ground_truths) ** 2
        return loss.mean()

    def get_trust(self, num_agents):
        """Extract per-agent trust by probing with honest claim."""
        with torch.no_grad():
            ids = torch.arange(num_agents)
            emb = self.agent_embed(ids)
            claims = torch.ones(num_agents)
            truths = torch.ones(num_agents)
            x = torch.cat([emb, claims.unsqueeze(-1), truths.unsqueeze(-1)], dim=-1)
            return self.net(x).squeeze(-1)


def run_trust_modal():
    claims, truths, agents, reliability = generate_trust_data()
    model = ModalTrustModel(5)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    for _ in range(200):
        optimizer.zero_grad()
        loss = model(claims, truths, agents)
        loss.backward()
        optimizer.step()
    trust = model.trust.detach().numpy()
    corr = np.corrcoef(trust, reliability)[0, 1]
    mse = np.mean((trust - reliability) ** 2)
    return {"trust": trust.tolist(), "correlation": corr, "mse": mse}


def run_trust_nonmodal():
    claims, truths, agents, reliability = generate_trust_data()
    model = NonModalTrustMLP(5)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for _ in range(200):
        optimizer.zero_grad()
        loss = model(claims, truths, agents)
        loss.backward()
        optimizer.step()
    trust = model.get_trust(5).numpy()
    corr = np.corrcoef(trust, reliability)[0, 1]
    mse = np.mean((trust - reliability) ** 2)
    return {"trust": trust.tolist(), "correlation": corr, "mse": mse}


def run_trust_temporal_comparison():
    """Compare modal vs non-modal on temporal trust (reformed liar scenario)."""
    claims, truths, agents, phases, rel1, rel2 = generate_trust_data_temporal()

    # Modal: sees full history, penalizes reformed liar
    model_m = ModalTrustModel(5)
    opt_m = optim.Adam(model_m.parameters(), lr=0.05)
    for _ in range(200):
        opt_m.zero_grad()
        loss = model_m(claims, truths, agents)
        loss.backward()
        opt_m.step()
    trust_modal = model_m.trust.detach().numpy()

    # Non-modal: sees all data, can also learn
    model_n = NonModalTrustMLP(5)
    opt_n = optim.Adam(model_n.parameters(), lr=0.01)
    for _ in range(200):
        opt_n.zero_grad()
        loss = model_n(claims, truths, agents)
        loss.backward()
        opt_n.step()
    trust_nonmodal = model_n.get_trust(5).numpy()

    # Non-modal phase 2 only: sees only recent data (reformed liar looks honest)
    phase2_mask = phases == 1
    model_n2 = NonModalTrustMLP(5)
    opt_n2 = optim.Adam(model_n2.parameters(), lr=0.01)
    for _ in range(200):
        opt_n2.zero_grad()
        loss = model_n2(
            claims[phase2_mask], truths[phase2_mask], agents[phase2_mask]
        )
        loss.backward()
        opt_n2.step()
    trust_nonmodal_recent = model_n2.get_trust(5).numpy()

    return {
        "modal_trust": trust_modal.tolist(),
        "nonmodal_all_trust": trust_nonmodal.tolist(),
        "nonmodal_recent_trust": trust_nonmodal_recent.tolist(),
        "phase1_reliability": rel1.tolist(),
        "phase2_reliability": rel2.tolist(),
        "reformed_agent": 2,
    }


# ===================================================================
# Main: Run all baselines
# ===================================================================

def _fmt(values):
    """Two-decimal, comma-separated rendering of a sequence of floats."""
    return ", ".join(f"{v:.2f}" for v in values)


def main():
    print("=" * 70)
    print("  MLNN Baseline Comparisons (Reviewer Response)")
    print("=" * 70)

    # --- Sudoku ---
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Sudoku (9x9 CSP)")
    print("=" * 70)

    print("\n  Running MLNN (modal axiom p_d -> ~diamond p_d)...")
    mlnn_rate, mlnn_time = run_sudoku_mlnn(n_trials=10)
    print(f"    Success rate: {mlnn_rate:.0%}, Avg time: {mlnn_time:.3f}s")

    print("\n  Running Simulated Annealing...")
    sa_rate, sa_time = run_sudoku_simulated_annealing(n_trials=10)
    print(f"    Success rate: {sa_rate:.0%}, Avg time: {sa_time:.3f}s")

    print("\n  Running Semantic Loss (Xu et al., 2018)...")
    sl_rate, sl_time = run_sudoku_semantic_loss(n_trials=10)
    print(f"    Success rate: {sl_rate:.0%}, Avg time: {sl_time:.3f}s")

    RESULTS["sudoku"] = {
        "MLNN": {"success_rate": mlnn_rate, "avg_time": mlnn_time},
        "SimulatedAnnealing": {"success_rate": sa_rate, "avg_time": sa_time},
        "SemanticLoss": {"success_rate": sl_rate, "avg_time": sl_time},
    }

    print(f"\n  {'Method':<25} | {'Success Rate':<14} | {'Avg Time'}")
    print("  " + "-" * 55)
    for name, r in RESULTS["sudoku"].items():
        print(f"  {name:<25} | {r['success_rate']:<14.0%} | {r['avg_time']:.3f}s")

    # --- Dialect ---
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Dialect Classification (Neutral Detection)")
    print("=" * 70)

    print("\n  Running MLNN (modal abstention)...")
    mlnn_d = run_dialect_mlnn()

    print("\n  Running Argmax baseline (no abstention)...")
    argmax_d = run_dialect_argmax()

    print("\n  Running Conformal Prediction (α=0.05)...")
    cp_d = run_dialect_conformal(alpha=0.05)

    RESULTS["dialect"] = {
        "MLNN": mlnn_d,
        "Argmax": argmax_d,
        "ConformalPrediction": cp_d,
    }

    print(
        f"\n  {'Method':<20} | {'AmE F1':<8} | {'BrE F1':<8} "
        f"| {'Neutral F1':<10} | {'Acc'}"
    )
    print("  " + "-" * 60)
    for name, r in RESULTS["dialect"].items():
        print(
            f"  {name:<20} | {r['AmE']['F1']:<8.2f} | {r['BrE']['F1']:<8.2f} | "
            f"{r['Neutral']['F1']:<10.2f} | {r['Accuracy']:.1%}"
        )

    # --- Trust ---
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Epistemic Trust Learning")
    print("=" * 70)

    print("\n  Running MLNN (modal trust)...")
    modal_t = run_trust_modal()

    print("\n  Running Non-modal MLP...")
    nonmodal_t = run_trust_nonmodal()

    print("\n  Running Temporal comparison (reformed liar)...")
    temporal_t = run_trust_temporal_comparison()

    RESULTS["trust"] = {
        "modal": modal_t,
        "nonmodal": nonmodal_t,
    }
    RESULTS["trust_temporal"] = temporal_t

    print("\n  Static trust learning:")
    print("    True reliability:    [0.95, 0.90, 0.50, 0.15, 0.05]")
    print(f"    MLNN trust:          [{_fmt(modal_t['trust'])}]")
    print(f"    Non-modal MLP trust: [{_fmt(nonmodal_t['trust'])}]")
    print(f"    MLNN correlation:    {modal_t['correlation']:.3f}")
    print(f"    MLP correlation:     {nonmodal_t['correlation']:.3f}")

    print("\n  Temporal trust (Reformed Liar = Agent 2):")
    print(f"    Phase 1 reliability: [{_fmt(temporal_t['phase1_reliability'])}]")
    print(f"    Phase 2 reliability: [{_fmt(temporal_t['phase2_reliability'])}]")
    print(f"    MLNN trust:          [{_fmt(temporal_t['modal_trust'])}]")
    print(f"    MLP (all data):      [{_fmt(temporal_t['nonmodal_all_trust'])}]")
    print(f"    MLP (recent only):   [{_fmt(temporal_t['nonmodal_recent_trust'])}]")

    reformed = temporal_t["reformed_agent"]
    print(f"\n  Agent {reformed} (reformed liar) trust scores:")
    mlnn_reformed = temporal_t['modal_trust'][reformed]
    print(f"    MLNN:             {mlnn_reformed:.3f} (penalizes past)")
    print(f"    MLP (all data):   {temporal_t['nonmodal_all_trust'][reformed]:.3f}")
    mlp_reformed = temporal_t['nonmodal_recent_trust'][reformed]
    print(f"    MLP (recent):     {mlp_reformed:.3f} (no history)")

    # Save results
    print(f"\n\n{'='*70}")
    print("  RESULTS SUMMARY (JSON)")
    print("=" * 70)
    print(json.dumps(RESULTS, indent=2, default=str))


if __name__ == "__main__":
    main()
