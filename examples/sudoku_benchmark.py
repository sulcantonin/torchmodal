r"""
Sudoku Benchmark — MLNN (modal Pure-Satisfiability) vs a wide baseline matrix
=============================================================================

Symbolic 9x9 Sudoku solved over a *difficulty-tiered puzzle distribution*
(not a single board), so every method is scored on solve-rate, cell
accuracy and wall-time across easy / medium / hard tiers.

This is the symbolic half of the third paper example.  The MLNN method is
the modal axiom  /\_d ( p_d -> ~<>p_d )  ("if digit d is here, it is not
possibly in any accessible cell") driven entirely by the contradiction /
axiom-violation residual — a direct demonstration of the paper's *Pure
Satisfiability Mode* (L_task = 0).

Baseline matrix
---------------
  Exact / complete :  Backtracking(+MRV), OR-Tools CP-SAT, Z3 (SMT),
                      PycoSAT (SAT-encoded), python-constraint (CSP)
  Propagation-only :  Naked + Hidden singles  (incomplete; the classical
                      analogue of MLNN's upward-downward bound propagation)
  Metaheuristic    :  Simulated Annealing, Min-Conflicts (random restart)
  Differentiable   :  MLNN (modal, this paper), Semantic Loss (Xu 2018),
                      Soft-nonmodal ablation (gradient descent, quadratic
                      peer penalty, modal operator removed)

External solvers are wrapped in a hard per-call timeout so the benchmark
never hangs; anything unavailable is skipped and reported, not faked.

Run:
  python examples/sudoku_benchmark.py            # full run
  MLNN_SUDOKU_SMOKE=1 python examples/sudoku_benchmark.py   # fast smoke
"""

from __future__ import annotations

import json
import os
import random
import signal

# Make sure the local development torchmodal (../torchmodal) shadows any
# PyPI-installed release, which lacks newer APIs and fails silently.
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torchmodal
from torchmodal import functional as F

N = 9          # grid side
B = 3          # box side
NW = N * N     # 81 worlds / cells
ND = N         # 9 digits

OUT = Path(__file__).resolve().parent / "sudoku_results"
OUT.mkdir(exist_ok=True)

SMOKE = os.environ.get("MLNN_SUDOKU_SMOKE", "0") == "1"

# Difficulty tiers by number of givens (clues).
TIERS = OrderedDict([("easy", 42), ("medium", 32), ("hard", 26)])
N_PUZZLES = 2 if SMOKE else 10          # puzzles per tier
MLNN_SEEDS = 2 if SMOKE else 5          # restart seeds for stochastic solvers
MLNN_EPOCHS = 600 if SMOKE else 2000    # matches torchmodal's proven setting
CALL_TIMEOUT = 30                       # seconds per solver call


# =====================================================================
#  Sudoku core
# =====================================================================
def _box_start(x):
    return (x // B) * B


def candidates(grid, r, c):
    """Valid digits 1..9 for cell (r,c) given current grid."""
    if grid[r, c] != 0:
        return set()
    used = set(grid[r, :]) | set(grid[:, c])
    br, bc = _box_start(r), _box_start(c)
    used |= set(grid[br:br + B, bc:bc + B].flatten())
    return set(range(1, 10)) - used


def _find_empty_mrv(grid):
    """Empty cell with the fewest candidates (minimum-remaining-values)."""
    best, best_n, best_cand = None, 10, None
    for r in range(N):
        for c in range(N):
            if grid[r, c] == 0:
                cand = candidates(grid, r, c)
                if len(cand) < best_n:
                    best, best_n, best_cand = (r, c), len(cand), cand
                    if best_n <= 1:
                        return best, best_cand
    return best, best_cand


def solve_backtracking(grid):
    """Complete DFS + MRV solver. Returns a solved grid or None."""
    g = grid.copy()

    def bt():
        cell, cand = _find_empty_mrv(g)
        if cell is None:
            return True            # no empty cell -> solved
        if not cand:
            return False           # dead end
        r, c = cell
        for d in cand:
            g[r, c] = d
            if bt():
                return True
            g[r, c] = 0
        return False

    return g if bt() else None


def count_solutions(grid, cap=2):
    """Count solutions up to `cap` (for uniqueness checking during gen)."""
    g = grid.copy()
    n = [0]

    def bt():
        cell, cand = _find_empty_mrv(g)
        if cell is None:
            n[0] += 1
            return n[0] >= cap
        if not cand:
            return False
        r, c = cell
        for d in cand:
            g[r, c] = d
            if bt():
                return True
            g[r, c] = 0
        return False

    bt()
    return n[0]


def _full_solution(rng):
    """Generate one complete valid grid via randomized backtracking."""
    g = np.zeros((N, N), dtype=int)

    def bt():
        cell, cand = _find_empty_mrv(g)
        if cell is None:
            return True
        if not cand:
            return False
        r, c = cell
        cand = list(cand)
        rng.shuffle(cand)
        for d in cand:
            g[r, c] = d
            if bt():
                return True
            g[r, c] = 0
        return False

    bt()
    return g


def generate_puzzle(n_clues, seed):
    """A puzzle (0 = blank) with a UNIQUE solution and ~n_clues givens."""
    rng = random.Random(seed)
    sol = _full_solution(rng)
    puzzle = sol.copy()
    cells = list(range(NW))
    rng.shuffle(cells)
    givens = NW
    for cell in cells:
        if givens <= n_clues:
            break
        r, c = divmod(cell, N)
        saved = puzzle[r, c]
        puzzle[r, c] = 0
        if count_solutions(puzzle, cap=2) != 1:   # keep uniqueness
            puzzle[r, c] = saved
        else:
            givens -= 1
    return puzzle, sol


def is_solved(grid):
    if grid is None or (grid == 0).any():
        return False
    for i in range(N):
        if len(set(grid[i, :])) != N or len(set(grid[:, i])) != N:
            return False
    for br in range(0, N, B):
        for bc in range(0, N, B):
            if len(set(grid[br:br + B, bc:bc + B].flatten())) != N:
                return False
    return True


def cell_accuracy(grid, sol):
    if grid is None:
        return 0.0
    return float((grid == sol).sum()) / NW


def build_R():
    """Fixed accessibility: 1 if cells share a row, column, or box."""
    A = torch.zeros(NW, NW)
    for i in range(NW):
        ri, ci = divmod(i, N)
        for j in range(NW):
            if i == j:
                continue
            rj, cj = divmod(j, N)
            if ri == rj or ci == cj or (_box_start(ri) == _box_start(rj)
                                        and _box_start(ci) == _box_start(cj)):
                A[i, j] = 1.0
    return A


R_GLOBAL = build_R()


# =====================================================================
#  Per-call hard timeout (so no solver can hang the benchmark)
# =====================================================================
class _Timeout(Exception):
    pass


def with_timeout(fn, puzzle, cap=CALL_TIMEOUT):
    """Run fn(puzzle); on timeout/error return (None, elapsed, status)."""
    def _handler(signum, frame):
        raise _Timeout()

    t0 = time.time()
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(cap)
    try:
        grid = fn(puzzle)
        status = "ok"
    except _Timeout:
        grid, status = None, "timeout"
    except Exception as e:                       # noqa: BLE001
        grid, status = None, f"error:{type(e).__name__}: {str(e)[:60]}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    return grid, time.time() - t0, status


# =====================================================================
#  Propagation-only baseline (naked + hidden singles, no search)
# =====================================================================
def solve_propagation(puzzle):
    g = puzzle.copy()
    changed = True
    while changed:
        changed = False
        # naked singles
        for r in range(N):
            for c in range(N):
                if g[r, c] == 0:
                    cand = candidates(g, r, c)
                    if len(cand) == 1:
                        g[r, c] = cand.pop()
                        changed = True
        # hidden singles (digit with a unique home in a unit)
        units = ([[(r, c) for c in range(N)] for r in range(N)]
                 + [[(r, c) for r in range(N)] for c in range(N)]
                 + [[(br + dr, bc + dc) for dr in range(B) for dc in range(B)]
                    for br in range(0, N, B) for bc in range(0, N, B)])
        for unit in units:
            for d in range(1, 10):
                spots = [(r, c) for (r, c) in unit
                         if g[r, c] == 0 and d in candidates(g, r, c)]
                if len(spots) == 1:
                    r, c = spots[0]
                    g[r, c] = d
                    changed = True
    return g


# =====================================================================
#  Exact external solvers
# =====================================================================
def solve_ortools(puzzle):
    from ortools.sat.python import cp_model
    m = cp_model.CpModel()
    x = [[m.NewIntVar(1, 9, f"x{r}{c}") for c in range(N)] for r in range(N)]
    for r in range(N):
        for c in range(N):
            if puzzle[r, c]:
                m.Add(x[r][c] == int(puzzle[r, c]))
    for i in range(N):
        m.AddAllDifferent([x[i][c] for c in range(N)])
        m.AddAllDifferent([x[r][i] for r in range(N)])
    for br in range(0, N, B):
        for bc in range(0, N, B):
            m.AddAllDifferent([x[br + dr][bc + dc]
                               for dr in range(B) for dc in range(B)])
    solver = cp_model.CpSolver()
    if solver.Solve(m) in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return np.array([[solver.Value(x[r][c]) for c in range(N)]
                         for r in range(N)])
    return None


def solve_z3(puzzle):
    import z3
    x = [[z3.Int(f"x_{r}_{c}") for c in range(N)] for r in range(N)]
    s = z3.Solver()
    for r in range(N):
        for c in range(N):
            s.add(x[r][c] >= 1, x[r][c] <= 9)
            if puzzle[r, c]:
                s.add(x[r][c] == int(puzzle[r, c]))
    for i in range(N):
        s.add(z3.Distinct([x[i][c] for c in range(N)]))
        s.add(z3.Distinct([x[r][i] for r in range(N)]))
    for br in range(0, N, B):
        for bc in range(0, N, B):
            s.add(z3.Distinct([x[br + dr][bc + dc]
                               for dr in range(B) for dc in range(B)]))
    if s.check() == z3.sat:
        mdl = s.model()
        return np.array([[mdl.evaluate(x[r][c]).as_long() for c in range(N)]
                         for r in range(N)])
    return None


def solve_pycosat(puzzle):
    import pycosat

    def v(r, c, d):                       # 1-based SAT variable id
        return r * 81 + c * 9 + d + 1
    clauses = []
    for r in range(N):
        for c in range(N):
            clauses.append([v(r, c, d) for d in range(9)])          # >=1 digit
            for d1 in range(9):
                for d2 in range(d1 + 1, 9):
                    clauses.append([-v(r, c, d1), -v(r, c, d2)])     # <=1 digit
    units = ([[(r, c) for c in range(N)] for r in range(N)]
             + [[(r, c) for r in range(N)] for c in range(N)]
             + [[(br + dr, bc + dc) for dr in range(B) for dc in range(B)]
                for br in range(0, N, B) for bc in range(0, N, B)])
    for unit in units:
        for d in range(9):
            for i in range(len(unit)):
                for j in range(i + 1, len(unit)):
                    (r1, c1), (r2, c2) = unit[i], unit[j]
                    clauses.append([-v(r1, c1, d), -v(r2, c2, d)])   # unit <=1
    for r in range(N):
        for c in range(N):
            if puzzle[r, c]:
                clauses.append([v(r, c, int(puzzle[r, c]) - 1)])     # givens
    sol = pycosat.solve(clauses)
    if sol in ("UNSAT", "UNKNOWN"):
        return None
    truth = set(lit for lit in sol if lit > 0)
    g = np.zeros((N, N), dtype=int)
    for r in range(N):
        for c in range(N):
            for d in range(9):
                if v(r, c, d) in truth:
                    g[r, c] = d + 1
    return g


def solve_python_constraint(puzzle):
    import constraint
    p = constraint.Problem()
    for r in range(N):
        for c in range(N):
            if puzzle[r, c]:
                p.addVariable((r, c), [int(puzzle[r, c])])
            else:
                p.addVariable((r, c), range(1, 10))
    for i in range(N):
        p.addConstraint(constraint.AllDifferentConstraint(),
                        [(i, c) for c in range(N)])
        p.addConstraint(constraint.AllDifferentConstraint(),
                        [(r, i) for r in range(N)])
    for br in range(0, N, B):
        for bc in range(0, N, B):
            p.addConstraint(constraint.AllDifferentConstraint(),
                            [(br + dr, bc + dc)
                             for dr in range(B) for dc in range(B)])
    s = p.getSolution()
    if not s:
        return None
    return np.array([[s[(r, c)] for c in range(N)] for r in range(N)])


# =====================================================================
#  Metaheuristics
# =====================================================================
def _box_init(puzzle, rng):
    """Fill each box with a valid permutation; return grid + free positions."""
    g = puzzle.copy()
    given = puzzle > 0
    free = {}
    for br in range(0, N, B):
        for bc in range(0, N, B):
            block = g[br:br + B, bc:bc + B]
            mask = given[br:br + B, bc:bc + B]
            missing = [d for d in range(1, 10) if d not in set(block[mask])]
            rng.shuffle(missing)
            pos = list(zip(*np.where(~mask)))
            for (pr, pc), d in zip(pos, missing):
                block[pr, pc] = d
            g[br:br + B, bc:bc + B] = block
            free[(br, bc)] = [(br + pr, bc + pc) for pr, pc in pos]
    return g, free


def _rc_violations(g):
    v = 0
    for i in range(N):
        v += (N - len(set(g[i, :]))) + (N - len(set(g[:, i])))
    return v


def solve_simulated_annealing(puzzle):
    rng = np.random.RandomState(abs(hash(puzzle.tobytes())) % (2 ** 31))
    g, free = _box_init(puzzle, rng)
    cur = _rc_violations(g)
    T0, T1 = 2.0, 1e-3
    iters = 60000 if SMOKE else 200000
    for it in range(iters):
        if cur == 0:
            break
        T = T0 * (T1 / T0) ** (it / iters)
        br, bc = rng.randint(0, B) * B, rng.randint(0, B) * B
        cells = free[(br, bc)]
        if len(cells) < 2:
            continue
        i, j = rng.choice(len(cells), 2, replace=False)
        (r1, c1), (r2, c2) = cells[i], cells[j]
        g[r1, c1], g[r2, c2] = g[r2, c2], g[r1, c1]
        new = _rc_violations(g)
        if new <= cur or rng.random() < np.exp(-(new - cur) / T):
            cur = new
        else:
            g[r1, c1], g[r2, c2] = g[r2, c2], g[r1, c1]
    return g


def solve_min_conflicts(puzzle):
    """Greedy min-conflicts over box swaps with random restarts."""
    base = np.random.RandomState(abs(hash(puzzle.tobytes())) % (2 ** 31))
    restarts = 5 if SMOKE else 30
    steps = 4000 if SMOKE else 20000
    best_grid, best_v = None, 1e9
    for _ in range(restarts):
        rng = np.random.RandomState(base.randint(0, 2 ** 31))
        g, free = _box_init(puzzle, rng)
        cur = _rc_violations(g)
        for _ in range(steps):
            if cur == 0:
                return g
            br, bc = rng.randint(0, B) * B, rng.randint(0, B) * B
            cells = free[(br, bc)]
            if len(cells) < 2:
                continue
            # try a few swaps, take the best (greedy, allow equal)
            best_local, bij = cur, None
            for _try in range(6):
                i, j = rng.choice(len(cells), 2, replace=False)
                (r1, c1), (r2, c2) = cells[i], cells[j]
                g[r1, c1], g[r2, c2] = g[r2, c2], g[r1, c1]
                nv = _rc_violations(g)
                g[r1, c1], g[r2, c2] = g[r2, c2], g[r1, c1]
                if nv <= best_local:
                    best_local, bij = nv, (r1, c1, r2, c2)
            if bij is not None and (best_local < cur or rng.random() < 0.3):
                r1, c1, r2, c2 = bij
                g[r1, c1], g[r2, c2] = g[r2, c2], g[r1, c1]
                cur = best_local
        if cur < best_v:
            best_grid, best_v = g.copy(), cur
    return best_grid


# =====================================================================
#  Differentiable methods (share the fixed peer graph R)
# =====================================================================
def _setup(puzzle):
    given, free = {}, []
    for r in range(N):
        for c in range(N):
            cell = r * N + c
            if puzzle[r, c]:
                given[cell] = int(puzzle[r, c]) - 1
            else:
                free.append(cell)
    return given, free


def _assemble(given, free, free_logits):
    probs = torch.zeros(NW, ND)
    for cell, d in given.items():
        probs[cell, d] = 1.0
    fp = torch.softmax(free_logits, dim=-1)
    for idx, cell in enumerate(free):
        probs[cell] = fp[idx]
    return probs, fp


def _decode(given, free, free_logits):
    with torch.no_grad():
        probs, _ = _assemble(given, free, free_logits)
    a = (probs.argmax(dim=-1) + 1).numpy()
    return a.reshape(N, N)


def solve_mlnn(puzzle, capture=False):
    """Modal axiom  p_d -> ~<>p_d  (Pure Satisfiability).  Multi-seed retry."""
    given, free = _setup(puzzle)
    R = R_GLOBAL
    crystal = torchmodal.CrystallizationLoss()
    snapshots = []
    for seed in range(MLNN_SEEDS):
        torch.manual_seed(seed)
        logits = nn.Parameter(torch.randn(len(free), ND) * 0.1)
        opt = optim.Adam([logits], lr=0.15)
        for ep in range(MLNN_EPOCHS):
            opt.zero_grad()
            tau = 0.5 * (0.01 / 0.5) ** (ep / MLNN_EPOCHS)
            probs, fp = _assemble(given, free, logits)
            axiom_loss = torch.tensor(0.0)
            percell = torch.zeros(NW)
            for d in range(ND):
                p_d = probs[:, d]
                viol = (1.0 - F.implication(p_d, F.negation(
                    F.possibility(p_d, R, tau=tau)))).clamp(min=0)
                axiom_loss = axiom_loss + viol.sum()
                percell = percell + viol.detach()
            crys_w = min(1.0, ep / (MLNN_EPOCHS * 0.3))
            (axiom_loss + crys_w * crystal(fp)).backward()
            opt.step()
            if capture and seed == 0 and ep in (
                    0, MLNN_EPOCHS // 4, MLNN_EPOCHS // 2, MLNN_EPOCHS - 1):
                snapshots.append((ep, percell.reshape(N, N).clone().numpy()))
        grid = _decode(given, free, logits)
        if is_solved(grid):
            return (grid, snapshots) if capture else grid
    return (grid, snapshots) if capture else grid


def solve_semantic_loss(puzzle):
    """Semantic Loss (Xu et al. 2018) + structural row/col/box marginals."""
    given, free = _setup(puzzle)
    sem = torchmodal.SemanticLoss()
    for seed in range(MLNN_SEEDS):
        torch.manual_seed(seed)
        logits = nn.Parameter(torch.randn(len(free), ND) * 0.1)
        opt = optim.Adam([logits], lr=0.15)
        for ep in range(MLNN_EPOCHS):
            opt.zero_grad()
            probs, fp = _assemble(given, free, logits)
            loss = sem.forward_mutual_exclusive(fp)
            grid = probs.view(N, N, ND)
            for d in range(ND):
                loss = loss + ((grid[:, :, d].sum(1) - 1.0) ** 2).sum()
                loss = loss + ((grid[:, :, d].sum(0) - 1.0) ** 2).sum()
                for br in range(0, N, B):
                    for bc in range(0, N, B):
                        loss = loss + (grid[br:br + B, bc:bc + B, d].sum()
                                       - 1.0) ** 2
            loss.backward()
            opt.step()
        g = _decode(given, free, logits)
        if is_solved(g):
            return g
    return g


def solve_soft_nonmodal(puzzle):
    """Ablation: same peer graph R, gradient descent, but a plain quadratic
    peer-conflict penalty instead of the modal operator."""
    given, free = _setup(puzzle)
    R = R_GLOBAL
    crystal = torchmodal.CrystallizationLoss()
    for seed in range(MLNN_SEEDS):
        torch.manual_seed(seed)
        logits = nn.Parameter(torch.randn(len(free), ND) * 0.1)
        opt = optim.Adam([logits], lr=0.15)
        for ep in range(MLNN_EPOCHS):
            opt.zero_grad()
            probs, fp = _assemble(given, free, logits)
            # sum_d  p_d^T R p_d  = shared-digit mass over accessible peers
            conflict = (probs * (R @ probs)).sum()
            crys_w = min(1.0, ep / (MLNN_EPOCHS * 0.3))
            (conflict + crys_w * crystal(fp)).backward()
            opt.step()
        g = _decode(given, free, logits)
        if is_solved(g):
            return g
    return g


# =====================================================================
#  Benchmark driver
# =====================================================================
# (name, fn, available)  — order = display order
def _avail(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


METHODS = [
    ("Backtracking(MRV)", solve_backtracking, True, "exact"),
    ("OR-Tools CP-SAT", solve_ortools, _avail("ortools"), "exact"),
    ("Z3 (SMT)", solve_z3, _avail("z3"), "exact"),
    ("PycoSAT (SAT)", solve_pycosat, _avail("pycosat"), "exact"),
    ("python-constraint", solve_python_constraint, _avail("constraint"), "exact"),
    ("Propagation-only", solve_propagation, True, "propagation"),
    ("SimulatedAnnealing", solve_simulated_annealing, True, "metaheuristic"),
    ("Min-Conflicts", solve_min_conflicts, True, "metaheuristic"),
    ("MLNN (modal, ours)", solve_mlnn, True, "differentiable"),
    ("SemanticLoss", solve_semantic_loss, True, "differentiable"),
    ("Soft-nonmodal(abl.)", solve_soft_nonmodal, True, "differentiable"),
]


def main():
    print("=" * 70)
    print(f"  Sudoku Benchmark  |  {'SMOKE' if SMOKE else 'FULL'}  |  "
          f"{N_PUZZLES} puzzles/tier, MLNN {MLNN_SEEDS} seeds x "
          f"{MLNN_EPOCHS} ep")
    print("=" * 70)

    # 1. Build the puzzle distribution
    puzzles = OrderedDict()
    for tier, clues in TIERS.items():
        puzzles[tier] = [generate_puzzle(clues, seed=1000 * list(TIERS).index(tier) + i)
                         for i in range(N_PUZZLES)]
        avg = np.mean([(p > 0).sum() for p, _ in puzzles[tier]])
        print(f"  generated {tier:7s}: {N_PUZZLES} puzzles, avg {avg:.0f} givens")

    # 2. Run every available method on every puzzle
    results = OrderedDict()
    for name, fn, ok, family in METHODS:
        if not ok:
            print(f"\n  [skip] {name} (dependency unavailable)")
            continue
        results[name] = {"family": family, "tiers": {}}
        # differentiable solvers run many epochs x seeds -> generous cap
        cap = (300 if SMOKE else 600) if family == "differentiable" \
            else (30 if SMOKE else 60)
        print(f"\n  >>> {name}  (timeout {cap}s/board)")
        for tier in TIERS:
            solved, accs, times = [], [], []
            statuses = Counter()
            for p, sol in puzzles[tier]:
                grid, dt, status = with_timeout(fn, p, cap=cap)
                statuses[status] += 1
                solved.append(1.0 if is_solved(grid) else 0.0)
                accs.append(cell_accuracy(grid, sol))
                times.append(dt)
            results[name]["tiers"][tier] = {
                "solve_rate": float(np.mean(solved)),
                "cell_acc": float(np.mean(accs)),
                "time_s": float(np.mean(times)),
                "statuses": dict(statuses),
            }
            bad = {s: c for s, c in statuses.items() if s != "ok"}
            bad_note = ("  !! " + ", ".join(f"{s} x{c}" for s, c in bad.items())
                        if bad else "")
            print(f"      {tier:7s}  solve {np.mean(solved):4.0%}  "
                  f"cell {np.mean(accs):5.1%}  {np.mean(times):6.2f}s{bad_note}")

    # 3. Interpretive MLNN surface: per-cell conflict map draining to 0
    print("\n  capturing MLNN conflict-map on one medium puzzle ...")
    demo_p, _ = puzzles["medium"][0]
    _, snaps = solve_mlnn(demo_p, capture=True)

    # 4. Persist
    save = {"config": {"smoke": SMOKE, "n_puzzles": N_PUZZLES,
                        "mlnn_seeds": MLNN_SEEDS, "mlnn_epochs": MLNN_EPOCHS,
                        "tiers": TIERS}, "results": results}
    (OUT / "results.json").write_text(json.dumps(save, indent=2))
    _write_markdown(results)
    _plot(results, snaps, demo_p)
    print(f"\n  wrote {OUT}/results.json, RESULTS.md, *.png")


def _write_markdown(results):
    lines = ["# Sudoku Benchmark — MLNN vs baseline matrix\n",
             f"_{'SMOKE' if SMOKE else 'FULL'} run: {N_PUZZLES} puzzles/tier; "
             f"MLNN {MLNN_SEEDS} seeds x {MLNN_EPOCHS} epochs._\n",
             "Solve-rate (% boards fully solved) per method x difficulty:\n",
             "| Method | Family | Easy | Medium | Hard | Overall | mean s |",
             "|---|---|---|---|---|---|---|"]
    for name, d in results.items():
        t = d["tiers"]
        ov = np.mean([t[k]["solve_rate"] for k in t])
        mt = np.mean([t[k]["time_s"] for k in t])
        lines.append(
            f"| {name} | {d['family']} | "
            + " | ".join(f"{t[k]['solve_rate']:.0%}" for k in TIERS)
            + f" | **{ov:.0%}** | {mt:.2f} |")
    lines += ["", "Mean cell accuracy per method x difficulty:\n",
              "| Method | Easy | Medium | Hard |", "|---|---|---|---|"]
    for name, d in results.items():
        t = d["tiers"]
        lines.append(f"| {name} | "
                     + " | ".join(f"{t[k]['cell_acc']:.1%}" for k in TIERS)
                     + " |")
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")


def _plot(results, snaps, demo_p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (a) grouped solve-rate bars
    names = list(results.keys())
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(12, 5))
    for k, tier in enumerate(TIERS):
        ax.bar(x + (k - 1) * w,
               [results[n]["tiers"][tier]["solve_rate"] for n in names],
               w, label=tier)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("board solve-rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Sudoku solve-rate by method and difficulty")
    ax.legend(title="tier")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "solve_rate.png", dpi=130)
    plt.close(fig)

    # (b) MLNN conflict-map draining to zero
    if snaps:
        fig, axes = plt.subplots(1, len(snaps), figsize=(3 * len(snaps), 3.2))
        if len(snaps) == 1:
            axes = [axes]
        vmax = max(s[1].max() for s in snaps) or 1.0
        for ax, (ep, m) in zip(axes, snaps):
            im = ax.imshow(m, cmap="magma_r", vmin=0, vmax=vmax)
            ax.set_title(f"epoch {ep}\nΣ L_contra={m.sum():.2f}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in (3, 6):
                ax.axhline(s - 0.5, color="c", lw=0.6)
                ax.axvline(s - 0.5, color="c", lw=0.6)
        fig.suptitle("MLNN per-cell contradiction residual L_contra "
                     "(Pure Satisfiability solve)", fontsize=11)
        fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
        fig.savefig(OUT / "mlnn_conflict_map.png", dpi=130,
                    bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    main()
