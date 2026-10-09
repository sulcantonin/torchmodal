"""
Graph k-Coloring Benchmark — MLNN (modal Pure-Satisfiability) vs baseline matrix
================================================================================

Constraint satisfaction *perfectly tailored to modal logic*: in graph
coloring the adjacency relation **is** the Kripke accessibility relation A,
and "adjacent nodes get different colors" is literally the modal axiom

        /\\_c ( p_c -> ~<>p_c )
        "if I am color c, no accessible (adjacent) node is color c."

Sudoku is the special case (A = the 81-cell peer graph, k = 9). Here A is an
arbitrary graph, so the modal framing is the natural one, not scaffolding.

Two capabilities:
  (1) SOLVE  — fixed A, find a proper k-coloring by Pure Satisfiability
               (L_task = 0; minimize the axiom-violation residual). Scored
               against a wide baseline matrix over a difficulty-tiered
               distribution of planted-k-colorable graphs.
  (2) LEARN  — hidden A, recover the conflict graph A_theta from valid
               colorings alone (the inspectable-accessibility demo Sudoku
               cannot give, because its rules are known).

Baseline matrix (capability 1)
------------------------------
  Exact / complete :  Backtracking(+DSATUR), OR-Tools CP-SAT, Z3 (SMT),
                      PycoSAT (SAT-encoded), python-constraint (CSP)
  Heuristic        :  Greedy DSATUR (incomplete; classical analogue of
                      MLNN's upward-downward propagation)
  Metaheuristic    :  Simulated Annealing, Min-Conflicts (random restart)
  Differentiable   :  MLNN (modal, ours), Semantic Loss (Xu 2018),
                      Soft-nonmodal ablation (quadratic peer penalty, modal
                      operator removed)

Run:
  python examples/graph_coloring_benchmark.py
  MLNN_COLOR_SMOKE=1 python examples/graph_coloring_benchmark.py   # fast smoke
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

OUT = Path(__file__).resolve().parent / "coloring_results"
OUT.mkdir(exist_ok=True)

SMOKE = os.environ.get("MLNN_COLOR_SMOKE", "0") == "1"

# Difficulty tiers: (n_nodes, k_colors, edge_density). Graphs are planted
# k-colorable; density rising => fewer proper colorings => harder for the
# incomplete methods (exact solvers stay at 100%).
TIERS = OrderedDict([
    ("easy",   dict(n=12, k=3, p=0.30)),
    ("medium", dict(n=20, k=3, p=0.35)),
    ("hard",   dict(n=30, k=3, p=0.40)),
])
N_GRAPHS = 2 if SMOKE else 10
MLNN_SEEDS = 2 if SMOKE else 5
MLNN_EPOCHS = 600 if SMOKE else 2000
CALL_TIMEOUT = 30


# =====================================================================
#  Graph generation + scoring
# =====================================================================
def generate_graph(n, k, p, seed):
    """Planted-k-colorable G(n, p): edges only between differently-colored
    nodes, so the planted coloring is a proper k-coloring witness."""
    rng = random.Random(seed)
    planted = [rng.randrange(k) for _ in range(n)]
    A = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            if planted[i] != planted[j] and rng.random() < p:
                A[i, j] = A[j, i] = 1
    return A, planted


def _random_proper_coloring(A, k, rng):
    """A proper k-coloring via greedy with randomized node + color order
    (returns None if greedy gets stuck — caller just retries)."""
    n = len(A)
    adj = [np.where(A[i])[0] for i in range(n)]
    order = list(range(n))
    rng.shuffle(order)
    col = [-1] * n
    for v in order:
        used = {col[u] for u in adj[v] if col[u] >= 0}
        avail = [c for c in range(k) if c not in used]
        if not avail:
            return None
        col[v] = avail[rng.randrange(len(avail))]
    return col


def n_conflicts(A, coloring):
    """Number of monochromatic (improperly colored) edges."""
    if coloring is None:
        return 1 << 30
    col = np.asarray(coloring)
    same = (col[:, None] == col[None, :]).astype(np.int64)
    return int((np.triu(A, 1) * same).sum())


def is_proper(A, coloring):
    return coloring is not None and n_conflicts(A, coloring) == 0


def conflict_free_node_frac(A, coloring):
    """Fraction of nodes with no incident monochromatic edge (soft metric)."""
    if coloring is None:
        return 0.0
    col = np.asarray(coloring)
    bad = np.zeros(len(col), dtype=bool)
    ii, jj = np.where(np.triu(A, 1))
    for i, j in zip(ii, jj):
        if col[i] == col[j]:
            bad[i] = bad[j] = True
    return float((~bad).mean())


# =====================================================================
#  Per-call hard timeout
# =====================================================================
class _Timeout(Exception):
    pass


def with_timeout(fn, A, k, cap):
    def _handler(signum, frame):
        raise _Timeout()
    t0 = time.time()
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(cap)
    try:
        coloring = fn(A, k)
        status = "ok"
    except _Timeout:
        coloring, status = None, "timeout"
    except Exception as e:                       # noqa: BLE001
        coloring, status = None, f"error:{type(e).__name__}: {str(e)[:60]}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    return coloring, time.time() - t0, status


# =====================================================================
#  Exact / complete solvers
# =====================================================================
def solve_backtracking(A, k):
    """DFS with DSATUR-style most-constrained-node ordering."""
    n = len(A)
    adj = [np.where(A[i])[0] for i in range(n)]
    color = [-1] * n

    def saturation(v):
        return len({color[u] for u in adj[v] if color[u] >= 0})

    def pick():
        best, best_key = -1, None
        for v in range(n):
            if color[v] < 0:
                key = (saturation(v), len(adj[v]))
                if best_key is None or key > best_key:
                    best, best_key = v, key
        return best

    def bt(assigned):
        if assigned == n:
            return True
        v = pick()
        used = {color[u] for u in adj[v] if color[u] >= 0}
        for c in range(k):
            if c not in used:
                color[v] = c
                if bt(assigned + 1):
                    return True
                color[v] = -1
        return False

    return list(color) if bt(0) else None


def solve_ortools(A, k):
    from ortools.sat.python import cp_model
    n = len(A)
    m = cp_model.CpModel()
    x = [m.NewIntVar(0, k - 1, f"x{i}") for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                m.Add(x[i] != x[j])
    s = cp_model.CpSolver()
    if s.Solve(m) in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [int(s.Value(x[i])) for i in range(n)]
    return None


def solve_z3(A, k):
    import z3
    n = len(A)
    x = [z3.Int(f"c{i}") for i in range(n)]
    s = z3.Solver()
    for i in range(n):
        s.add(x[i] >= 0, x[i] < k)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                s.add(x[i] != x[j])
    if s.check() == z3.sat:
        mdl = s.model()
        return [mdl.evaluate(x[i]).as_long() for i in range(n)]
    return None


def solve_pycosat(A, k):
    import pycosat
    n = len(A)

    def v(node, color):
        return node * k + color + 1
    clauses = []
    for i in range(n):
        clauses.append([v(i, c) for c in range(k)])               # >=1 color
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                clauses.append([-v(i, c1), -v(i, c2)])             # <=1 color
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                for c in range(k):
                    clauses.append([-v(i, c), -v(j, c)])           # edge differ
    sol = pycosat.solve(clauses)
    if sol in ("UNSAT", "UNKNOWN"):
        return None
    truth = set(lit for lit in sol if lit > 0)
    return [next(c for c in range(k) if v(i, c) in truth) for i in range(n)]


def solve_python_constraint(A, k):
    import constraint
    n = len(A)
    p = constraint.Problem()
    p.addVariables(range(n), range(k))
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                p.addConstraint(lambda a, b: a != b, (i, j))
    s = p.getSolution()
    return [s[i] for i in range(n)] if s else None


# =====================================================================
#  Greedy heuristic (incomplete)
# =====================================================================
def solve_greedy_dsatur(A, k):
    """DSATUR greedy: color the most-saturated node with the smallest legal
    color; fail if any node needs a (k+1)-th color."""
    n = len(A)
    adj = [np.where(A[i])[0] for i in range(n)]
    color = [-1] * n
    for _ in range(n):
        best, best_key = -1, None
        for v in range(n):
            if color[v] < 0:
                sat = len({color[u] for u in adj[v] if color[u] >= 0})
                key = (sat, len(adj[v]))
                if best_key is None or key > best_key:
                    best, best_key = v, key
        used = {color[u] for u in adj[best] if color[u] >= 0}
        c = next((c for c in range(k) if c not in used), None)
        if c is None:
            return color            # stuck -> improper (incomplete)
        color[best] = c
    return color


# =====================================================================
#  Metaheuristics
# =====================================================================
def solve_simulated_annealing(A, k):
    rng = np.random.RandomState(abs(hash(A.tobytes())) % (2 ** 31))
    n = len(A)
    col = rng.randint(0, k, size=n)
    cur = n_conflicts(A, col)
    iters = 20000 if SMOKE else 100000
    T0, T1 = 2.0, 1e-3
    for it in range(iters):
        if cur == 0:
            break
        T = T0 * (T1 / T0) ** (it / iters)
        v = rng.randint(n)
        old = col[v]
        new = rng.randint(k)
        if new == old:
            continue
        col[v] = new
        nc = n_conflicts(A, col)
        if nc <= cur or rng.random() < np.exp(-(nc - cur) / T):
            cur = nc
        else:
            col[v] = old
    return list(col)


def solve_min_conflicts(A, k):
    base = np.random.RandomState(abs(hash(A.tobytes())) % (2 ** 31))
    n = len(A)
    adj = [np.where(A[i])[0] for i in range(n)]
    restarts = 5 if SMOKE else 25
    steps = 2000 if SMOKE else 10000
    best, best_c = None, 1 << 30
    for _ in range(restarts):
        rng = np.random.RandomState(base.randint(0, 2 ** 31))
        col = rng.randint(0, k, size=n)
        for _ in range(steps):
            conf = [v for v in range(n)
                    if any(col[u] == col[v] for u in adj[v])]
            if not conf:
                return list(col)
            v = conf[rng.randint(len(conf))]
            counts = [sum(1 for u in adj[v] if col[u] == c) for c in range(k)]
            col[v] = int(np.argmin(counts))
        c = n_conflicts(A, col)
        if c < best_c:
            best, best_c = list(col), c
    return best


# =====================================================================
#  Differentiable methods (share the adjacency R = A)
# =====================================================================
def _decode(logits):
    with torch.no_grad():
        return torch.softmax(logits, dim=-1).argmax(-1).tolist()


def solve_mlnn(A, k, capture=False):
    """Modal axiom  p_c -> ~<>p_c  over the adjacency (Pure Satisfiability)."""
    n = len(A)
    R = torch.tensor(A, dtype=torch.float32)
    crystal = torchmodal.CrystallizationLoss()
    snaps = []
    for seed in range(MLNN_SEEDS):
        torch.manual_seed(seed)
        logits = nn.Parameter(torch.randn(n, k) * 0.1)
        opt = optim.Adam([logits], lr=0.15)
        for ep in range(MLNN_EPOCHS):
            opt.zero_grad()
            tau = 0.5 * (0.01 / 0.5) ** (ep / MLNN_EPOCHS)
            p = torch.softmax(logits, dim=-1)
            axiom_loss = torch.tensor(0.0)
            percell = torch.zeros(n)
            for c in range(k):
                pc = p[:, c]
                viol = (1.0 - F.implication(pc, F.negation(
                    F.possibility(pc, R, tau=tau)))).clamp(min=0)
                axiom_loss = axiom_loss + viol.sum()
                percell = percell + viol.detach()
            crys_w = min(1.0, ep / (MLNN_EPOCHS * 0.3))
            (axiom_loss + crys_w * crystal(p)).backward()
            opt.step()
            if capture and seed == 0 and ep in (
                    0, MLNN_EPOCHS // 3, 2 * MLNN_EPOCHS // 3, MLNN_EPOCHS - 1):
                snaps.append((ep, float(percell.sum()),
                              _decode(logits), percell.clone().numpy()))
        col = _decode(logits)
        if is_proper(A, col):
            return (col, snaps) if capture else col
    return (col, snaps) if capture else col


def solve_semantic_loss(A, k):
    """Semantic Loss: one-color-per-node (mutual exclusive) + edge penalty."""
    n = len(A)
    R = torch.tensor(A, dtype=torch.float32)
    sem = torchmodal.SemanticLoss()
    for seed in range(MLNN_SEEDS):
        torch.manual_seed(seed)
        logits = nn.Parameter(torch.randn(n, k) * 0.1)
        opt = optim.Adam([logits], lr=0.15)
        for ep in range(MLNN_EPOCHS):
            opt.zero_grad()
            p = torch.softmax(logits, dim=-1)
            loss = sem.forward_mutual_exclusive(p)
            loss = loss + (p * (R @ p)).sum()          # shared-color edge mass
            loss.backward()
            opt.step()
        col = _decode(logits)
        if is_proper(A, col):
            return col
    return col


def solve_soft_nonmodal(A, k):
    """Ablation: same adjacency, gradient descent, quadratic peer penalty,
    modal operator removed."""
    n = len(A)
    R = torch.tensor(A, dtype=torch.float32)
    crystal = torchmodal.CrystallizationLoss()
    for seed in range(MLNN_SEEDS):
        torch.manual_seed(seed)
        logits = nn.Parameter(torch.randn(n, k) * 0.1)
        opt = optim.Adam([logits], lr=0.15)
        for ep in range(MLNN_EPOCHS):
            opt.zero_grad()
            p = torch.softmax(logits, dim=-1)
            conflict = (p * (R @ p)).sum()             # sum_c p_c^T A p_c
            crys_w = min(1.0, ep / (MLNN_EPOCHS * 0.3))
            (conflict + crys_w * crystal(p)).backward()
            opt.step()
        col = _decode(logits)
        if is_proper(A, col):
            return col
    return col


def solve_rrn(A, k):
    """RRN-style recurrent message-passing GNN (Palm et al. 2018 relational
    inductive bias) run per-instance as an energy minimiser: T rounds of
    learned neighbour messaging produce node colour logits, optimised to drive
    the colouring conflict to zero. The published RRN is trained *supervised*
    on a corpus of solved instances; lacking that protocol here we use this
    unsupervised per-instance proxy so it shares MLNN's setting exactly."""
    n = len(A)
    R = torch.tensor(A, dtype=torch.float32)
    Rn = R / R.sum(1, keepdim=True).clamp(min=1.0)        # mean-aggregation
    H, T = 32, 4
    for seed in range(MLNN_SEEDS):
        torch.manual_seed(seed)
        emb = nn.Parameter(torch.randn(n, H) * 0.1)
        msg, gru, out = nn.Linear(H, H), nn.GRUCell(H, H), nn.Linear(H, k)
        params = [emb, *msg.parameters(), *gru.parameters(), *out.parameters()]
        opt = optim.Adam(params, lr=0.05)

        def forward():
            h = emb
            for _ in range(T):
                h = gru(Rn @ torch.tanh(msg(h)), h)       # message + update
            return torch.softmax(out(h), dim=-1)

        for _ep in range(MLNN_EPOCHS):
            opt.zero_grad()
            p = forward()
            conflict = (p * (R @ p)).sum()                # shared-colour edge mass
            ent = -(p * (p + 1e-9).log()).sum()           # crystallise
            (conflict + 0.05 * ent).backward()
            opt.step()
        with torch.no_grad():
            col = forward().argmax(-1).tolist()
        if is_proper(A, col):
            return col
    return col


def solve_satnet(A, k):
    """SATNet differentiable MaxSAT layer (Wang et al. 2019). SATNet *learns*
    the constraint rules from a corpus of solved instances and requires a CUDA
    build, so it is a supervised, CUDA-only solver outside this per-instance
    CPU/MPS protocol. Gated off here; trained results are cited in the text
    (e.g. ~98% boards on symbolic Sudoku)."""
    import satnet  # noqa: F401  (gated by the METHODS availability check)
    raise NotImplementedError(
        "SATNet is supervised and CUDA-only; not run in this per-instance "
        "CPU/MPS protocol. See Wang et al. 2019.")


# =====================================================================
#  Capability 2 — learn the accessibility from valid colorings
# =====================================================================
def learn_accessibility_demo(seed=0):
    """Hide the graph; give only proper colorings; learn A_theta so the modal
    coloring axiom holds for every coloring while A is pushed *as large as
    possible*. Adjacent pairs (which never share a color) saturate to 1;
    non-adjacent pairs (which sometimes share a color) are driven to 0 by the
    axiom. Recovers the constraint graph from colorings alone."""
    from sklearn.metrics import roc_auc_score
    n, k, p = 16, 3, 0.25
    A_true, planted = generate_graph(n, k, p, seed=seed)

    rng = random.Random(seed + 1)
    seen, colorings, tries = {tuple(planted)}, [planted], 0
    target = 40 if SMOKE else 60
    while len(colorings) < target and tries < 60000:
        tries += 1
        col = _random_proper_coloring(A_true, k, rng)
        if col is not None and tuple(col) not in seen:
            seen.add(tuple(col))
            colorings.append(col)
    if len(colorings) < 5:
        return {"n": n, "k": k, "n_colorings": len(colorings),
                "edge_recovery_auc": float("nan"),
                "A_true": A_true, "A_learned": np.zeros((n, n))}

    Ps = [torch.eye(k)[torch.tensor(c)] for c in colorings]   # each (n, k)
    M = len(colorings)
    logitsA = nn.Parameter(torch.zeros(n, n))
    opt = optim.Adam([logitsA], lr=0.05)
    eye = torch.eye(n)
    LAM = 2.0
    epochs = 400 if SMOKE else 800
    for ep in range(epochs):
        opt.zero_grad()
        A = torch.sigmoid(logitsA)
        A = 0.5 * (A + A.t()) * (1 - eye)             # symmetric, hollow
        viol = torch.tensor(0.0)
        for Pc in Ps:
            for c in range(k):
                pc = Pc[:, c]
                dia = F.possibility(pc, A, tau=0.1)
                viol = viol + (1.0 - F.implication(
                    pc, F.negation(dia))).clamp(min=0).sum()
        loss = viol / M - LAM * A.mean()              # satisfy axiom, else maximise A
        loss.backward()
        opt.step()
    with torch.no_grad():
        A = torch.sigmoid(logitsA)
        A = (0.5 * (A + A.t()) * (1 - eye)).numpy()
    iu = np.triu_indices(n, 1)
    n_edges = int(A_true[iu].sum())
    auc = (float(roc_auc_score(A_true[iu], A[iu]))
           if 0 < n_edges < len(iu[0]) else float("nan"))
    return {"n": n, "k": k, "n_colorings": M, "n_edges": n_edges,
            "edge_recovery_auc": auc, "A_true": A_true, "A_learned": A}


# =====================================================================
#  Benchmark driver
# =====================================================================
def _avail(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


METHODS = [
    ("Backtracking(DSATUR)", solve_backtracking, True, "exact"),
    ("OR-Tools CP-SAT", solve_ortools, _avail("ortools"), "exact"),
    ("Z3 (SMT)", solve_z3, _avail("z3"), "exact"),
    ("PycoSAT (SAT)", solve_pycosat, _avail("pycosat"), "exact"),
    ("python-constraint", solve_python_constraint, _avail("constraint"), "exact"),
    ("Greedy DSATUR", solve_greedy_dsatur, True, "heuristic"),
    ("SimulatedAnnealing", solve_simulated_annealing, True, "metaheuristic"),
    ("Min-Conflicts", solve_min_conflicts, True, "metaheuristic"),
    ("MLNN (modal, ours)", solve_mlnn, True, "differentiable"),
    ("SemanticLoss", solve_semantic_loss, True, "differentiable"),
    ("RRN-style GNN", solve_rrn, True, "differentiable"),
    ("SATNet (Wang'19)", solve_satnet, _avail("satnet"), "differentiable"),
    ("Soft-nonmodal(abl.)", solve_soft_nonmodal, True, "differentiable"),
]


def main():
    print("=" * 70)
    print(f"  Graph k-Coloring Benchmark | {'SMOKE' if SMOKE else 'FULL'} | "
          f"{N_GRAPHS} graphs/tier, MLNN {MLNN_SEEDS}x{MLNN_EPOCHS}")
    print("=" * 70)

    graphs = OrderedDict()
    for tier, cfg in TIERS.items():
        gs = [generate_graph(cfg["n"], cfg["k"], cfg["p"],
                             seed=1000 * list(TIERS).index(tier) + i)
              for i in range(N_GRAPHS)]
        deg = np.mean([g[0].sum() / cfg["n"] for g in gs])
        graphs[tier] = (cfg, gs)
        print(f"  {tier:7s}: n={cfg['n']} k={cfg['k']} p={cfg['p']} "
              f"avg_deg={deg:.1f}  ({N_GRAPHS} graphs)")

    results = OrderedDict()
    for name, fn, ok, family in METHODS:
        if not ok:
            print(f"\n  [skip] {name}")
            continue
        results[name] = {"family": family, "tiers": {}}
        cap = (300 if SMOKE else 600) if family == "differentiable" \
            else (15 if SMOKE else 30)
        print(f"\n  >>> {name}  (timeout {cap}s)")
        for tier, (cfg, gs) in graphs.items():
            solved, frac, times = [], [], []
            statuses = Counter()
            for A, _planted in gs:
                col, dt, status = with_timeout(fn, A, cfg["k"], cap)
                statuses[status] += 1
                solved.append(1.0 if is_proper(A, col) else 0.0)
                frac.append(conflict_free_node_frac(A, col))
                times.append(dt)
            results[name]["tiers"][tier] = {
                "solve_rate": float(np.mean(solved)),
                "conflict_free_frac": float(np.mean(frac)),
                "time_s": float(np.mean(times)),
                "statuses": dict(statuses),
            }
            bad = {s: c for s, c in statuses.items() if s != "ok"}
            bad_note = ("  !! " + ", ".join(f"{s} x{c}" for s, c in bad.items())
                        if bad else "")
            print(f"      {tier:7s}  solve {np.mean(solved):4.0%}  "
                  f"cfree {np.mean(frac):5.1%}  {np.mean(times):6.2f}s{bad_note}")

    # interpretive MLNN surface on one medium graph
    print("\n  capturing MLNN conflict trajectory on one medium graph ...")
    demo_A = graphs["medium"][1][0][0]
    _, snaps = solve_mlnn(demo_A, TIERS["medium"]["k"], capture=True)

    # capability 2: learn the accessibility from valid colorings
    print("  learning accessibility A_theta from valid colorings ...")
    try:
        learn = learn_accessibility_demo(seed=0)
        print(f"      edge-recovery AUC (A_theta vs true adjacency) = "
              f"{learn['edge_recovery_auc']:.3f}  "
              f"(from {learn['n_colorings']} colorings of a hidden "
              f"{learn['n']}-node graph)")
    except Exception as e:                                   # noqa: BLE001
        print(f"      [learn-A skipped: {type(e).__name__}: {e}]")
        learn = {"n": 0, "k": 0, "n_colorings": 0, "n_edges": 0,
                 "edge_recovery_auc": float("nan"),
                 "A_true": np.zeros((1, 1)), "A_learned": np.zeros((1, 1))}

    save = {"config": {"smoke": SMOKE, "n_graphs": N_GRAPHS,
                       "mlnn_seeds": MLNN_SEEDS, "mlnn_epochs": MLNN_EPOCHS,
                       "tiers": {t: c for t, c in TIERS.items()}},
            "results": results,
            "learn_accessibility": {kk: vv for kk, vv in learn.items()
                                    if kk not in ("A_true", "A_learned")}}
    (OUT / "results.json").write_text(json.dumps(save, indent=2))
    _write_markdown(results, learn)
    _plot(results, snaps, demo_A, learn)
    print(f"\n  wrote {OUT}/results.json, RESULTS.md, *.png")


def _write_markdown(results, learn):
    lines = ["# Graph k-Coloring Benchmark — MLNN vs baseline matrix\n",
             f"_{'SMOKE' if SMOKE else 'FULL'}: {N_GRAPHS} planted-3-colorable "
             f"graphs/tier; MLNN {MLNN_SEEDS} seeds x {MLNN_EPOCHS} epochs._\n",
             "Proper-coloring solve-rate per method x difficulty:\n",
             "| Method | Family | Easy | Medium | Hard | Overall | mean s |",
             "|---|---|---|---|---|---|---|"]
    for name, d in results.items():
        t = d["tiers"]
        ov = np.mean([t[x]["solve_rate"] for x in t])
        mt = np.mean([t[x]["time_s"] for x in t])
        lines.append(f"| {name} | {d['family']} | "
                     + " | ".join(f"{t[x]['solve_rate']:.0%}" for x in TIERS)
                     + f" | **{ov:.0%}** | {mt:.2f} |")
    lines += ["",
              f"**Capability 2 (learnable A):** from {learn['n_colorings']} "
              f"valid colorings of a hidden {learn['n']}-node graph, the learned "
              f"A_theta recovers the true adjacency at "
              f"**edge-recovery AUC = {learn['edge_recovery_auc']:.3f}** "
              f"(the inspectable-accessibility result Sudoku cannot give).", ""]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")


def _plot(results, snaps, demo_A, learn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    # (a) grouped solve-rate bars
    names = list(results.keys())
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(12, 5))
    for ki, tier in enumerate(TIERS):
        ax.bar(x + (ki - 1) * w,
               [results[nm]["tiers"][tier]["solve_rate"] for nm in names],
               w, label=tier)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("proper-coloring solve-rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Graph 3-coloring solve-rate by method and difficulty")
    ax.legend(title="tier")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "solve_rate.png", dpi=130)
    plt.close(fig)

    # (b) MLNN solving one graph: conflict edges draining as L_contra -> 0
    if snaps:
        G = nx.from_numpy_array(demo_A)
        pos = nx.spring_layout(G, seed=0)
        palette = np.array([[0.20, 0.55, 0.85], [0.90, 0.45, 0.20],
                            [0.30, 0.70, 0.40], [0.7, 0.3, 0.7]])
        fig, axes = plt.subplots(1, len(snaps), figsize=(3.4 * len(snaps), 3.4))
        if len(snaps) == 1:
            axes = [axes]
        for ax, (ep, tot, col, _per) in zip(axes, snaps):
            col = np.asarray(col)
            ec = ["#d62728" if col[u] == col[v] else "#cccccc"
                  for u, v in G.edges()]
            ew = [2.2 if col[u] == col[v] else 0.6 for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color=ec, width=ew)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=110,
                                   node_color=palette[col % len(palette)])
            ax.set_title(f"epoch {ep}", fontsize=10)   # titleless: panel id only
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUT / "mlnn_coloring.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # (c) learnable-A: true vs learned adjacency
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    for ax, M, ttl in [(axes[0], learn["A_true"], "true adjacency"),
                       (axes[1], learn["A_learned"], "learned $A_\\theta$")]:
        im = ax.imshow(M, cmap="magma", vmin=0, vmax=1)
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    fig.savefig(OUT / "learned_accessibility.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    main()
