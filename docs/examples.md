# Examples

All examples are self-contained scripts in [`examples/`](https://github.com/sulcantonin/torchmodal/blob/main/examples/) and can be run directly:

```bash
python examples/sudoku.py
```

| Example | Modal Logic | Description |
|---------|-------------|-------------|
| [`muddy_children.py`](https://github.com/sulcantonin/torchmodal/blob/main/examples/muddy_children.py) | K_a | The classic epistemic puzzle, recovered exactly, with soundness asserted each round |
| [`sudoku.py`](https://github.com/sulcantonin/torchmodal/blob/main/examples/sudoku.py) | □, CSP | 4x4 Sudoku via modal contradiction + crystallization |
| [`temporal_epistemic.py`](https://github.com/sulcantonin/torchmodal/blob/main/examples/temporal_epistemic.py) | K, G, F, K∘G | Learns epistemic accessibility to resolve contradictions |
| [`epistemic_trust.py`](https://github.com/sulcantonin/torchmodal/blob/main/examples/epistemic_trust.py) | K_a | Trust learning from promise-keeping behavior |
| [`doxastic_belief.py`](https://github.com/sulcantonin/torchmodal/blob/main/examples/doxastic_belief.py) | B_a | Belief calibration and hallucination detection |
| [`temporal_causal.py`](https://github.com/sulcantonin/torchmodal/blob/main/examples/temporal_causal.py) | □(cause → crash) | Root cause analysis in event traces |
| [`deontic_boundary.py`](https://github.com/sulcantonin/torchmodal/blob/main/examples/deontic_boundary.py) | O, P | Normative boundary learning (spoofing detection) |
| [`trust_erosion.py`](https://github.com/sulcantonin/torchmodal/blob/main/examples/trust_erosion.py) | Temporal + Deontic | Retroactive lie detection collapses trust |
| [`dialect_classification.py`](https://github.com/sulcantonin/torchmodal/blob/main/examples/dialect_classification.py) | □, ♢ thresholds | OOD detection — 89% Neutral recall trained only on AmE/BrE |
| [`axiom_ablation.py`](https://github.com/sulcantonin/torchmodal/blob/main/examples/axiom_ablation.py) | T, 4, B axioms | Effect of reflexivity/transitivity/symmetry on structure learning |
| [`scalability_ring.py`](https://github.com/sulcantonin/torchmodal/blob/main/examples/scalability_ring.py) | □, ♢ | Ring structure recovery with tau/top-k/learnable ablation |
| [`graph_coloring_benchmark.py`](https://github.com/sulcantonin/torchmodal/blob/main/examples/graph_coloring_benchmark.py) | ⋀_c(p_c → ¬♢p_c) | 12-solver comparison on planted-colourable graphs + inductive constraint-graph recovery (edge AUC 1.0) |
| [`sudoku_benchmark.py`](https://github.com/sulcantonin/torchmodal/blob/main/examples/sudoku_benchmark.py) | □, CSP | Sudoku solver benchmark (peer-graph special case of colouring) |
| [`baseline_comparison.py`](https://github.com/sulcantonin/torchmodal/blob/main/examples/baseline_comparison.py) | — | Side-by-side differentiable baselines (Semantic Loss, soft non-modal penalty) |
| [`MLNN_AccesbilityScalabilityAblation.ipynb`](https://github.com/sulcantonin/torchmodal/blob/main/examples/MLNN_AccesbilityScalabilityAblation.ipynb) | □, ♢ | Dense vs. metric accessibility sweep, N = 20 → 20,000 worlds on one GPU |

### Epistemic Trust Learning (CaSiNo / Diplomacy)

```python
from torchmodal import MultiAgentKripke

# 7 agents (Diplomacy powers), 3 time steps
kripke = MultiAgentKripke(
    num_agents=7,
    num_steps=3,
    learnable_epistemic=True,
    init_bias=-2.0,
)

# Evaluate "agent knows claim is consistent over time"
K_G_claim = kripke.K_G(claim_bounds)

# Learn trust from contradiction minimization
A = kripke.get_epistemic_accessibility()
```

### Sudoku as Constraint Satisfaction

```python
import torchmodal
from torchmodal import KripkeModel, nn

# 81 worlds (cells), fixed Sudoku accessibility
R = torchmodal.build_sudoku_accessibility(3)
model = KripkeModel(
    num_worlds=81,
    accessibility=nn.FixedAccessibility(R),
)

# 9 propositions (digits)
for d in range(1, 10):
    model.add_proposition(f"d{d}", learnable=True)

# Train with contradiction loss + crystallization
contra_loss = torchmodal.ContradictionLoss(squared=True)
crystal_loss = torchmodal.CrystallizationLoss()
```

### POS Tagging with Grammatical Guardrails

```python
from torchmodal import nn, functional as F

# 3-world structure: Real, Pessimistic, Exploratory
box = nn.Necessity(tau=0.1)
access = nn.LearnableAccessibility(3)

# Enforce axiom: □¬(DET_i ∧ VERB_{i+1})
A = access()
det_bounds = ...   # from proposer network
verb_bounds = ...
conj = F.conjunction(det_bounds, verb_bounds)
neg_conj = F.negation(conj)
box_constraint = box(neg_conj, A)  # must be high (true)
```

### Formula Graph Inference

```python
from torchmodal import FormulaGraph, upward_downward

graph = FormulaGraph()
graph.add_atomic("p")
graph.add_atomic("q")
graph.add_conjunction("p_and_q", "p", "q")
graph.add_necessity("box_p_and_q", "p_and_q")

# Initialize bounds
bounds = {
    "p": torch.tensor([[0.8, 1.0], [0.3, 0.5], [0.9, 1.0]]),
    "q": torch.tensor([[0.7, 0.9], [0.6, 0.8], [0.4, 0.6]]),
    "p_and_q": torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
    "box_p_and_q": torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
}

# Run inference
A = torch.eye(3)  # reflexive accessibility
tightened = upward_downward(graph, bounds, A, tau=0.1)
```

The upward pass evaluates every node type. The downward pass inverts each
connective on both endpoints and each modal operator on the one endpoint that
factorises per world:

| node | downward rule |
|---|---|
| `¬a` | both endpoints (exact — negation is an involution) |
| `a ∧ b` | both: `L_a ← L_φ`, `U_a ← U_φ + 1 − L_b` |
| `a ∨ b` | both: `L_a ← L_φ − U_b`, `U_a ← U_φ` |
| `a → b` | both: `L_b ← L_φ + L_a − 1` (modus ponens), `U_a ← 1 − L_φ + U_b` (modus tollens), and `U_b ← U_φ + U_a − 1`, `L_a ← 1 + L_b − U_φ` where `U_φ < 1` |
| `□ϕ` | lower only: `L_ϕ[w'] ← max_w (L_φ[w] − 1 + A[w,w'])` |
| `♢ϕ` | upper only: `U_ϕ[w'] ← min_w (U_φ[w] + 1 − A[w,w'])` |
| `ϕ U ψ` | none — the backward DP couples every time step |

`□` upper and `♢` lower are not inverted: they bound an aggregate without saying
which neighbour realises it, so no canonical per-world constraint exists. The
two passes are **iterated** to `convergence_threshold`, not run once — a
downward update can stale a sibling formula that shares a leaf — and a
`RuntimeWarning` is raised if `max_iterations` is exhausted first. The input
dict is not modified; the tightened bounds are returned in a new one.


## Notebooks

Three of the examples are also available as one-click Colab notebooks — see the badges in the table above.
