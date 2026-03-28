# CLAUDE.md — torchmodal

> **Modal Logical Neural Networks (MLNNs)**: a neurosymbolic framework integrating deep learning
> with Kripke-style modal logic (necessity □ and possibility ♢) over a set of possible worlds.
> Paper: *Modal Logical Neural Networks*, Antonin Sulc, Lawrence Berkeley National Lab (arXiv:2512.03491v2).

---

## What this repo is

`torchmodal` is a PyTorch implementation of the MLNN framework. It provides:

- Differentiable **□ (necessity)** and **♢ (possibility)** neurons operating over Kripke structures
- A **learnable accessibility relation** `A_θ` (fixed binary `R` or neurally parameterized)
- An **Upward–Downward inference algorithm** that converges to a fixed point (proven)
- A **contradiction loss** `L_contra` that penalizes states where lower bound `L > U` (upper bound)
- End-to-end differentiable training: `L_total = L_task + β * L_contra`
- Support for **temporal epistemic logic** via spacetime worlds `S = W × T`

The framework supports two learning modes:
- **(A) Deductive** — fixed accessibility `R`, learn propositional content (state representations)
- **(B) Inductive** — learnable `A_θ`, discover relational/logical structure from data

---

## Core concepts (read before touching code)

### Kripke model
A Kripke model is `M = ⟨W, R, V⟩`:
- `W` — finite set of possible worlds
- `R ⊆ W × W` — binary accessibility relation (or learned `A_θ ∈ [0,1]^{|W|×|W|}`)
- `V` — valuation: truth bounds `[L_{p,w}, U_{p,w}] ∈ [0,1]` for each proposition `p` in world `w`

Truth bounds are stored as tensors of shape `(|W|, 2)`.

### Modal operators
All operators use soft relaxations for differentiability (`τ = 0.1` default):

```
softmin_τ(x)       = -τ log Σ exp(-xᵢ/τ)     # sound lower bound on min(x)
softmax_τ(x)       =  τ log Σ exp( xᵢ/τ)     # sound upper bound on max(x)
conv_pool_τ(x, z)  = Σ wᵢxᵢ  where wᵢ = softmax(zᵢ/τ)   # weighted average
```

**□ (Necessity) neuron** — "ϕ must hold in ALL accessible worlds":
```
L_{□ϕ,w} = softmin_τ  ( (1 - Ã_{w,w'}) + L_{ϕ,w'} )   for w' ∈ W
U_{□ϕ,w} = conv_pool_τ( (1 - Ã_{w,w'}) + U_{ϕ,w'} )   for w' ∈ W
```

**♢ (Possibility) neuron** — "ϕ holds in SOME accessible world":
```
L_{♢ϕ,w} = conv_pool_τ( Ã_{w,w'} + L_{ϕ,w'} - 1 )     for w' ∈ W
U_{♢ϕ,w} = softmax_τ  ( Ã_{w,w'} + U_{ϕ,w'} - 1 )     for w' ∈ W
```

Modal duality is preserved: `♢ϕ ≡ ¬□¬ϕ` via `softmax(x) = 1 - softmin(1 - x)`.

As `τ → 0`, operators recover crisp classical modal semantics.

### Contradiction loss
```
L_contra = Σ_{w,ϕ} max(0, L_{ϕ,w} - U_{ϕ,w})
L_total  = L_task + β * L_contra
```
`β` balances statistical learning vs. logical consistency. Typical ranges:
- Safety-critical: `β ∈ [0.5, 1.0]`
- Accuracy-critical: `β ∈ [0.1, 0.3]`

### Accessibility relation
Two modes:

| Mode | Symbol | Use |
|------|--------|-----|
| Fixed | `R` | Known rules (grammar, Sudoku constraints, temporal flow) |
| Learnable | `A_θ` | Discover relational structure (trust, epistemic access) |

`A_θ` parameterizations:
- **Direct matrix**: `A_θ = sigmoid(logits)` — for small `|W|`, `O(|W|²)` params
- **Metric learning**: `A_θ(wᵢ, wⱼ) = σ(hᵢᵀ hⱼ)` — for large `|W|`, `O(|W| · d)` params, enables ~20k worlds on a single T4 GPU

Initialization strategies:
```python
# No prior
logits = N(0, 0.01)   # A_θ ≈ 0.5 uniformly

# Reflexivity prior (epistemic logic)
diag_logits = 2.0     # A_θ(w,w) ≈ 0.88

# Distrust prior (adversarial domains like Diplomacy)
logits = -2.0         # A_θ ≈ 0.12

# Identity prior (agents initially see only themselves)
logits = identity_matrix * large_positive_value
```

---

## Key files and structure

```
torchmodal/
├── torchmodal/
│   ├── core/
│   │   ├── worlds.py          # World sets W, truth bound tensors [L, U] ∈ [0,1]^{|W|×2}
│   │   ├── operators.py       # □ and ♢ neurons; softmin/softmax/conv_pool
│   │   ├── inference.py       # Upward–Downward algorithm
│   │   ├── loss.py            # L_contra, L_total
│   │   └── accessibility.py   # Fixed R and learnable A_θ (direct + metric)
│   ├── axioms/
│   │   ├── regularizers.py    # Axiom T (reflexivity), 4 (transitivity), B (symmetry)
│   │   └── grammar.py         # Example logical axiom sets
│   ├── temporal/
│   │   └── spacetime.py       # S = W × T, temporal operators G and F
│   └── models/
│       ├── guardrail.py       # Deductive guardrail (fixed R, learn content)
│       └── relational.py      # Inductive relational learner (learn A_θ)
├── experiments/
│   ├── pos_tagging/           # Case study 5.1: grammatical guardrailing
│   ├── dialect/               # Case study 5.2: logical indeterminacy / abstention
│   ├── epistemic/             # Case study 5.3: learning epistemic relations
│   ├── diplomacy/             # Case study 5.4: multi-agent trust (Diplomacy dataset)
│   ├── casino/                # Case study 5.5: deception detection (CaSiNo dataset)
│   ├── scalability/           # Case study 5.6: synthetic ring, N up to 20k worlds
│   └── sudoku/                # Case study 5.7: AI Escargot puzzle
├── tests/
├── CLAUDE.md                  # ← you are here
├── README.md
└── pyproject.toml
```

---

## Inference algorithm

The **Upward–Downward** algorithm (from LNN, extended for modal operators):

**Upward pass**: propagate truth upward from leaves to root
- Classical connectives (∧, ∨, →) use existing LNN operators
- Modal □ and ♢ neurons aggregate across worlds using `Ã` (masked accessibility)

**Downward pass**: propagate constraints from parent to children across worlds
```
# Necessity downward
L_{ϕ,w'} ← max(L_{ϕ,w'}, L_{□ϕ,w})    ∀w' s.t. Ã_{w,w'} > 0

# Possibility downward  
U_{ϕ,w'} ← min(U_{ϕ,w'}, U_{♢ϕ,w})    ∀w' s.t. Ã_{w,w'} > 0
```

**Cross-world contradiction propagation**:
1. Direct: `L_{ϕ,w} > U_{ϕ,w}` contributes to `L_contra`
2. Modal: high `L_{□ϕ,w}` + low `U_{ϕ,w'}` in accessible `w'` → gradient signal
3. Accessibility learning: gradient from cross-world tension flows back to `A_θ`, potentially severing links to resolve contradictions

**Convergence is guaranteed** (Theorem 2): all bound sequences are monotone and bounded, so the joint fixed point is reached by the monotone convergence theorem.

---

## Axiomatic regularization

Add regularizers to guide `A_θ` toward known modal systems:

```python
# Axiom T — Reflexivity: □ϕ → ϕ  (wRw for all w)
L_T = Σᵢ (1 - A_θ(wᵢ, wᵢ))²

# Axiom 4 — Transitivity: □ϕ → □□ϕ
L_4 = Σᵢⱼ max(0, (A_θ²)ᵢⱼ - A_θ(wᵢ, wⱼ))²

# Axiom B — Symmetry
L_S = Σᵢ<ⱼ (A_θ(wᵢ, wⱼ) - A_θ(wⱼ, wᵢ))²

# Combined: S4 system (reflexive + transitive)
L_total = L_task + β * L_contra + λ_T * L_T + λ_4 * L_4
```

Note: these are *soft* constraints, not hard guarantees. At high `λ`, the model approximates the axiom at the cost of task performance (see Table 7 in paper). Symmetry (B) is easiest to satisfy; Reflexivity (T) faces the strongest resistance from non-reflexive data topologies.

---

## Temporal epistemic logic

For multi-agent / temporal reasoning, worlds are spacetime points `S = W × T`:

```python
# Spacetime setup
states = [(agent, t) for agent in agents for t in timesteps]
# truth bounds: tensor of shape (|W| * |T|, 2)

# Operators via distinct accessibility matrices
K_a(ϕ)    # "Agent a knows ϕ" — □ neuron with epistemic A_θ^a
G(ϕ)      # "Always in future" — □ neuron with temporal A_θ^T
F(ϕ)      # "Eventually"      — ♢ neuron with temporal A_θ^T

# Composite: "Agent a will always know ϕ"
GK_a(ϕ)   # nested: temporal □ of (epistemic □ of ϕ)
```

---

## Common patterns

### Deductive guardrail (fixed rules)
```python
from torchmodal import KripkeModel, NecessityNeuron, ContradictionLoss

model = KripkeModel(worlds=["real", "pessimistic", "exploratory"])
model.set_accessibility(R_fixed)  # user-defined binary relation

# Axiom: □¬(DET_i ∧ VERB_{i+1})
axiom = NecessityNeuron(Not(And(det_prop, verb_prop)))

loss = L_task + beta * ContradictionLoss(model)
```

### Inductive relational learner (learn A_θ)
```python
from torchmodal import LearnableAccessibility, MetricAccessibility

# Small scale: direct matrix
A_theta = LearnableAccessibility(n_worlds=7, init="distrust")

# Large scale: metric learning (linear params, ~20k worlds feasible)
A_theta = MetricAccessibility(n_worlds=20000, embed_dim=64)

# Sparsity regularization (useful for Diplomacy-style tasks)
L_total = L_contra + lambda_sparse * A_theta.l1_norm()
```

### Sudoku / constraint satisfaction
```python
# 81 worlds (cells), fixed R = row ∪ col ∪ box adjacency
# 9 propositions {p_1, ..., p_9} per world
# Axiom: V_k (p_k → ¬♢p_k)  [value k must be absent in all accessible cells]
# Uniqueness: Σ_k L_{p_k,w} = 1.0
# Run 512 parallel universes, anneal τ from 2.0 → 0.1
```

---

## Performance & scaling

| `|W|` (worlds) | Mode | Params | Peak GPU mem | Notes |
|---|---|---|---|---|
| ≤ 200 | Dense or Metric | small | ~18 MB | both fine |
| 1,000 | Dense | 1M | ~56 MB | |
| 10,000 | Dense | 100M | ~3.8 GB | near limit |
| 10,000 | Metric (d=64) | 1.28M | ~2.7 GB | preferred |
| 20,000 | Dense | >400M | OOM | don't use |
| 20,000 | Metric (d=64) | 2.56M | ~10.7 GB | ✓ single T4 |

Rule of thumb: use `MetricAccessibility` for `|W| > 1000`.

For very large `|W|`, approximate nearest-neighbor search (LSH) can reduce aggregation from `O(|W|²)` to `O(|W| log |W|)` — not yet implemented but planned.

---

## Datasets used in experiments

| Experiment | Dataset | Task |
|---|---|---|
| 5.1 POS guardrail | Penn Treebank (PTB) | Sequence labeling + axiom enforcement |
| 5.2 Dialect abstention | Synthetic AmE/BrE corpus | 3-class classification with abstention |
| 5.3 Epistemic toy | Synthetic spacetime model | Learn single epistemic link |
| 5.4 Diplomacy trust | Diplomacy `in-the-wild` (Bakhtin et al. 2022) | Self-supervised trust recovery |
| 5.5 Negotiation deception | CaSiNo (Chawla et al. 2021) | Deception / reputational penalty |
| 5.6 Scalability ring | Synthetic directed ring | Recover known Kripke structure at scale |
| 5.7 Sudoku | AI Escargot (hardest known Sudoku) | Constraint satisfaction |

---

## Architecture for BiLSTM + MLNN guardrail (POS task)
```
embedding_dim  = 64
lstm_hidden    = 128  (bidirectional → 256 output)
worlds         = 3    ("real", "pessimistic", "exploratory")
A_θ            = 3×3 sigmoid-normalized learnable matrix
optimizer      = Adam, lr = 0.001
epochs         = 32
supervised_α   = 0.1
β sweep        = {0, 0.1, 0.3, 0.5, 0.9, 1.0}
```

---

## Known limitations

- **Axiom misspecification**: wrong axioms actively harm performance; the model enforces whatever rules you give it.
- **Overfitting of A_θ**: may learn spurious correlational links; mitigate with dropout on weights, Bayesian parameterization, or spectral regularization.
- **Discrete worlds only** (current): continuous-time extensions (Neural ODEs for `A_θ^T(t, t')`) are future work.
- **Soft guarantees**: axiomatic regularizers (T, 4, B) provide soft guidance, not hard constraints.

---

## Testing

```bash
pytest tests/                          # all tests
pytest tests/test_operators.py         # modal operator soundness
pytest tests/test_inference.py         # convergence / fixed-point
pytest tests/test_accessibility.py     # A_θ gradient flow
pytest tests/test_axioms.py            # regularizer correctness
```

Key invariants to assert in tests:
- `softmin_τ(x) ≤ min(x)` for all τ > 0
- `softmax_τ(x) ≥ max(x)` for all τ > 0
- `L_{□ϕ,w}` is non-decreasing in each `L_{ϕ,w'}` (Lemma 1)
- `U_{□ϕ,w}` is non-decreasing in each `U_{ϕ,w'}`
- Modal duality: `U_{♢ϕ,w} = 1 - softmin(1 - U_{□¬ϕ,w})`
- Bounds stay in `[0, 1]` after clipping
- Upward–Downward converges (loss monotonically non-increasing)

---

## Reference

```bibtex
@article{sulc2025mlnn,
  title   = {Modal Logical Neural Networks},
  author  = {Sulc, Antonin},
  journal = {arXiv preprint arXiv:2512.03491},
  year    = {2025},
  url     = {https://arxiv.org/abs/2512.03491}
}
```

Built on top of:
- **LNN** (Riegel et al. 2020, arXiv:2006.13155) — weighted real-valued logic, [L,U] bounds, contradiction loss
- **Kripke semantics** (Fagin et al. 1995, *Reasoning About Knowledge*)
- **CML** (Garcez et al. 2007) — predecessor: connectionist modal logic with fixed accessibility
