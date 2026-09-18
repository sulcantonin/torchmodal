# CLAUDE.md — torchmodal

> **Modal Logic Neural Networks (MLNNs)**: a neurosymbolic framework integrating deep learning
> with Kripke-style modal logic (necessity □, possibility ♢, and until U) over a set of possible worlds.
> Paper: *Modal Logic Neural Networks*, Antonin Sulc (Lawrence Berkeley National Laboratory)
> and Noor Naddour (The University of Queensland). **Oral, NeSy 2026** (20th Conference on
> Neurosymbolic Learning and Reasoning), PMLR vol. 284.
> <https://openreview.net/pdf?id=uLOdtBm0Cx> · arXiv:2512.03491 (secondary).

---

## What this repo is

`torchmodal` is a PyTorch implementation of the MLNN framework. It provides:

- Differentiable **□ (necessity)**, **♢ (possibility)**, and **U (until)** neurons operating over Kripke structures
- A **learnable accessibility relation** `A_θ` (fixed binary `R`, direct logit matrix, metric-learning kernel, or attention-based)
- An **Upward–Downward inference algorithm** that converges to a fixed point (proven) with cycle detection
- A **contradiction loss** `L_contra` that penalizes states where lower bound `L > U` (upper bound)
- A **SemanticLoss** baseline (Xu et al., 2018) for comparison with non-modal NeSy approaches
- End-to-end differentiable training: `L_total = L_task + β * L_contra`
- Support for **temporal epistemic logic** via spacetime worlds `S = W × T`

The framework supports two learning modes:
- **(A) Deductive** — fixed accessibility `R`, learn propositional content (state representations)
- **(B) Inductive** — learnable `A_θ`, discover relational/logical structure from data

### Relationship to existing NeSy systems

MLNN builds on a rich literature of neurosymbolic AI:

- **LNN** (Riegel et al., 2020): weighted real-valued logic with [L,U] bounds — MLNN extends this with Kripke semantics
- **DeepProbLog** (Manhaeve et al., 2018): combines neural networks with probabilistic logic programming
- **Semantic Loss** (Xu et al., 2018): differentiable loss for propositional constraints — see `SemanticLoss` class
- **Scallop** (Li et al., 2023): differentiable Datalog with provenance
- **NeurASP** (Yang et al., 2020): neural network integration with answer set programming
- **STLCG** (Leung et al., 2023): differentiable signal temporal logic — MLNN now includes the Until operator
- **LTN** (Serafini & Garcez, 2016): logic tensor networks with universal/existential quantification
- **CML** (Garcez et al., 2007): connectionist modal logic with fixed Kripke structures

The key distinction of MLNN is the **learnable accessibility relation** combined with **sound bounds** for modal operators over Kripke structures. Non-modal NeSy systems (DeepProbLog, Semantic Loss, Scallop) can encode propositional constraints but lack native support for cross-world reasoning (□, ♢). STLCG supports temporal logic but operates over fixed temporal structures rather than learnable accessibility relations.

---

## Core concepts (read before touching code)

### Kripke model
A Kripke model is `M = ⟨W, R, V⟩`:
- `W` — finite set of possible worlds
- `R ⊆ W × W` — binary accessibility relation (or learned `A_θ ∈ [0,1]^{|W|×|W|}`)
- `V` — valuation: truth bounds `[L_{p,w}, U_{p,w}] ∈ [0,1]` for each proposition `p` in world `w`

Truth bounds are stored as tensors of shape `(|W|, 2)`.

### Modal operators
All operators use smooth relaxations for differentiability (`τ = 0.1` default):

```
smooth_min_τ(x)    = -τ log Σ exp(-xᵢ/τ)     # sound lower bound on min(x)
smooth_max_τ(x)    =  τ log Σ exp( xᵢ/τ)     # sound upper bound on max(x)
conv_pool_τ(x, z)  = Σ wᵢxᵢ  where wᵢ = softmax(zᵢ/τ)   # weighted average
```

> **Naming convention**: The aggregation operators are called `smooth_min` / `smooth_max`
> (not `softmin` / `softmax`) to avoid confusion with the standard probability-normalization
> `torch.softmax`, which is used internally by `conv_pool`. Legacy aliases are provided.

**□ (Necessity) neuron** — "ϕ must hold in ALL accessible worlds":
```
L_{□ϕ,w} = smooth_min_τ  ( (1 - Ã_{w,w'}) + L_{ϕ,w'} )   for w' ∈ W
U_{□ϕ,w} = conv_pool_τ( (1 - Ã_{w,w'}) + U_{ϕ,w'} )   for w' ∈ W
```

**♢ (Possibility) neuron** — "ϕ holds in SOME accessible world":
```
L_{♢ϕ,w} = conv_pool_τ( Ã_{w,w'} + L_{ϕ,w'} - 1 )     for w' ∈ W
U_{♢ϕ,w} = smooth_max_τ  ( Ã_{w,w'} + U_{ϕ,w'} - 1 )     for w' ∈ W
```

**U (Until) operator** — "ϕ holds until ψ becomes true":
```
U_t = ψ_t ∨ (ϕ_t ∧ U_{t+1})     (backward DP sweep)
```

Modal duality is preserved: `♢ϕ ≡ ¬□¬ϕ` via `smooth_max(x) = 1 - smooth_min(1 - x)`.

As `τ → 0`, operators recover crisp classical modal semantics.

**Top-k aggregation lives on the operators** (`necessity(..., top_k=k)`,
`nn.Necessity(top_k=k)`, and threaded through `KripkeModel`, `upward_downward`,
`systems.*`). Each endpoint keeps the `k` extreme *aggregation terms* — the `k`
smallest of `(1 − Ã) + L` / `(1 − Ã) + U` for □, the `k` largest of `Ã + L − 1` /
`Ã + U − 1` for ♢ — and aggregates only those, so the true extremum is always kept,
Theorem 1 holds, the gap is `τ·log k`, and nothing depends on `|W|`. Never emulate
this by zeroing entries of `Ã` (the deprecated `top_k=` on the accessibility
modules): selecting by `Ã` alone can drop the world carrying the violation, and
zeroed entries still enter the log-sum-exp and drive the bounds to `[0, 1]` as
`|W|` grows. `sparsify=k` on the accessibility modules is a different thing — a
deliberately sparser Kripke frame (modelling choice), not an optimisation.

### Propositional connectives (Łukasiewicz fuzzy logic)

All Boolean connectives are explicitly defined using Łukasiewicz t-norms:

| Connective | Formula | Bounds computation |
|---|---|---|
| **Negation** ¬ϕ | `1 - x` | `[1-U, 1-L]` |
| **Conjunction** ϕ ∧ ψ | `max(0, a+b-1)` | `L = max(0, L_a+L_b-1)`, `U = min(U_a, U_b)` |
| **Disjunction** ϕ ∨ ψ | `min(1, a+b)` | `L = max(L_a, L_b)`, `U = min(1, U_a+U_b)` |
| **Implication** ϕ → ψ | `min(1, 1-a+b)` | `L = max(0, 1-U_a+L_b)`, `U = min(1, 1-L_a+U_b)` |

This choice of fuzzy semantics follows van Krieken, Acar & van Harmelen (AI 2022) — see that work for a thorough analysis of trade-offs between Łukasiewicz, Gödel, and product t-norms.

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

| Parameterization | Parameters | Best for |
|---|---|---|
| **Direct matrix** (`LearnableAccessibility`) | `O(\|W\|²)` | Small `\|W\|` ≤ 1000 |
| **Metric learning** (`MetricAccessibility`) | `O(\|W\| · d)` | Large `\|W\|`, learnable embeddings |
| **Attention** (`AttentionAccessibility`) | `O(d²)` | Rich per-world features, asymmetric relations |

---

## Key files and structure

```
torchmodal/
├── torchmodal/
│   ├── __init__.py           # Public API and version
│   ├── functional.py         # Stateless functions: smooth_min, smooth_max, conv_pool,
│   │                         #   negation, conjunction, disjunction, implication,
│   │                         #   necessity, possibility, until, contradiction
│   ├── kripke.py             # KripkeModel, Proposition — central data structures
│   ├── inference.py          # FormulaGraph, FormulaNode, upward_downward algorithm
│   ├── losses.py             # ContradictionLoss, ModalLoss, SparsityLoss,
│   │                         #   CrystallizationLoss, AxiomRegularization, SemanticLoss
│   ├── systems.py            # EpistemicOperator, DoxasticOperator, TemporalOperator,
│   │                         #   MultiAgentKripke
│   ├── utils.py              # Temperature annealing, accessibility builders, decoders
│   └── nn/
│       ├── __init__.py       # Subpackage exports
│       ├── operators.py      # SmoothMin, SmoothMax, ConvPool modules
│       ├── connectives.py    # Negation, Conjunction, Disjunction, Implication modules
│       ├── modal.py          # Necessity (□), Possibility (♢) neuron modules
│       └── accessibility.py  # FixedAccessibility, LearnableAccessibility,
│                             #   MetricAccessibility, AttentionAccessibility
├── tests/
│   ├── test_functional.py    # Operator soundness, gradient flow
│   ├── test_modules.py       # nn.Module wrappers, KripkeModel, losses
│   ├── test_inference.py     # Upward-downward convergence
│   └── test_systems.py       # Epistemic, temporal, multi-agent operators
├── examples/
│   ├── sudoku.py             # AI Escargot via contradiction minimization
│   ├── epistemic_trust.py    # CaSiNo negotiation trust learning
│   ├── dialect_classification.py  # OOD detection via modal abstention
│   ├── scalability_ring.py   # Synthetic ring scalability benchmark
│   ├── graph_coloring_benchmark.py # 12-solver comparison + inductive A_θ recovery
│   ├── sudoku_benchmark.py   # Sudoku solver benchmark (colouring special case)
│   ├── baseline_comparison.py      # Semantic Loss / non-modal baselines side-by-side
│   ├── MLNN_AccesbilityScalabilityAblation.ipynb  # dense-vs-metric sweep, N=20→20k
│   └── ...                   # Additional examples
├── CLAUDE.md                 # ← you are here
├── CHANGELOG.md              # Release history (current: 0.7.0)
├── README.md
├── LICENSE
└── pyproject.toml
```

> All example scripts insert their checkout root onto `sys.path` before
> `import torchmodal`, so running them from the repo always uses the local
> package — never a previously installed PyPI release. Keep this shim in any
> new example: a stale installed release can silently shadow the checkout
> (this once produced an invisible 0% baseline row in the colouring benchmark
> because `SemanticLoss.forward_mutual_exclusive` didn't exist in PyPI 0.1.0.x).
> Errors raised inside the benchmark's `with_timeout` wrapper aggregate into low
> solve rates without a traceback, so always surface per-call statuses when
> debugging an unexpectedly low row.

---

## Inference algorithm

The **Upward–Downward** algorithm (from LNN, extended for modal operators):

**Upward pass**: propagate truth upward from leaves to root
- Classical connectives (∧, ∨, →) use Łukasiewicz fuzzy logic
- Modal □ and ♢ neurons aggregate across worlds using `Ã` (masked accessibility)
- Until (U) uses backward dynamic programming

**Downward pass**: propagate constraints from parent to children across worlds

**Cycle detection**: `FormulaGraph.is_acyclic()` verifies the DAG invariant required by Theorem 2 (convergence). Call this before running inference on user-constructed graphs.

**Convergence is guaranteed** (Theorem 2): all bound sequences are monotone and bounded, so the joint fixed point is reached by the monotone convergence theorem.

---

## Testing

```bash
pytest tests/                          # all tests
pytest tests/test_functional.py        # operator soundness, gradient flow
pytest tests/test_modules.py           # nn.Module wrappers, KripkeModel
pytest tests/test_inference.py         # convergence / fixed-point
pytest tests/test_systems.py           # epistemic, temporal, multi-agent
```

Key invariants to assert in tests:
- `smooth_min_τ(x) ≤ min(x)` for all τ > 0
- `smooth_max_τ(x) ≥ max(x)` for all τ > 0
- Modal duality: `U_{♢ϕ,w} = 1 - smooth_min(1 - U_{□¬ϕ,w})`
- Bounds stay in `[0, 1]` after clipping
- Upward–Downward converges (loss monotonically non-increasing)
- `FormulaGraph.is_acyclic()` returns True for all valid graphs

---

## Reference

```bibtex
@inproceedings{sulc2026mlnn,
  title     = {Modal Logic Neural Networks},
  author    = {Sulc, Antonin and Naddour, Noor},
  booktitle = {Proceedings of the 20th Conference on Neurosymbolic
               Learning and Reasoning (NeSy 2026)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {284},
  year      = {2026},
  publisher = {PMLR},
  url       = {https://openreview.net/pdf?id=uLOdtBm0Cx},
  note      = {Oral presentation. arXiv:2512.03491}
}
```

Built on top of:
- **LNN** (Riegel et al. 2020, arXiv:2006.13155) — weighted real-valued logic, [L,U] bounds, contradiction loss
- **Kripke semantics** (Fagin et al. 1995, *Reasoning About Knowledge*)
- **CML** (Garcez et al. 2007) — predecessor: connectionist modal logic (note: CML does train end-to-end with backpropagation through Kripke models)
- **Semantic Loss** (Xu et al. 2018) — differentiable propositional constraint loss (provided as baseline)
- **STLCG** (Leung et al. 2023) — differentiable signal temporal logic including Until
- **van Krieken et al. 2022** — analysis of fuzzy operator semantics trade-offs
