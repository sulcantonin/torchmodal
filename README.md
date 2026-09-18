![](https://github.com/sulcantonin/torchmodal/blob/main/misc/torchmodal.png)

[![PyPI](https://img.shields.io/pypi/v/torchmodal)](https://pypi.org/project/torchmodal/)
[![CI](https://github.com/sulcantonin/torchmodal/actions/workflows/ci.yml/badge.svg)](https://github.com/sulcantonin/torchmodal/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/torchmodal)](https://pypi.org/project/torchmodal/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22825059.svg)](https://doi.org/10.5281/zenodo.22825059)
[![codecov](https://codecov.io/gh/sulcantonin/torchmodal/branch/main/graph/badge.svg)](https://codecov.io/gh/sulcantonin/torchmodal)

**Differentiable Modal Logic for PyTorch**

torchmodal makes the modal operators **□ (necessity)** and **♢ (possibility)** into trainable
neural layers over a Kripke structure: every truth value is an interval `[L, U]` that
**provably brackets** the crisp modal-logic answer, and the accessibility relation between
worlds can be **learned by gradient descent** instead of being specified.

> 📣 **Oral presentation at [NeSy 2026](https://openreview.net/pdf?id=uLOdtBm0Cx)** — the 20th
> Conference on Neurosymbolic Learning and Reasoning. *Modal Logic Neural Networks*,
> Antonin Sulc (Lawrence Berkeley National Laboratory) and Noor Naddour (The University of
> Queensland). PMLR vol. 284. **[Read the paper →](https://openreview.net/pdf?id=uLOdtBm0Cx)**

## A classic epistemic puzzle, in ten lines

Three children have muddy foreheads. Each sees the others but not themselves. Their father
says "at least one of you is muddy", then asks repeatedly whether anyone knows. Nobody does —
until the third round, when all three suddenly do. Nothing new is ever observed; the only
information is that nobody else could answer.

```python
import torch
from itertools import product
from torchmodal import functional as F

worlds = list(product([0, 1], repeat=3))          # 3 children, muddy or not
here = worlds.index((1, 1, 1))                    # all three really are muddy

for rnd in range(3):                              # each round rules out more worlds
    alive = torch.tensor([float(sum(w) > rnd) for w in worlds])
    # child 0 sees the others but not itself:
    A = torch.tensor([[float(w[1:] == v[1:]) for v in worlds] for w in worlds])
    muddy = torch.tensor([[float(w[0])] * 2 for w in worlds])
    L, U = F.necessity(muddy, A * alive, tau=0.05)[here]
    print(f"round {rnd + 1}: child 0 knows it is muddy -> [{L:.3f}, {U:.3f}]")
```

```
round 1: child 0 knows it is muddy -> [0.000, 0.000]
round 2: child 0 knows it is muddy -> [0.000, 0.000]
round 3: child 0 knows it is muddy -> [0.920, 1.000]
```

The textbook answer — *no, no, yes* — falls out of the modal operator alone, and every
interval contains it. See [`examples/muddy_children.py`](examples/muddy_children.py) for the
full version, which checks all three children and asserts soundness at every round.

## What makes this different

| | learnable relation between worlds | sound bounds on the crisp answer | native □ / ♢ |
|---|:---:|:---:|:---:|
| **torchmodal (MLNN)** | ✅ learned `A_θ` | ✅ `L ≤ crisp ≤ U`, gap `τ·H(w)` | ✅ |
| [LNN](https://arxiv.org/abs/2006.13155) (Riegel et al. 2020) | — no world structure | ✅ `[L, U]` bounds | — propositional |
| [LTN](https://arxiv.org/abs/1606.04422) (Serafini & Garcez 2016) | — | — point-valued | — ∀/∃ over domains |
| [DeepProbLog](https://arxiv.org/abs/1805.10872) (Manhaeve et al. 2018) | — fixed program | — exact probabilities | — |
| [Semantic Loss](https://arxiv.org/abs/1711.11157) (Xu et al. 2018) | — | — scalar penalty | — propositional |
| [Scallop](https://arxiv.org/abs/2304.04812) (Li et al. 2023) | — fixed Datalog | ~ provenance-dependent | — |
| [SATNet](https://arxiv.org/abs/1905.12149) (Wang et al. 2019) | ✅ learned MAXSAT | — no bounds | — |
| [STLCG](https://arxiv.org/abs/2008.00097) (Leung et al. 2023) | — fixed time axis | — point-valued | ~ temporal only |

The combination in the first row is what is unusual: other systems either fix the relational
structure and reason exactly over it, or learn structure without bracketing anything. A
`SemanticLoss` baseline ships in this package so the comparison can be run rather than
argued — see [`examples/baseline_comparison.py`](examples/baseline_comparison.py).

## Soundness is a property you can check

Every operator's docstring states **which crisp operator it bounds, in which direction, and
what the gap is**. The gap is not a hand-wave — it is an exact, computable quantity:

```python
from torchmodal import functional as F

F.box_width_entropy(A, bounds, tau=0.1)   # τ·H(w): the exact width one □ level adds
```

This is `conv_pool(x, -x) - smooth_min(x)` identically (verified to 8.9e-16 in float64), it
is bounded by `τ·log n`, and it tells you three things at once: how loose this `□` is, how
many levels you can nest before the bound floors (`k* = ⌈1/(τ·H̄)⌉`), and how large a
contradiction can hide from `L_contra` without producing any gradient.

### Ask for a precision, not a temperature

Because the width is exactly computable, it can be inverted. Rather than guessing `τ` and
finding out afterwards how wrong the answer might be, state what you can tolerate:

```python
box = F.necessity(bounds, A, precision=0.05)   # bracket at most 0.05 wide — guaranteed
tau = F.auto_tau(A, target_width=0.05)         # or just get the temperature
```

The returned temperature is always *safe* — the realised width never exceeds the target.
Pass `prop_bounds=` to `auto_tau` for the exact mode, which bisects on the true width and
returns the largest `τ` that still meets it: 2.4×–3.7× larger than the closed form on a
random 12-world frame, and a larger `τ` means better-conditioned gradients.

### Is it satisfied, or just vacuous?

Every `□`-built quantity is **maximal on the empty relation** — an agent that sees nothing
vacuously knows everything. So a specification written only in `□` has a global optimum
that satisfies every axiom and coordinates nothing, and an `ℓ₁` sparsity penalty pushes
*toward* that optimum rather than against it.

```python
from torchmodal.diagnostics import vacuity_report

vacuity_report(lambda A: F.necessity(phi, A)[:, 0], A)
# {'observed_value': 0.10, 'vacuous_value': 0.82, 'vacuous': True,
#  'direction': 'maximal_when_empty', ...}
```

### Find the silently-dead term in your neurosymbolic loss

The characteristic failure of a differentiable logic is not an exception — it is a term
pinned to 0 or 1 whose gradient has vanished. It raises nothing; it just stops contributing
while everything else keeps training.

```python
from torchmodal.diagnostics import gradient_health

report = gradient_health(lambda: my_modal_term(A), {"A": A})
report["healthy"]   # False
report["issues"]    # ["term 'output.L' is dead: pinned at the floor (0.0)
                    #   with no gradient to any parameter"]
```

`gradient_health` splits `[L, U]` bounds into their two endpoints — the dead state of a box
neuron is `L = 0` *with* `U = 1`, which neither column reveals on its own — and attributes
gradients per endpoint. `assert_has_signal(...)` is the raising variant for tests. No other
neurosymbolic library ships one.

### Batched

`necessity`, `possibility` and `box_width_entropy` take any number of leading batch
dimensions — `(B, |W|, 2)` bounds against `(B, |W|, |W|)` relations — so training over a
dataset of Kripke models needs no Python loop. Results are bit-identical to looping.

## Installation

```bash
pip install torchmodal
```

Or from a checkout (recommended while a release is pending, since features land here first):

```bash
pip install -e .
```

See [CHANGELOG.md](CHANGELOG.md) for what each release contains.

## Quick Start

```python
import torch
import torchmodal
from torchmodal import nn, KripkeModel

# Create a 3-world Kripke model with learnable accessibility
model = KripkeModel(
    num_worlds=3,
    accessibility=nn.LearnableAccessibility(3, init_bias=-2.0),
    tau=0.1,
)

# Add propositions
model.add_proposition("safe", learnable=True)
model.add_proposition("online", learnable=False)

# Evaluate modal operators
A = model.get_accessibility()
box_safe = model.necessity("safe", A)       # □(safe) — necessarily safe
dia_online = model.possibility("online", A)  # ♢(online) — possibly online

# Compute contradiction loss
loss = model.contradiction_loss()
```

## Architecture

```
torchmodal/
├── __init__.py          # Public API
├── functional.py        # Stateless functional operators (like torch.nn.functional)
├── nn/
│   ├── operators.py     # SmoothMin, SmoothMax, ConvPool modules
│   ├── connectives.py   # Negation, Conjunction, Disjunction, Implication
│   ├── modal.py         # Necessity (□), Possibility (♢) neurons
│   └── accessibility.py # Fixed, Learnable, Metric, Attention accessibility relations
├── kripke.py            # KripkeModel, Proposition
├── losses.py            # ContradictionLoss, ModalLoss, SparsityLoss, CrystallizationLoss, SemanticLoss
├── inference.py         # Upward-downward bound propagation
├── systems.py           # EpistemicOperator, DoxasticOperator, TemporalOperator, MultiAgentKripke
└── utils.py             # Temperature annealing, accessibility builders, decoding
```

## Core Concepts

### Differentiable Kripke Semantics

A Kripke model M = ⟨W, R, V⟩ is realized as differentiable tensors:

- **W** (Worlds): A finite set of possible worlds — agents, time steps, or contexts
- **R** (Accessibility): A relation determining which worlds can "see" each other
- **V** (Valuation): Truth bounds [L, U] ⊆ [0, 1] for each proposition in each world

### Modal Operators

| Operator | Symbol | Semantics | Implementation |
|----------|--------|-----------|----------------|
| Necessity | □ | True in *all* accessible worlds | `smooth_min` over weighted implications |
| Possibility | ♢ | True in *some* accessible world | `smooth_max` over weighted conjunctions |
| Until | U | ϕ holds until ψ becomes true | backward DP sweep `U_t = ψ_t ∨ (ϕ_t ∧ U_{t+1})` — **total order only**, ignores its relation |
| Until (graph) | U | ϕ holds until ψ, over *any* relation | least fixpoint of `U = ψ ∨ (φ ∧ ♢U)`, Gödel connectives, annealed τ |
| Knowledge | K_a | Agent *a* knows ϕ | □ restricted to agent's row |
| Belief | B_a | Agent *a* believes ϕ | □ with non-reflexive access |
| Globally | G | ϕ at all future times | □ over temporal accessibility |
| Finally | F | ϕ at some future time | ♢ over temporal accessibility |

> Aggregators are named `smooth_min` / `smooth_max` (not `softmin` / `softmax`) to avoid
> confusion with the probability-normalizing `torch.softmax`; the old names remain as
> deprecated aliases.

**Top-k aggregation** is a parameter of the operators — `nn.Necessity(tau, top_k=k)`,
`nn.Possibility(tau, top_k=k)`, `functional.necessity(..., top_k=k)` — not of the
accessibility modules. Each endpoint keeps the `k` extreme *aggregation terms*
(the `k` smallest of `(1 − A) + L` for `L_□`, the `k` largest of `A + U − 1` for `U_♢`, …)
and aggregates only those, so the true min/max is always kept, the bounds stay sound,
the smooth endpoints are within `τ·log k` of the crisp value, and nothing depends on
`|W|`. Masking `A` by its `k` largest entries before aggregation (the
`top_k=` of the accessibility modules up to 0.2.0) was unsound and is deprecated — see the
CHANGELOG.

### Accessibility Relations

```python
# Fixed (deductive mode): enforce known rules
R = torchmodal.build_sudoku_accessibility(3)
access = nn.FixedAccessibility(R)

# Learnable direct matrix (small worlds)
access = nn.LearnableAccessibility(num_worlds=7, init_bias=-2.0)

# Metric learning (scales to 20,000+ worlds)
access = nn.MetricAccessibility(num_worlds=10000, embed_dim=64)

# Attention-based (rich per-world features, asymmetric relations)
access = nn.AttentionAccessibility(input_dim=384, num_heads=4)
A = access(features)  # features: (num_worlds, 384)

# Deliberately sparsified relation (a modelling choice: each world keeps only
# its k most accessible worlds, the rest become inaccessible). This is NOT an
# aggregation optimisation — for that use top_k= on the operators.
access = nn.MetricAccessibility(num_worlds=10000, embed_dim=64, sparsify=8)
```

### Loss Functions

```python
# Combined modal loss: L_total = L_task + β * L_contra
criterion = torchmodal.ModalLoss(beta=0.3)
loss = criterion(task_loss, model.all_bounds())

# Sparsity regularization on accessibility
sparse_loss = torchmodal.SparsityLoss(lambda_sparse=0.05)

# Crystallization for SAT mode (forces crisp 0/1 assignments)
crystal_loss = torchmodal.CrystallizationLoss()

# Semantic Loss baseline (Xu et al. 2018), incl. mutual-exclusion constraints
sem = torchmodal.SemanticLoss()
loss = sem.forward_mutual_exclusive(probs)  # "exactly one of k" per row
```

## Examples

### Run in Colab — no install

| Notebook | What it shows |
|---|---|
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sulcantonin/torchmodal/blob/main/examples/notebooks/01_muddy_children.ipynb) **Muddy children** | The classic epistemic puzzle recovered exactly, with the bounds shown to bracket the crisp answer and tighten as `τ → 0` |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sulcantonin/torchmodal/blob/main/examples/notebooks/02_temporal_epistemic.ipynb) **Temporal epistemic read-out** | `G`, `F` and `K` over a spacetime frame, the 2× cost of the `K∘G` composite, and `gradient_health` locating the nesting floor |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sulcantonin/torchmodal/blob/main/examples/notebooks/03_graph_coloring.ipynb) **Learning the constraint graph** | The hidden graph recovered from valid colourings alone — **edge-recovery AUC 1.000** |

### Scripts

All examples are self-contained scripts in [`examples/`](examples/) and can be run directly:

```bash
python examples/muddy_children.py
```

| Example | Modal Logic | Description |
|---------|-------------|-------------|
| [`muddy_children.py`](examples/muddy_children.py) | K_a | The classic epistemic puzzle, recovered exactly, with soundness asserted each round |
| [`sudoku.py`](examples/sudoku.py) | □, CSP | 4x4 Sudoku via modal contradiction + crystallization |
| [`temporal_epistemic.py`](examples/temporal_epistemic.py) | K, G, F, K∘G | Learns epistemic accessibility to resolve contradictions |
| [`epistemic_trust.py`](examples/epistemic_trust.py) | K_a | Trust learning from promise-keeping behavior |
| [`doxastic_belief.py`](examples/doxastic_belief.py) | B_a | Belief calibration and hallucination detection |
| [`temporal_causal.py`](examples/temporal_causal.py) | □(cause → crash) | Root cause analysis in event traces |
| [`deontic_boundary.py`](examples/deontic_boundary.py) | O, P | Normative boundary learning (spoofing detection) |
| [`trust_erosion.py`](examples/trust_erosion.py) | Temporal + Deontic | Retroactive lie detection collapses trust |
| [`dialect_classification.py`](examples/dialect_classification.py) | □, ♢ thresholds | OOD detection — 89% Neutral recall trained only on AmE/BrE |
| [`axiom_ablation.py`](examples/axiom_ablation.py) | T, 4, B axioms | Effect of reflexivity/transitivity/symmetry on structure learning |
| [`scalability_ring.py`](examples/scalability_ring.py) | □, ♢ | Ring structure recovery with tau/top-k/learnable ablation |
| [`graph_coloring_benchmark.py`](examples/graph_coloring_benchmark.py) | ⋀_c(p_c → ¬♢p_c) | 12-solver comparison on planted-colourable graphs + inductive constraint-graph recovery (edge AUC 1.0) |
| [`sudoku_benchmark.py`](examples/sudoku_benchmark.py) | □, CSP | Sudoku solver benchmark (peer-graph special case of colouring) |
| [`baseline_comparison.py`](examples/baseline_comparison.py) | — | Side-by-side differentiable baselines (Semantic Loss, soft non-modal penalty) |
| [`MLNN_AccesbilityScalabilityAblation.ipynb`](examples/MLNN_AccesbilityScalabilityAblation.ipynb) | □, ♢ | Dense vs. metric accessibility sweep, N = 20 → 20,000 worlds on one GPU |

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
| `a → b` | `L_b ← L_φ + L_a − 1` (modus ponens; no modus tollens) |
| `□ϕ` | lower only: `L_ϕ[w'] ← max_w (L_φ[w] − 1 + A[w,w'])` |
| `♢ϕ` | upper only: `U_ϕ[w'] ← min_w (U_φ[w] + 1 − A[w,w'])` |
| `ϕ U ψ` | none — the backward DP couples every time step |

`□` upper and `♢` lower are not inverted: they bound an aggregate without saying
which neighbour realises it, so no canonical per-world constraint exists. The
two passes are **iterated** to `convergence_threshold`, not run once — a
downward update can stale a sibling formula that shares a leaf — and a
`RuntimeWarning` is raised if `max_iterations` is exhausted first.

## Two Learning Modes

| Mode | Fixed | Learned | Use Case |
|------|-------|---------|----------|
| **Deductive** | Accessibility R | Propositions V | POS guardrails, Sudoku, OOD detection |
| **Inductive** | Propositions V | Accessibility A_θ | Trust learning, social structure discovery |

## Limitations

These are measured properties of the implementation, not speculation. Each has a regression
test in [`tests/test_traps.py`](tests/test_traps.py) so it cannot silently change.

- **`conv_pool` is not monotone.** Its derivative `w_k·(1 − (x_k − f)/τ)` goes negative once
  `x_k − f > τ`, so raising a term that is already far above the pooled value *lowers* the
  result. Soundness is unaffected, but the tempting argument "the box neuron is monotone in
  `A`, therefore the bound is sound" is **not available** — the correct route is monotonicity
  of the hard `min` plus the one-sided enclosure.

- **`contradiction` has a dead zone after a modal neuron.** It is identically zero, with zero
  gradient, until the bound crossing exceeds the box width `τ·H(w)` — exactly 0.1792 for a
  fan-in of 6 at `τ = 0.1`. Do not rely on `L_contra` as the *sole* guard against a degenerate
  optimum; anneal `τ`, or pair it with `gradient_health`.

- **Each modal level costs `τ·H(w)` of interval width.** On a densely connected frame this is
  `τ·log|W|`, which is not negligible: with `τ = 0.1` and 8 fully-connected worlds, a nest of
  necessities floors at depth 5 and the lower bound is then dead. Compute the budget with
  `box_width_entropy` rather than assuming it. `MultiAgentKripke.K_G` / `K_F` are *two*
  levels and consume it twice as fast.

- **`functional.until` ignores its accessibility relation.** It is correct for a total order
  (consecutive time steps) and only for that: `until(φ, ψ, A)` is bit-identical for any `A`,
  no gradient flows into the relation, and cutting an edge changes nothing. Its Łukasiewicz
  sweep also loses `1 − L_φ` per step, flooring the lower bound over a long horizon. Use
  `until_graph` for an arbitrary or learned relation.

- **`until_graph(quantifier="box")` is sound only on a serial frame.** A dead end makes `□U`
  vacuously true, so a path that simply stops satisfies the formula. Prefer the default
  `"diamond"` (EU) unless every world is known to have a successor — or enforce seriality
  with `AxiomRegularization(seriality=...)`.

- **Only two of the four bound endpoints are monotone in `A`.** `necessity.L` and
  `possibility.U` are; the two `conv_pool` endpoints are not. A monotonicity argument is
  available only for the first two — see `torchmodal.diagnostics.MONOTONICITY`.

- **`until` and `until_graph` are not batched.** The modal operators are; these two still
  take one model at a time.

## Support

Questions, bugs and feature requests all go to
[issues](https://github.com/sulcantonin/torchmodal/issues); see
[SUPPORT.md](SUPPORT.md) for what makes a report actionable. If a constraint
"has no effect", run `torchmodal.diagnostics.gradient_health` on it first —
that is usually the whole diagnosis. Several behaviours that look like bugs
are documented traps, listed under [Limitations](#limitations).

## Citation

If you use torchmodal in your research, please cite:

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

To cite **the software** rather than the paper, use the Zenodo concept DOI
[`10.5281/zenodo.22825059`](https://doi.org/10.5281/zenodo.22825059), which always resolves to the latest
release; each release also has its own version DOI.

The proceedings version is the one to cite;
[arXiv:2512.03491](https://arxiv.org/abs/2512.03491)
([doi:10.48550/arXiv.2512.03491](https://doi.org/10.48550/arXiv.2512.03491))
remains available as a secondary identifier. GitHub's *Cite this repository* button reads
[`CITATION.cff`](CITATION.cff).

## License

MIT

## Authors

- [Antonin Sulc](https://sulcantonin.github.io) — Lawrence Berkeley National Laboratory
- Noor Naddour — The University of Queensland

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and the
[good first issues](https://github.com/sulcantonin/torchmodal/labels/good%20first%20issue).

## Media
- Substack https://open.substack.com/pub/sulcantonin/p/the-architecture-of-trust-in-agents?r=2p2sn8&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true
