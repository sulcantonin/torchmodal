<p align="center">
  <img src="https://github.com/sulcantonin/torchmodal/blob/main/misc/torchmodal.png" alt="torchmodal" width="640">
</p>

<p align="center">
  <a href="https://pypi.org/project/torchmodal/"><img src="https://img.shields.io/pypi/v/torchmodal" alt="PyPI"></a>
  <a href="https://pypi.org/project/torchmodal/"><img src="https://img.shields.io/pypi/pyversions/torchmodal" alt="Python"></a>
  <a href="https://github.com/sulcantonin/torchmodal/actions/workflows/ci.yml"><img src="https://github.com/sulcantonin/torchmodal/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/sulcantonin/torchmodal"><img src="https://codecov.io/gh/sulcantonin/torchmodal/branch/main/graph/badge.svg" alt="codecov"></a>
  <a href="https://sulcantonin.github.io/torchmodal/"><img src="https://img.shields.io/badge/docs-mkdocs-blue" alt="Docs"></a>
  <a href="https://doi.org/10.5281/zenodo.22825059"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.22825059.svg" alt="DOI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

<h1 align="center">torchmodal</h1>
<p align="center"><strong>Differentiable modal logic for PyTorch.</strong></p>

torchmodal turns the modal operators **□ (necessity)** and **♢ (possibility)** into trainable
layers over a Kripke structure. Every truth value is an interval `[L, U]` that **provably
brackets** the crisp modal-logic answer, and the accessibility relation between worlds can be
**learned by gradient descent** instead of being specified.

> 📣 **Oral at [NeSy 2026](https://openreview.net/pdf?id=uLOdtBm0Cx)**, the 20th Conference on
> Neurosymbolic Learning and Reasoning. *Modal Logic Neural Networks*, Antonin Sulc (Lawrence
> Berkeley National Laboratory) and Noor Naddour (The University of Queensland). PMLR vol. 284.
> **[Read the paper →](https://openreview.net/pdf?id=uLOdtBm0Cx)**

## Contents

- [Installation](#installation)
- [A classic epistemic puzzle, in ten lines](#a-classic-epistemic-puzzle-in-ten-lines)
- [Quick start](#quick-start)
- [What makes this different](#what-makes-this-different)
- [Soundness is a property you can check](#soundness-is-a-property-you-can-check)
- [Core concepts](#core-concepts)
- [Formula-graph inference](#formula-graph-inference)
- [From a learned relation to a certificate](#from-a-learned-relation-to-a-certificate)
- [Examples](#examples)
- [Limitations](#limitations)
- [Documentation and support](#documentation-and-support)
- [Citation](#citation)
- [License](#license)

## Installation

```bash
pip install torchmodal
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0. From a checkout, `pip install -e ".[dev]"` adds the
test and lint tooling; see [CHANGELOG.md](CHANGELOG.md) for what each release contains.

## A classic epistemic puzzle, in ten lines

Three children have muddy foreheads. Each sees the others but not themselves. Their father
says "at least one of you is muddy", then asks repeatedly whether anyone knows. Nobody does,
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

The textbook answer, *no, no, yes*, falls out of the modal operator alone, and every interval
contains it. [`examples/muddy_children.py`](examples/muddy_children.py) checks all three
children and asserts soundness at every round.

## Quick start

A `KripkeModel` bundles worlds, propositions and an accessibility relation. The objective is
the **contradiction** between what a proposition asserts and what inference derives for it.

```python
import torch
from torchmodal import KripkeModel, nn

# Three worlds, learnable accessibility, a prior of distrust between them
model = KripkeModel(
    num_worlds=3,
    accessibility=nn.LearnableAccessibility(3, init_bias=-2.0),
    tau=0.1,
)
model.add_proposition("safe", learnable=True)
model.add_proposition("online", learnable=False)
model.get_proposition("online").set_bounds(
    torch.tensor([[0.9, 1.0], [0.0, 0.1], [0.5, 0.8]])
)

A = model.get_accessibility()
box_safe = model.necessity("safe", A)       # □safe  — safe in every accessible world
dia_online = model.possibility("online", A) # ♢online — online in some accessible world

# On a reflexive frame □safe bounds safe itself: the two intervals must
# intersect. Their failure to do so is the contradiction loss, differentiable
# in both the proposition and the relation.
loss = model.contradiction_loss({"safe": box_safe})
loss.backward()
```

For the functional API, `F.necessity(bounds, A)` and `F.possibility(bounds, A)` take
`(|W|, 2)` bounds (or `(|W|,)` point values) and a `(|W|, |W|)` relation and return bounds of
the same shape. Both accept a leading batch dimension.

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
argued; see [`examples/baseline_comparison.py`](examples/baseline_comparison.py).

## Soundness is a property you can check

Every operator's docstring states **which crisp operator it bounds, in which direction, and
what the gap is**. The gap is an exact, computable quantity:

```python
from torchmodal import functional as F

F.box_width_entropy(A, bounds, tau=0.1)   # τ·H(w): the exact width one □ level adds
```

This is `conv_pool(x, -x) - smooth_min(x)` identically (verified to 8.9e-16 in float64), it
is bounded by `τ·log n`, and it tells you three things at once: how loose this `□` is, how
many levels you can nest before the bound floors (`k* = ⌈1/(τ·H̄)⌉`), and how large a
contradiction can hide from `L_contra` without producing any gradient.

### Ask for a precision, not a temperature

Because the width is exactly computable, it can be inverted. State what you can tolerate
instead of guessing `τ`:

```python
box = F.necessity(bounds, A, precision=0.05)   # bracket at most 0.05 wide, guaranteed
tau = F.auto_tau(A, target_width=0.05)         # or just get the temperature
```

The returned temperature is always *safe*: the realised width never exceeds the target.
Pass `prop_bounds=` to `auto_tau` for the exact mode, which bisects on the true width and
returns the largest `τ` that still meets it, 2.4×–3.7× larger than the closed form on a
random 12-world frame. A larger `τ` means better-conditioned gradients.

### Is it satisfied, or just vacuous?

Every `□`-built quantity is **maximal on the empty relation**: an agent that sees nothing
vacuously knows everything. A specification written only in `□` therefore has a global
optimum that satisfies every axiom and coordinates nothing, and an `ℓ₁` sparsity penalty
pushes *toward* that optimum rather than against it.

```python
from torchmodal.diagnostics import vacuity_report

vacuity_report(lambda A: F.necessity(phi, A)[:, 0], A)
# {'observed_value': 0.10, 'vacuous_value': 0.82, 'vacuous': True,
#  'direction': 'maximal_when_empty', ...}
```

### Find the silently dead term in your loss

The characteristic failure of a differentiable logic is not an exception. It is a term pinned
to 0 or 1 whose gradient has vanished; it raises nothing and simply stops contributing.

```python
from torchmodal.diagnostics import gradient_health

report = gradient_health(lambda: my_modal_term(A), {"A": A})
report["healthy"]   # False
report["issues"]    # ["term 'output.L' is dead: pinned at the floor (0.0)
                    #   with no gradient to any parameter"]
```

`gradient_health` splits `[L, U]` bounds into their two endpoints (the dead state of a box
neuron is `L = 0` *with* `U = 1`, which neither column reveals on its own) and attributes
gradients per endpoint. `assert_has_signal(...)` is the raising variant for tests.

## Core concepts

A Kripke model `M = ⟨W, R, V⟩` is realised as tensors:

- **W** (worlds): a finite set of agents, time steps, or contexts
- **R** (accessibility): a `(|W|, |W|)` relation in `[0, 1]`, fixed or learned
- **V** (valuation): truth bounds `[L, U] ⊆ [0, 1]` per proposition and world

### Operators

| Operator | Symbol | Semantics | Implementation |
|----------|--------|-----------|----------------|
| Necessity | □ | true in *all* accessible worlds | `smooth_min` over weighted implications |
| Possibility | ♢ | true in *some* accessible world | `smooth_max` over weighted conjunctions |
| Until | U | ϕ holds until ψ, along a total order | backward DP `U_t = ψ_t ∨ (ϕ_t ∧ U_{t+1})`; ignores its relation |
| Until (graph) | U | ϕ holds until ψ, over *any* relation | least fixpoint of `U = ψ ∨ (φ ∧ ♢U)`, Gödel connectives, annealed τ |
| CTL | EX AX EF EG AF AG EU AU | path quantifiers | `torchmodal.fixpoint`, soft or exact mode |
| Knowledge | K_a, E_G, D_G, C_G | single-agent and group knowledge | `EpistemicOperator`, `torchmodal.epistemic` |
| Belief | B_a | agent *a* believes ϕ | □ with non-reflexive access |
| Globally / Finally | G, F | at all / some future times | □ / ♢ over temporal accessibility |
| Announcement | [ψ] | public or group announcement | `F.announce`, `F.necessity_after`, `F.group_announce` |

All connectives are Łukasiewicz (`∧`: `max(0, a+b−1)`, `∨`: `min(1, a+b)`, `→`:
`min(1, 1−a+b)`, `¬`: `1−a`). Aggregators are named `smooth_min` / `smooth_max` rather than
`softmin` / `softmax` to avoid confusion with the probability-normalising `torch.softmax`.

**Top-k aggregation** is a parameter of the operators (`nn.Necessity(tau, top_k=k)`,
`F.necessity(..., top_k=k)`), not of the accessibility modules. Each endpoint keeps the `k`
extreme *aggregation terms* and aggregates only those, so the true min/max is always kept,
the bounds stay sound, and the gap is `τ·log k` independent of `|W|`.

**Exact mode.** Every modal operator takes `mode="exact"`: zero temperature, no gradient, no
gap. Train in soft mode, then round the relation and certify in exact mode.

### Accessibility relations

```python
R = torchmodal.build_sudoku_accessibility(3)
access = nn.FixedAccessibility(R)                         # known rules (deductive mode)
access = nn.LearnableAccessibility(7, init_bias=-2.0)     # direct logits, O(|W|²)
access = nn.MetricAccessibility(10_000, embed_dim=64)     # embeddings, O(|W|·d), symmetric
access = nn.AttentionAccessibility(input_dim=384)         # query/key scores, O(d²), asymmetric
A = access(features)                                      # features: (|W|, 384)

# A deliberately sparser frame (modelling choice, not an optimisation):
access = nn.MetricAccessibility(10_000, embed_dim=64, sparsify=8)
```

### Losses and regularisers

```python
criterion = torchmodal.ModalLoss(beta=0.3)          # L_task + β·L_contra
sparse = torchmodal.SparsityLoss(lambda_sparse=0.05) # ℓ₁ on the off-diagonal of A
crystal = torchmodal.CrystallizationLoss()           # push truth values to 0 / 1
axioms = torchmodal.AxiomRegularization(             # T, 4, B, D, 5 as soft penalties
    reflexivity=1.0, transitivity=0.5, seriality=0.5
)
sem = torchmodal.SemanticLoss()                      # Xu et al. 2018 baseline
```

`torchmodal.epistemic.frame_audit(A)` measures the same five axioms *after* training,
vacuity-corrected and against a shape-matched shuffled null, so a recovered structure can be
read as evidence rather than as an artefact of shape.

## Formula-graph inference

```python
from torchmodal import FormulaGraph, upward_downward

graph = FormulaGraph()
graph.add_atomic("p")
graph.add_atomic("q")
graph.add_conjunction("p_and_q", "p", "q")
graph.add_necessity("box_p_and_q", "p_and_q")

bounds = {
    "p": torch.tensor([[0.8, 1.0], [0.3, 0.5], [0.9, 1.0]]),
    "q": torch.tensor([[0.7, 0.9], [0.6, 0.8], [0.4, 0.6]]),
    "p_and_q": torch.tensor([[0.0, 1.0]] * 3),
    "box_p_and_q": torch.tensor([[0.0, 1.0]] * 3),
}
tightened = upward_downward(graph, bounds, torch.eye(3), tau=0.1)
```

The upward pass evaluates every node type. The downward pass inverts each connective on
both endpoints and each modal operator on the one endpoint that factorises per world:

| node | downward rule |
|---|---|
| `¬a` | both endpoints (exact; negation is an involution) |
| `a ∧ b` | both: `L_a ← L_φ`, `U_a ← U_φ + 1 − L_b` |
| `a ∨ b` | both: `L_a ← L_φ − U_b`, `U_a ← U_φ` |
| `a → b` | both: `L_b ← L_φ + L_a − 1` (modus ponens), `U_a ← 1 − L_φ + U_b` (modus tollens), and `U_b ← U_φ + U_a − 1`, `L_a ← 1 + L_b − U_φ` where `U_φ < 1` |
| `□ϕ` | lower only: `L_ϕ[w'] ← max_w (L_φ[w] − 1 + A[w,w'])` |
| `♢ϕ` | upper only: `U_ϕ[w'] ← min_w (U_φ[w] + 1 − A[w,w'])` |
| `ϕ U ψ` | none; the backward DP couples every time step |

`□` upper and `♢` lower are not inverted: they bound an aggregate without saying which
neighbour realises it, so no canonical per-world constraint exists. The two passes are
**iterated** to `convergence_threshold`, and a `RuntimeWarning` is raised if
`max_iterations` is exhausted first. The input dict is not modified.

## From a learned relation to a certificate

A graded relation is not a Kripke frame, so a statement about it is not a statement about
anything checkable. `torchmodal.verify` closes that loop:

```python
from torchmodal import verify

r = verify.round_and_certify(phi, A_learned, operator="ef")
r.verdicts      # ['PROVEN', 'UNDECIDED', 'REFUTED', ...] per world, from exact mode
r.witnesses[0]  # [0, 1, 3]: the path behind a PROVEN reachability verdict
r.margin        # how far each soft midpoint sat from the 0.5 boundary before rounding
smv = verify.to_smv(r.relation, {"p": phi.tolist()}, spec="AG (p)")   # for nuXmv / NuSMV
```

`UNDECIDED` is a first-class outcome: a library that says "I don't know" when it does not
know is more useful than one that rounds.

## Examples

### Run in Colab

| Notebook | What it shows |
|---|---|
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sulcantonin/torchmodal/blob/main/examples/notebooks/01_muddy_children.ipynb) **Muddy children** | The classic epistemic puzzle recovered exactly, with the bounds shown to bracket the crisp answer and tighten as `τ → 0` |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sulcantonin/torchmodal/blob/main/examples/notebooks/02_temporal_epistemic.ipynb) **Temporal epistemic read-out** | `G`, `F` and `K` over a spacetime frame, the 2× cost of the `K∘G` composite, and `gradient_health` locating the nesting floor |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sulcantonin/torchmodal/blob/main/examples/notebooks/03_graph_coloring.ipynb) **Learning the constraint graph** | The hidden graph recovered from valid colourings alone, edge-recovery AUC 1.000 |

### Scripts

Every script in [`examples/`](examples/) is self-contained and smoke-tested in CI:

```bash
python examples/muddy_children.py
```

| Example | Logic | Description |
|---------|-------|-------------|
| [`muddy_children.py`](examples/muddy_children.py) | K_a | The classic epistemic puzzle, with soundness asserted each round |
| [`sudoku.py`](examples/sudoku.py) | □, CSP | 4×4 Sudoku via modal contradiction + crystallisation |
| [`temporal_epistemic.py`](examples/temporal_epistemic.py) | K, G, F, K∘G | Learns epistemic accessibility to resolve contradictions |
| [`epistemic_trust.py`](examples/epistemic_trust.py) | K_a | Trust learning from promise-keeping behaviour |
| [`doxastic_belief.py`](examples/doxastic_belief.py) | B_a | Belief calibration and hallucination detection |
| [`temporal_causal.py`](examples/temporal_causal.py) | □(cause → crash) | Root-cause analysis in event traces |
| [`deontic_boundary.py`](examples/deontic_boundary.py) | O, P | Normative boundary learning (spoofing detection) |
| [`trust_erosion.py`](examples/trust_erosion.py) | temporal + deontic | Retroactive lie detection collapses trust |
| [`dialect_classification.py`](examples/dialect_classification.py) | □, ♢ thresholds | OOD detection: 89% Neutral recall trained only on AmE/BrE |
| [`axiom_ablation.py`](examples/axiom_ablation.py) | T, 4, B | Effect of reflexivity/transitivity/symmetry on structure learning |
| [`scalability_ring.py`](examples/scalability_ring.py) | □, ♢ | Ring structure recovery with τ / top-k / learnable ablation |
| [`graph_coloring_benchmark.py`](examples/graph_coloring_benchmark.py) | ⋀_c(p_c → ¬♢p_c) | 12-solver comparison on planted-colourable graphs + inductive constraint-graph recovery |
| [`sudoku_benchmark.py`](examples/sudoku_benchmark.py) | □, CSP | Sudoku solver benchmark (peer-graph special case of colouring) |
| [`baseline_comparison.py`](examples/baseline_comparison.py) | — | Side-by-side differentiable baselines (Semantic Loss, soft non-modal penalty) |
| [`MLNN_AccesbilityScalabilityAblation.ipynb`](examples/MLNN_AccesbilityScalabilityAblation.ipynb) | □, ♢ | Dense vs. metric accessibility sweep, N = 20 → 20,000 worlds on one GPU |

## Limitations

These are measured properties of the implementation, not speculation. Each has a regression
test in [`tests/test_traps.py`](tests/test_traps.py) so it cannot silently change; the full
list with numbers is in [docs/limitations.md](docs/limitations.md).

- **`conv_pool` is not monotone**, so the argument "the box neuron is monotone in `A`,
  therefore the bound is sound" is not available. The correct route is monotonicity of the
  hard `min` plus the one-sided enclosure. Only `necessity.L` and `possibility.U` are
  monotone in `A`; see `torchmodal.diagnostics.MONOTONICITY`.
- **`contradiction` has a dead zone after a modal neuron**: zero loss and zero gradient until
  the crossing exceeds the box width `τ·H(w)`. Do not rely on it as the sole guard against a
  degenerate optimum.
- **Each modal level costs `τ·H(w)` of width**; a nest of necessities on 8 fully connected
  worlds at `τ = 0.1` floors at depth 5. `MultiAgentKripke.K_G` / `K_F` are two levels.
- **`functional.until` ignores its relation** and is correct for a total order only. Use
  `until_graph` for an arbitrary or learned relation.
- **Universal operators (`AU`, `AX`, `AG`, `AF`, `until_graph(quantifier="box")`) are sound
  only on a serial frame.** `fixpoint` repairs dead ends with `serialize` by default.
- **The greatest-fixpoint cliff**: a soft `gfp` over a graded relation can collapse to 0.
  Round the relation and use `mode="exact"` for a trustworthy answer.
- **`until` and `until_graph` are not batched.**

## Documentation and support

- **API reference and guides**: <https://sulcantonin.github.io/torchmodal/>
- **Bugs and questions**: [GitHub issues](https://github.com/sulcantonin/torchmodal/issues);
  [SUPPORT.md](SUPPORT.md) says what makes a report actionable. If a constraint "has no
  effect", run `torchmodal.diagnostics.gradient_health` on it first.
- **Contributing**: see [CONTRIBUTING.md](CONTRIBUTING.md). The one rule: every operator's
  docstring states which crisp operator it bounds, in which direction, and what the gap is.
- **Changes**: [CHANGELOG.md](CHANGELOG.md).

## Citation

If you use torchmodal in your research, please cite the paper:

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

To cite **the software**, use the Zenodo concept DOI
[`10.5281/zenodo.22825059`](https://doi.org/10.5281/zenodo.22825059), which resolves to the
latest release; GitHub's *Cite this repository* button reads [`CITATION.cff`](CITATION.cff).
[arXiv:2512.03491](https://arxiv.org/abs/2512.03491) is a secondary identifier for the paper.

## License

MIT. See [LICENSE](LICENSE).

**Authors**: [Antonin Sulc](https://sulcantonin.github.io) (Lawrence Berkeley National
Laboratory) and Noor Naddour (The University of Queensland).
**Media**: [The architecture of trust in agents](https://open.substack.com/pub/sulcantonin/p/the-architecture-of-trust-in-agents) (Substack).
