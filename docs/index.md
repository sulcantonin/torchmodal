# torchmodal

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
interval contains it. See [`examples/muddy_children.py`](https://github.com/sulcantonin/torchmodal/blob/main/examples/muddy_children.py) for the
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
argued — see [`examples/baseline_comparison.py`](https://github.com/sulcantonin/torchmodal/blob/main/examples/baseline_comparison.py).

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

See the [API reference](api/functional.md), the [limitations](https://github.com/sulcantonin/torchmodal/blob/main/limitations.md)
for the measured caveats, and [examples](https://github.com/sulcantonin/torchmodal/blob/main/examples.md).
