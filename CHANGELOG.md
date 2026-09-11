# Changelog

All notable changes to `torchmodal` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.1] — 2026-09-11

### Fixed

- **Top-k masking of the accessibility matrix was unsound and vacuous at scale**
  (`nn/accessibility.py: top_k_mask`, applied inside `forward()` of
  `FixedAccessibility`, `LearnableAccessibility`, `MetricAccessibility` and
  `AttentionAccessibility`). The mask kept the `k` largest entries of each row of
  `A` and set the rest to 0; `functional.necessity` / `possibility` then
  aggregated the **full** row. Two defects:

  1. **Selection by the wrong quantity.** `L_□ϕ(w) = smooth_min_τ over w′ of
     (1 − A[w,w′]) + L_ϕ[w′]` depends on `L`, but neighbours were chosen by `A`
     alone, so a world just below the k-th access value whose `L` carries the
     violation was dropped and necessity was over-reported. With
     `A[w,·] = [0.90 0.88 0.86 0.84 0.83]`, `L = [0.9 0.9 0.9 0.9 0.0]`, `τ = 0.1`
     the terms are `[1.00 1.02 1.04 1.06 0.17]` (true min 0.170); the released
     top-4 gave `L_□ = 0.860, U_□ = 1.000` — Theorem 1 (`L ≤ min`) violated by
     0.69. The dual holds for ♢ (`A + U − 1`, max). This hides contradictions
     and lets a violated axiom train to zero loss.
  2. **Zeroing is not excluding.** A zeroed entry still entered the
     log-sum-exp with term `1 + L` (weight `exp(−(1+L)/τ) ≈ 4.5e−5` per world
     at `τ = 0.1`), and summed over `|W| − k` worlds that mass dominates once
     `|W|` is in the thousands. With `k = 8`, the same 8 real neighbours held
     fixed and ϕ false elsewhere, `[L_□, U_□]` drifted from `[0.403, 0.718]` at
     `|W| = 16` to `[0.027, 0.997]` at `|W| = 16384` (the neighbourhood alone
     gives `[0.405, 0.702]`). The effective fan-in was `|W|`, not `k`, so the
     advertised `τ·log k` gap and the "n ≤ k" / "O(k·|W|)" claims were not
     what the code computed; at 20k worlds every bound was vacuous and the
     gradient through the kept neighbours vanished. ♢ degrades the same way
     when `U = 1` at dropped worlds.

  **Fix.** Top-k now lives on the operators and selects by the *aggregation
  argument, per endpoint*: `functional.necessity(..., top_k=k)` keeps the `k`
  smallest of `(1 − A) + L` for `L_□` and of `(1 − A) + U` for `U_□`;
  `functional.possibility(..., top_k=k)` keeps the `k` largest of `A + L − 1`
  for `L_♢` and of `A + U − 1` for `U_♢` (`torch.topk(...).values`), and
  aggregates only those. The true extremum is always in the kept set, so the
  masked min/max is exact, `smooth_min` / `smooth_max` are within `τ·log k`,
  nothing depends on `|W|`, and gradients reach exactly the `k` selected
  entries of `A` per endpoint. On the cases above the fixed operator returns
  `L_□ = 0.170, U_□ = 0.171` and `[0.405, 0.702]` at every `|W|`. `top_k=None`
  (the default) is bit-for-bit the previous unmasked computation.
  Regression tests: `tests/test_masking.py`.

### Changed

- **`top_k` moved from the accessibility modules to the operators.**
  `nn.Necessity(tau, top_k=)`, `nn.Possibility(tau, top_k=)`,
  `functional.necessity` / `possibility(top_k=)`, and — threaded through —
  `KripkeModel(top_k=)`, `inference.upward_downward(top_k=)`,
  `EpistemicOperator`, `DoxasticOperator`, `TemporalOperator` and
  `MultiAgentKripke(top_k=)`. `functional.until` does not aggregate over the
  relation and has no `top_k`.
- **`top_k=` on `FixedAccessibility`, `LearnableAccessibility`,
  `MetricAccessibility` and `AttentionAccessibility` is deprecated.** It emits a
  `DeprecationWarning` explaining the above and is **ignored** — the relation is
  no longer zeroed (doing so silently would keep producing the unsound bounds).
  Move the argument to the operator (`nn.Necessity(top_k=k)`).
- **`sparsify=k` added to the four accessibility modules** for the case where
  a *sparsified relation* is wanted as a modelling choice: each world keeps
  only its `k` most accessible worlds and every other world becomes
  inaccessible (`A = 0`). This defines a different Kripke frame — it is not an
  aggregation optimisation and does not reduce the operators' cost. The
  operators then reason soundly about the sparsified frame.
  `nn.top_k_mask` remains exported as the underlying utility, with its docstring
  corrected (it previously claimed to reduce cost from `O(|W|²)` to `O(k·|W|)`).
- The downward □ / ♢ rules in `upward_downward` are unchanged and still run
  over the full `A`; they are sound for any sound parent bound, which the
  top-k upward bounds are. The comment claiming the rule was "safe under top-k
  masking" because masked pairs are inert now states this directly.
- `examples/scalability_ring.py`: its "Top-k Mask" sweep never went through the
  operators — the masked `A` itself is scored against the ring ground truth —
  so it is a relation sparsification and now says so
  (`LearnableAccessibility(sparsify=k)`); output is unchanged.

### Experiments to re-run

Every reported number produced with `top_k` set on an accessibility module
was computed with the defective masking and should be regenerated with the
operator-level `top_k`:

- the ring scaling runs (Table S3) at `k = 8`, 10k / 20k worlds — accuracy is
  `argmax(A)` and may survive, the reported bounds will change;
- any run with the C-MAPSS `τ = 0.03` setting;
- Appendix B.3's "n ≤ k throughout" / worst-case-gap-at-`n = k = 8` statements
  and the A.1 paragraph on τ shrinking with the fan-in should be re-read
  against the corrected operators.

`examples/MLNN_AccesbilityScalabilityAblation.ipynb` is self-contained and does
not use top-k masking; it is unaffected.

## [0.2.0] — 2026-08-09

This release completes the downward half of `inference.upward_downward`. Before
it, the downward pass covered three of the eight node types and silently skipped
disjunction and every modal operator, so the paper's stated inverse-update rules
did not all correspond to shipped code.

### Added

- **Downward inverse for `DISJUNCTION`** — previously the downward pass had no
  `DISJUNCTION` branch at all, so an asserted disjunction propagated nothing to
  its disjuncts. Adds the Łukasiewicz inverses for `parent = min(1, a + b)`:
  `L_a ← max(L_a, L_parent − U_b)` (clamped at 0) and `U_a ← min(U_a, U_parent)`,
  applied symmetrically. Asserting `a ∨ b` true with `b` known false now pins
  `a` true.
- **Downward inverses for `NECESSITY` and `POSSIBILITY`** — the downward pass
  previously skipped all modal nodes, on the grounds that inverting an
  aggregation over `Ã` has no canonical per-world factorisation. That is true of
  only one endpoint per operator. A universally quantified lower bound
  distributes over the neighbourhood and an existential upper bound caps every
  disjunct, giving two sound, canonical rules:

  ```
  □ϕ:  L_ϕ[w'] ← max( L_ϕ[w'],  max_w ( L_parent[w] − 1 + A[w,w'] ) )
  ♢ϕ:  U_ϕ[w'] ← min( U_ϕ[w'],  min_w ( U_parent[w] + 1 − A[w,w'] ) )
  ```

  The opposite directions (`□` upper, `♢` lower) constrain an aggregate without
  identifying which neighbour realises it, and remain un-inverted. Both rules are
  sound at any temperature, because `smooth_min` under-estimates `min` and
  `smooth_max` over-estimates `max`, and both are inert on masked pairs
  (`A = 0` contributes `L_parent − 1 ≤ 0` and `U_parent + 1 ≥ 1`), so they are
  safe under top-`k` sparsification.
- **Non-convergence `RuntimeWarning`** — `upward_downward` now warns when
  `max_iterations` is exhausted before `convergence_threshold` is met. The
  returned bounds are still sound (every update is a pure tightening), but they
  are not the fixed point, and this previously failed silently.

### Changed

- **Documented why the two passes are iterated rather than run once.** A single
  upward sweep is exact for the upward system alone, and a single downward sweep
  is exact given fixed parent bounds, but the joint fixed point generally needs
  more than one round: the downward pass tightens a leaf the upward pass has
  already consumed, leaving any sibling formula that shares that leaf stale.
  Since shared subformulae are exactly what the downward pass exists for, the
  single-sweep reading of the convergence result does not apply to the combined
  system. `inference.py`'s module docstring now states this and enumerates which
  endpoints each node type inverts.

### Fixed

- **Downward conjunction inverse in `inference.upward_downward`** — the previous
  update `U_child ← min(U_child, U_parent)` is only sound when the sibling's lower
  bound is 1 and could otherwise exclude a child's true value (e.g. `a = 0.9,
  b = 0.2` clamped `U_a` to `0.2`). Replaced with the general Łukasiewicz inverse
  `U_a ← min(U_a, U_parent + 1 − L_b)` (clamped to 1), and added the sound
  lower-bound update `L_child ← max(L_child, L_parent)`, so asserting a conjunction
  true now propagates truth to both conjuncts. Regression tests added.

### Notes for users

- The new rules only ever *tighten* bounds, so any bracket that was sound before
  remains sound. Two consequences are worth knowing about: inference on graphs
  containing disjunctions or modal nodes may now return strictly tighter bounds
  than 0.1.1 did, and an infeasible assertion over a modal node can now drive an
  atomic child to `L > U`. The latter is the intended contradiction signal — it
  is what `functional.contradiction` and `L_contra` consume — but it means leaves
  are no longer guaranteed to satisfy `L ≤ U` after a downward pass.
- Verified by a randomised soundness check over 300 models (every leaf pinned to
  a point value, every compound node left at `[0, 1]`): no downward rule excluded
  a true leaf value.

## [0.1.1] — 2026-06-09

### Added

- **`functional.until(phi, psi, R)`** — the temporal *Until* operator
  (`U_t = psi_t OR (phi_t AND U_{t+1})`), computed as a backward dynamic-programming
  sweep over the temporal accessibility relation. Exposed on formula graphs via
  `FormulaGraph.add_until(...)` and in `systems.TemporalOperator` (which now covers
  G, F, **and U**).
- **`losses.SemanticLoss`** — the Semantic Loss baseline (Xu et al., 2018) for
  comparing MLNNs against non-modal neurosymbolic constraints, including
  **`forward_mutual_exclusive(probs)`** for one-hot / mutual-exclusion constraints
  (e.g. "each node takes exactly one colour"). Note: the PyPI `0.1.0.6` build shipped
  a partial `SemanticLoss` without `forward_mutual_exclusive`; code calling that
  method against `0.1.0.x` fails with `AttributeError`.
- **`nn.AttentionAccessibility`** — attention-based accessibility head
  (`O(d^2)` parameters) for rich per-world features and asymmetric relations,
  alongside the existing Fixed / Learnable / Metric heads.
- **`FormulaGraph.is_acyclic()`** — cycle detection guarding the DAG invariant
  required by the convergence theorem; call before running inference on
  user-constructed graphs.
- `functional.contradiction(bounds, upper=None)` now accepts a separate upper-bound
  tensor in addition to the packed `[L, U]` form.
- **Examples**: `graph_coloring_benchmark.py` (12-method solver comparison on
  planted-colourable graphs + inductive constraint-graph recovery),
  `sudoku_benchmark.py`, `baseline_comparison.py`, `regen_coloring_figs.py`, and the
  `MLNN_AccesbilityScalabilityAblation.ipynb` notebook (dense-vs-metric accessibility
  sweep, N = 20 to 20,000 worlds).

### Changed

- **Aggregator naming**: `functional.softmin` / `functional.softmax` are renamed to
  **`smooth_min` / `smooth_max`** to avoid confusion with the standard
  probability-normalizing `torch.softmax` (used internally by `conv_pool`).
  The `nn.Softmin` / `nn.Softmax` modules are likewise renamed to
  **`nn.SmoothMin` / `nn.SmoothMax`**. The old names remain available as
  deprecated aliases (functions and module factories), so existing code keeps
  working.
- All example scripts now pin the local package onto `sys.path` before
  `import torchmodal`, so running them from a checkout always uses that checkout
  rather than a previously installed release.
- Expanded docstrings and module documentation throughout
  (`functional`, `inference`, `kripke`, `losses`, `systems`, `nn`).

## [0.1.0.x] — 2025

- Initial public releases: differentiable necessity / possibility neurons with
  sound `[L, U]` bounds, Łukasiewicz connectives, learnable accessibility
  (direct, metric), upward–downward inference, contradiction loss, epistemic /
  doxastic / temporal / multi-agent operators.
- Bugfix: necessity neuron `[L, U]` bound computation used `p` where the bound
  tensor was intended (thanks Noor Naddour).
