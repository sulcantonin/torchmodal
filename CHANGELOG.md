# Changelog

All notable changes to `torchmodal` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.8.0] — 2026-09-18

Project health, aimed at what a reviewer checks first. No public API change.

### Changed

- **`mypy` is clean: 32 errors to 0, and CI no longer suppresses the gate.**
  The `mypy` step ran with `|| true`, so the job was green regardless — a
  suppressed gate reads as a smell even when the errors are benign, and it
  meant nothing was actually being checked.

  The errors were almost all the standard PyTorch typing friction:
  `nn.Module.__call__` is typed as returning `Any`, and attribute access on a
  registered buffer widens to `Union[Tensor, Module]`. Fixed by declaring
  buffer types at class level, annotating optional submodules before the
  branch that assigns them, giving `KripkeModel` a typed accessor for its
  `ModuleDict`, and casting where the torch stubs are genuinely lossy — not by
  adding blanket ignores. `warn_unused_ignores`, `warn_redundant_casts` and
  `no_implicit_reexport` are now on, so a stale suppression becomes an error
  rather than debt.

### Added

- **Coverage in CI, published.** `pytest-cov` was already a dev dependency and
  unused. Coverage now runs on every job, is written to the workflow summary,
  and is uploaded to Codecov with a README badge. Current: **88%** line
  coverage, 1586 statements.

- **Every example is smoke-tested in CI** (`scripts/run_examples.py`). All 16
  scripts run, not a chosen subset — an example that cannot be run in CI is
  one that will break without anyone noticing. A script that exits 0 but
  prints nothing counts as a failure, since each is meant to demonstrate a
  result. Full run is about four minutes in smoke mode.

  The runner **restores committed artefacts afterwards**. Several benchmarks
  write into `examples/coloring_results/`, `examples/sudoku_results/` and
  `examples/coloring_solve.json`, which are tracked and hold *full*-mode
  results (10 graphs per tier, 5 seeds, 2000 epochs); a smoke run would
  quietly overwrite them with far weaker numbers. This was found by doing
  exactly that during development.

- **Notebooks execute in CI** (`scripts/run_notebooks.py`). They carry Colab
  badges, so a reader's first contact with the library may be running one.

- **`SUPPORT.md` and issue templates** — bug report, feature request and
  question forms, with a config that points at the API reference and the
  limitations page. The bug template asks for the torch version and the
  evaluation mode, because several documented behaviours differ between torch
  2.8 and 2.14 and between `soft` and `exact`.

- **ORCID for Antonin Sulc** (`0000-0001-7767-778X`) in both the software and
  paper entries of `CITATION.cff`.

- The **Zenodo DOI** badge and citation metadata, which had been written on a
  branch whose pull request was already merged and so never reached `main`:
  concept DOI `10.5281/zenodo.22825059`.

### Fixed

- **`examples/regen_coloring_figs.py` had a hardcoded absolute path** to one
  machine's home directory (`/Users/asulc/PycharmProjects/...`), so it raised
  `PermissionError` for every other user — including any reviewer who ran it.
  The output directory now defaults to `examples/figures/` inside the checkout
  and is overridable with `MLNN_FIG_DIR`. Found by smoke-testing the examples,
  which is the point of doing so.


## [0.7.0] — 2026-09-18

### Added

- **A Zenodo DOI.** The GitHub–Zenodo integration is live and archived
  `v0.6.0`: concept DOI `10.5281/zenodo.22825059` (always resolves to the
  latest release) and version DOI `10.5281/zenodo.22825060`. Zenodo read the
  title, both authors and the licence from `CITATION.cff` rather than falling
  back to the repository description. Badge added to the README, and the
  concept DOI added to `CITATION.cff`'s software entry.

### Removed — **breaking**

- **`functional.until`'s `tau` argument is gone.** Passing it now raises
  `TypeError`; the migration is to delete it from the call.

  It never had any effect — the backward DP contains no smooth aggregation, so
  no temperature enters it — and it has emitted a `DeprecationWarning` since
  0.2.2. The notice named 0.4.0 as the removal version and we reached 0.6.0
  without acting, which left the library shipping a deprecation notice that
  was itself wrong. Removing the argument is the smaller error: a no-op
  parameter that *looks* like it controls the operator is exactly the kind of
  misleading surface this library exists to avoid.

  No caller in this repository passed it except the trap tests that pinned the
  deprecation, which are updated to pin the removal instead. If you want a
  temperature-controlled, relation-aware Until, that is `until_graph`.

### Added

- **SMV round-trip tests.** The exporter's structural checks could not catch
  the failure that matters most — well-formed SMV describing the *wrong*
  transition relation. The new tests parse the generated module back into a
  relation and a label set, and assert both reconstruct the input exactly;
  one goes further and re-runs `EF` on the reconstructed frame, requiring the
  same answer.

  This is still **not** a nuXmv round trip. No checker is installable here —
  neither nuXmv nor NuSMV is available through a package manager, and nuXmv
  requires registration and licence acceptance — so the gap noted in 0.6.0
  stands. What these tests remove is the exporter-correctness half of the
  risk; the semantics half still needs an external checker.


## [0.6.0] — 2026-09-18

Abstract inputs and certificates: the point at which "soundness is a property
you can check" becomes true end to end. Additive — the API fingerprint is
unchanged apart from `__all__` and `__version__`.

### Added

- **`functional.necessity_mts` / `possibility_mts`** — modal transition
  systems, carrying a `must` relation (required transitions) and a `may`
  relation (permitted ones), with well-formedness `must <= may` checked.

  The returned interval brackets the value on **every** Kripke frame lying
  between the two, so a conclusion proved on the abstraction holds for every
  concretisation. Verified by sampling: **0 of 400 concretisations escaped the
  interval**.

  Which relation each endpoint uses follows from the quantifier. Box is
  universal, so its lower bound quantifies over `may` (it must survive every
  transition that could exist) and its upper over `must`. Diamond is
  existential and swaps them. With `must == may` both reduce **exactly** to
  the single-relation operators, which is what makes this additive.

- **`torchmodal.verify`** —

  - `certify` returns `PROVEN` / `REFUTED` / **`UNDECIDED`** per world, the
    last being a first-class outcome rather than an error. An interval that
    straddles the decision boundary does not settle the question, and saying
    so is more useful than rounding.
  - `round_and_certify` thresholds a learned relation, re-evaluates it with
    `mode="exact"`, and returns the verdicts together with the rounded frame
    the certificate is actually *about* and `n_flipped`, how far that frame
    moved from the learned one.
  - `rounding_margin` — distance of each truth midpoint from 0.5. This is to
    the rounding step what `box_width_entropy` is to the modal step: it turns
    "is this certificate trustworthy?" into a number. A margin near zero means
    the crisp label is a coin flip.
  - `certificate_gap` — how often the soft rounding and the exact answer
    disagree. Zero means the soft model is already making the decisions the
    exact checker would, which is the condition under which training against
    the soft operators is safe.
  - `witness_path` — a shortest path backing a reachability verdict, or
    `None`, which is itself the evidence behind a refutation.
  - `to_smv` — export a rounded frame as a nuXmv / NuSMV module, with
    `DEFINE` predicates and an optional `CTLSPEC`. Dead ends are rejected with
    a pointer to `serialize`, since SMV requires a total transition relation.

### Notes

- **The SMV exporter is structurally validated only.** This package does not
  bundle nuXmv, and none was available when the exporter was written, so the
  generated module is checked for well-formedness but has **not** been run
  through a real model checker. It produces input for a tool you then run; it
  is not itself a verified oracle. A genuine round-trip test against nuXmv
  remains outstanding and is the single most valuable thing that could be
  added to this module.
- Soft interval evaluation through the MTS operators is a relaxation *of an
  abstraction*: only the outer enclosure survives, because the soft endpoints
  are not monotone in the relation. Use `mode="exact"` when the result is
  meant as a certificate.


## [0.5.0] — 2026-09-18

CTL model checking, and an evaluation mode with no temperature in it. Additive:
the API fingerprint is unchanged apart from `__all__` and `__version__`.

### Added

- **`mode="exact"` on `necessity` and `possibility`** — zero-temperature
  evaluation using the true extremum. The bracket is exact, the gap is 0, and
  there is no smoothing error. Two properties follow that soft mode does not
  have:

  - it is **monotone in `A`** on both endpoints, verified at **0 violations
    over 300 random perturbations** against 247 and 252 for the `conv_pool`
    endpoints in soft mode;
  - interval inputs propagate soundly through it, which is what lets the
    operators act as an abstract interpreter rather than only as a trainable
    relaxation.

  Train in soft mode, certify in exact mode.

- **`functional.serialize`** — adds a self-loop at every dead end. Several
  operators are sound only on a serial frame, because at a dead end a
  universal modality is *vacuously* satisfied and a computation that simply
  stops counts as success. This retires a limitation `until_graph` has carried
  since 0.2.2 with no way to fix it, and complements
  `AxiomRegularization(seriality=...)` from 0.3.0: the regulariser for
  learning, the helper for a guarantee.

- **`torchmodal.fixpoint`** — `lfp` and `gfp` combinators over an arbitrary
  step function, and all eight CTL operators built on them: `ex`, `ax`, `ef`,
  `eg`, `eu`, `af`, `ag`, `au`. This generalises `until_graph` (the lfp of EU)
  and `TemporalOperator.globally` (one box over a precomputed reachability
  matrix).

  **Validated against an independent crisp checker** written from the textbook
  set-based labelling definitions: exact agreement on all eight operators over
  320 randomly generated cyclic frames.

  Three design decisions, each forced by a measurement:

  - **A stop rule that terminates.** A strict tolerance does not — even at
    edge weight 1.0 the soft gfp iteration is still creeping at a 200-sweep
    cap. The combinators also stop on **rounding stabilisation**, when the
    crisp label implied by the bounds has not moved for `patience` sweeps,
    because the certificate is final long before the value settles.
    `FixpointResult` reports `n_iters`, `converged` and `stopped_by`, since
    all three enter the gap statement.
  - **Seriality repaired, not warned about.** `serial=True` is the default on
    every operator, so `AX` as `¬EX¬` is sound and `AF`/`AG`/`AU` follow.
  - **The greatest-fixpoint cliff is documented on every gfp operator.**

  What exact mode does *not* fix is stated explicitly: it removes the
  temperature gap, not the gradedness. On a 0.99-weighted cycle it still
  decays to 0.97, and only rounding the relation first gives no decay at all.

### Notes

- Soft mode remains differentiable through the fixpoint by unrolled
  backpropagation. Implicit differentiation via the fixpoint equation, which
  would give constant memory in the iteration count, is not implemented.
- `until`'s deprecation notice still names 0.4.0 as the removal version. The
  argument has not been removed; the notice predates these releases and needs
  a decision rather than a silent change.


## [0.4.0] — 2026-09-18

The group-knowledge layer and dynamic epistemic logic. Additive: the 180-entry
API fingerprint is unchanged apart from the `__all__` lists and `__version__`.

### Added

- **`torchmodal.epistemic`** — the group operators, completing what
  `systems.EpistemicOperator` started with `K_a`: `everybody_knows`,
  `mutual_knowledge`, `distributed_knowledge`, `common_knowledge`,
  `pooled_accessibility` and `and_bounds`.

  The group fold defaults to **Gödel**, not Łukasiewicz. Measured with every
  agent at `K_a = [0.1, 0.2]`, the Łukasiewicz fold gives `[0.000, 0.200]` at
  **every** group size from two upward — dead on arrival for any realistic
  bound — while Gödel holds `0.100`. Same reasoning that made `until_graph`
  use Gödel: idempotence is what survives iteration.

- **Frame audit** — `frame_audit`, `shuffled_null`, `AxiomReport`. Reports
  satisfaction *with* coverage and a shape-matched null, because a bare axiom
  score is not evidence of structure: Łukasiewicz implication makes any triple
  with `A_uv + A_vw <= 1` vacuously satisfied, so 1.00 at low coverage means
  nothing. The corrected protocol is the default; the bare score is not
  reachable by accident.

- **Dynamic epistemic logic** — `functional.announce`, `necessity_after` and
  `group_announce`: graded public and group announcement as a relativisation
  of the relation, with the `A_hi <= A_crisp <= A_lo` sandwich.

### Changed

- **`common_knowledge` now defaults to `tau_decay=0.5`.** It previously
  defaulted to `None` and shipped unusable: the lower bound was **exactly
  0.0000 with exactly zero gradient for every input tried**, so the operator
  could not be trained against or reported.

  The cause is the **greatest-fixpoint cliff**, now pinned in
  `tests/test_traps.py`. Iterating a gfp down from the top through a smooth
  diamond loses a little each sweep, and below roughly 0.999 edge weight there
  is no non-zero fixed point to land on. Measured on a 6-cycle at `tau=0.1`
  with phi true everywhere, `EG` falls **0.955 -> 0.754 -> 0.000** as the
  weight goes **1.0 -> 0.999 -> 0.99** — a 1% softening takes the value from
  0.95 to nothing, while the crisp answer is 1 throughout.

  The collapse is **not** a t-norm artefact: Gödel, product and Łukasiewicz
  all do it, because the lossy step is the modal one, not the conjunction.
  Swapping the fold cannot help; an annealed temperature can, because it makes
  the per-sweep loss summable. Hence the new default, matching `until_graph`.

  Verified sound against a crisp reference checker on 8 frames (complete,
  two-clique, ring, with phi falsified at a world): the lower bound never
  exceeds the crisp value. `tau_decay=None` still reproduces the old behaviour
  and a regression test holds it in place.

- Even at edge weight 1.0 the gfp iteration is **still creeping at a 200-sweep
  cap**, returning 0.955 rather than the true 1.0 — a strict tolerance does not
  terminate. Any future gfp operator must report its iteration count rather
  than imply convergence.


## [0.3.0] — 2026-09-18

> **Additive.** No existing public call returns a different value: a 180-entry
> bit-exact fingerprint of the public API is unchanged apart from the two
> `__all__` lists gaining the new names. The one exception is a bug fix, noted
> under *Fixed* — `box_width_entropy` previously returned `NaN` for small
> `tau`, and now returns the correct value.

### Added

- **`functional.auto_tau`** — the inverse of the bracket. Every other entry
  point asks for a temperature and tells you afterwards how wide the resulting
  interval is; this asks for the width you can tolerate and supplies the
  temperature. The returned `tau` is always *safe*: the realised width never
  exceeds the target.

  Two modes. The **closed form** (`prop_bounds=None`) uses the frame-only
  bound `tau = eps / log n`, valid for any proposition and therefore the one to
  use during training, where the bounds change every step. The **exact** mode
  bisects on the true `box_width_entropy` for given bounds and returns the
  largest `tau` meeting the target — measured 2.4x to 3.7x larger than the
  closed form on a random 12-world frame, which means correspondingly
  better-conditioned gradients.

  This closes a gap the library had been carrying: `box_width_entropy` shipped
  in 0.2.2 and was called **zero times** anywhere in the library. The width was
  computable and entirely unconsumed.

- **`precision=` on `necessity` and `possibility`** — state a bracket width
  instead of a temperature. `precision` overrides `tau` when both are given;
  omitting it is silent and unchanged.

- **Batched evaluation.** `necessity`, `possibility` and `box_width_entropy`
  now accept any number of leading batch dimensions — `(B, |W|, 2)` bounds
  against `(B, |W|, |W|)` relations — and results are **bit-identical** to
  looping over the batch. Training over a dataset of Kripke models no longer
  requires a Python loop. Point-valued input is recognised by rank *relative to
  the relation*, which stays unambiguous when `|W| == 2`, where a rule based on
  the trailing extent would guess wrong. `until` and `until_graph` remain
  single-model.

- **`diagnostics.vacuity_report`** — distinguishes a term that is satisfied
  because it is *true* from one satisfied because the relation is *empty*.
  Every box-built quantity is maximal on the empty relation — an agent that
  sees nothing vacuously knows everything — so a specification written only in
  box has a global optimum that satisfies every axiom and constrains nothing,
  and an L1 sparsity penalty pushes *toward* it rather than against it. This is
  the tool the `contradiction` docstring asks for when it warns that
  `L_contra` "must not be the sole guard against a degenerate optimum".

- **`diagnostics.MONOTONICITY` and `monotone_in_accessibility`** — which bound
  endpoints are monotone in `A`. Only the two log-sum-exp aggregators are
  (`necessity.L`, `possibility.U`); the two `conv_pool` endpoints are not.
  Which endpoint you use determines whether a monotonicity argument is
  available, and the table is now checked by an empirical perturbation test
  rather than asserted.

- **Axioms D (seriality) and 5 (Euclidean)** in `AxiomRegularization`, which
  previously covered only T, 4 and B. D matters because it is the precondition
  for `until_graph(quantifier="box")` being sound — the library documented the
  requirement in 0.2.2 without offering any way to satisfy it.

  Seriality defaults to `serial_hollow=True`. The naive reading
  (`max_j A[i,j] = 1`) is satisfied perfectly by the **identity matrix**, which
  has no dead ends and relates nothing to anything else — and since
  `LearnableAccessibility` is reflexive by default, the identity is exactly
  where a fit can settle. Measured: the identity scores 0.0000 under the naive
  reading and 1.0000 under the hollow one. The penalty also uses a hard `max`
  rather than `smooth_max`, which as an upper bound would understate the
  violation unless debiased by `tau * log n`. Axiom 5's antecedent uses Gödel
  `min` rather than Łukasiewicz, which would collapse to 0 — and so be
  vacuously satisfied — on exactly the sparse relations the axiom should catch.

- 58 new tests (262 from 204), including the batching/loop equivalence, the
  `auto_tau` safety guarantee in both modes, and the NaN regression below.

### Fixed

- **`box_width_entropy` returned `NaN` for small `tau` in float32.** The guard
  `weights.clamp_min(1e-300)` is itself flushed to zero in float32, whose
  smallest normal is about 1e-38, so an underflowed weight produced
  `0 * log(0)`. Entropy is now computed from `log_softmax` with the zero-weight
  terms masked, which is exact and dtype-agnostic. At `tau = 0.008` on a random
  12-world frame the function returned `nan` and now returns `0.0036`; the
  identity against `conv_pool - smooth_min` still holds to 4.4e-16 in float64.

  This also made `auto_tau`'s bisection unusable at tight targets, so the two
  land together.

## [0.2.2] — 2026-09-17

> **No existing public call returns a different value.** Every change below is
> additive — new functions, new modules, or new parameters whose defaults
> reproduce today's output exactly. Verified two ways: a 180-entry bit-exact
> fingerprint of the public API (aggregations, connectives, modal operators,
> `until`, contradiction, utils, every `nn` module, `KripkeModel`, losses,
> systems and `upward_downward`) is unchanged apart from the two `__all__` lists
> gaining the new names; and the ten seeded example scripts produce byte-identical
> output against the 0.2.1 tree.

### Added

- **`functional.until_graph`** — a relation-aware Until, the least fixpoint of
  `U = ψ ∨ (φ ∧ ♢U)` over an arbitrary (cyclic, branching, disconnected or
  learned) relation. Uses **Gödel** connectives, which are idempotent, so the
  lower bound does not decay per step the way `until`'s Łukasiewicz sweep does,
  and an **annealed** temperature `τ_j = τ·ρ^j`, so the accumulated slack is a
  geometric series bounded by `τ·log|W| / (1 − ρ)` rather than growing with the
  sweep count. `quantifier="diamond"` is EU (default, sound on any frame);
  `quantifier="box"` is AU and is sound **only on a serial relation** — this is
  documented with the measurement that shows it failing otherwise.

  On a 6-step chain with `L_φ = 0.9` and ψ true only at the end:

  | | `L` per step | after cutting the 2→3 edge | `∂L[0]/∂A` |
  |---|---|---|---|
  | `until` (existing, unchanged) | 0.500 … 1.000 | **0.500 — unchanged** | 0.0 |
  | `until_graph` (diamond) | 0.900 … 1.000 | **0.000 — broken path detected** | ≈0.50 |

- **`torchmodal.diagnostics`** — a new module, exported at package level as
  `gradient_health`, `assert_has_signal` and `GradientHealthError`. It detects
  the characteristic failure of a differentiable logic: a term pinned to 0 or 1
  whose gradient has vanished, which raises nothing and simply stops
  contributing. It splits `(..., 2)` bound tensors into their `L` and `U`
  endpoints — the dead state of a box neuron is `L = 0` *with* `U = 1`, which
  neither column reveals on its own — and attributes gradients per endpoint.
  A term is reported *dead* when pinned with no gradient, and merely *saturated*
  when pinned but still differentiable. Neither alone makes a report unhealthy:
  a sound upper bound that has legitimately reached 1 looks identical, at the
  term level, to a broken one, and whether the interval clamp passes gradient
  exactly *at* the boundary is a torch-version convention (2.8 passes 1.0, 2.14
  passes 0.0). `healthy` keys instead on signals that are unambiguous and stable
  across versions — a **vacuous** bound spanning the whole interval, no
  parameter receiving a usable gradient, or a missing autograd path.
  `assert_has_signal` is the raising variant for tests.

- **`functional.box_width_entropy`** — the per-world interval width that one
  necessity level contributes, `τ·H(softmin weights)`. This is an identity, not
  an estimate: it equals `conv_pool(x, −x) − smooth_min(x)` (verified to 8.9e-16
  in float64) and, when `L == U` and the output clamp does not engage, equals
  `U_□ − L_□` exactly (3.9e-16). It is bounded by `τ·log n` with equality iff
  every aggregated term ties, and it turns three previously hand-waved
  quantities into computed ones: how loose a given `□` is, the faithful nesting
  depth `k* = ⌈1/(τ·H̄)⌉`, and the width of the `contradiction` dead zone.

- **`examples/muddy_children.py`** and three runnable Colab notebooks under
  `examples/notebooks/` (the muddy-children puzzle, a temporal epistemic
  read-out, and recovery of a hidden constraint graph from valid colourings at
  edge-recovery AUC 1.000).

- **`tests/test_traps.py`** — 27 regression tests that pin the library's *known
  limitations*, so a trap cannot silently return: `until`'s invariance to its
  accessibility and its temperature, `conv_pool`'s non-monotonicity and its exact
  derivative, the `contradiction` dead zone and its identity with the box width,
  and the nesting depth at which necessity floors. 73 new tests in total (202
  from 129).

- Documentation site (`mkdocs.yml`, `docs/`, mkdocs-material + mkdocstrings,
  builds `--strict`), `CONTRIBUTING.md`, `.gitignore`, and CI workflows
  (`.github/workflows/ci.yml`, `docs.yml`) — the repository previously had **no
  CI workflow at all**, although the README badge pointed at one.

### Changed

- **`functional.until` now raises a `DeprecationWarning` if `tau` is passed.**
  The argument has never had any effect — the backward DP contains no smooth
  aggregation — and it is scheduled for removal in 0.4.0. Omitting it is silent
  and unchanged. The signature default is a `float` subclass carrying the
  historical value `0.1`, so `inspect.signature` and any code reading the value
  are unaffected; only identity distinguishes "not passed" from an explicit
  `tau=0.1`. `inference.upward_downward` no longer forwards `tau` to it.

- **Docstrings now state every operator's trap with its measurement**, per the
  library's convention that a bound-producing function says what it bounds, in
  which direction, and by how much:

  - `conv_pool` — documents that it is **not monotone** (`∂f/∂x_k =
    w_k·(1 − (x_k − f)/τ)`, negative once `x_k − f > τ`), and that this
    invalidates the argument "the box neuron is monotone in `A`, therefore the
    bound is sound". The correct route is monotonicity of the hard `min` plus the
    one-sided enclosure. Also states the exact width identity.
  - `contradiction` — documents the **dead zone**: identically zero with zero
    gradient until the bound crossing exceeds the box width. The correspondence
    is exact — measured edges 0.109861 / 0.179176 / 0.230259 for fan-in 3 / 6 / 10
    at `τ = 0.1`, against `τ·log n` of 0.109861 / 0.179176 / 0.230259 — and it is
    now stated that `L_contra` must not be the sole guard against a degenerate
    optimum.
  - `necessity` / `possibility` — document the **accumulated slack** of `τ·H(w)`
    per level with a measured depth table. The per-level loss is the frame's
    branching factor (`τ·log 8`, `τ·log 3`, `τ·log 2` for a complete, bidirectional
    -ring and ring frame over 8 worlds), and a complete frame floors at depth 5,
    exactly where `k*` predicts.
  - `MultiAgentKripke.K_G` / `K_F` — document that they are **two modal levels**
    and therefore carry twice the slack (measured: `G` alone gives `L = 0.861`,
    `K_G` gives `L = 0.770`, the two levels contributing 0.1387 each).
  - `until` — documents that it is **inert with respect to its relation**, reading
    only `accessibility.shape[0]`, with no autograd path back to it, and that its
    Łukasiewicz sweep floors the lower bound.

- **Paper metadata corrected across the repository.** The title is *Modal Logic
  Neural Networks*; the earlier mis-spelling of it (with an adjectival "Logical")
  no longer appears anywhere in the repository. The
  authors are Antonin Sulc (Lawrence Berkeley National Laboratory) and Noor
  Naddour (The University of Queensland); the venue is an **oral presentation at
  NeSy 2026**, the 20th Conference on Neurosymbolic Learning and Reasoning, PMLR
  vol. 284 — `torchmodal/__init__.py` previously said "NeuS"; the year is 2026;
  and the canonical link is <https://openreview.net/pdf?id=uLOdtBm0Cx>, with
  arXiv:2512.03491 retained as a secondary identifier. `CITATION.cff` gained the
  second author and an `@inproceedings` `preferred-citation`, and validates
  against the CFF 1.2.0 schema.

- **Packaging metadata** — `pyproject.toml` gained keywords, full trove
  classifiers, a `Paper` and `Documentation` URL, and a `docs` extra.

- **`misc/MLNN.pdf` replaced with the arXiv v3 camera-ready** (revised
  2026-09-05). The bundled copy was the v2 build, titled *Modal Logical Neural
  Networks* with a single author, and so contradicted the corrected metadata in
  the rest of the repository. The v3 title page confirms the title, both
  authors, and PMLR vol. 284, pp. 1–34.

- **`.gitignore` now covers `.env` and other secret files.** A `.env` present in
  the working tree was not ignored, so a `git add -A` would have committed it to
  a public repository. It had never been committed — verified against the full
  history.

- **README restructured** to lead with what the library is in one sentence, the
  NeSy 2026 oral, a runnable ten-line epistemic puzzle, a comparison against LNN,
  LTN, DeepProbLog, Semantic Loss, Scallop, SATNet and STLCG, and a
  **Limitations** section. The API reference moved below the fold.


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
