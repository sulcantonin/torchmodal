# Limitations

These are measured properties of the implementation, not speculation. Each has a regression
test in [`tests/test_traps.py`](https://github.com/sulcantonin/torchmodal/blob/main/tests/test_traps.py) so it cannot silently change.

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

