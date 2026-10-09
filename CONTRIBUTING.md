# Contributing to torchmodal

Thanks for considering a contribution. This is a small research library, so the
bar is less "match our process" and more "keep the guarantees true".

## The one rule that matters

**Every operator's docstring must state which crisp operator it bounds, in which
direction, and what the gap is.** That convention is the library's main selling
point, and it is only worth anything if it is total. A new operator without it
will be sent back, however correct the maths.

Concretely, a bound-producing function says:

- what it approximates (`min`, `max`, the crisp Kripke value of `□ϕ`, …),
- which way the error goes (`smooth_min(x) <= min(x)`, always),
- how big the error can be (`τ·log n`, `τ·H(w)`, exactly — with the measurement
  that backs it up).

If the operator has a trap — a regime where it is inert, unsound, or silently
dead — say so in a `.. warning::` with the numbers, and add a regression test to
[`tests/test_traps.py`](tests/test_traps.py) so the trap cannot quietly return.

## Setup

```bash
git clone https://github.com/sulcantonin/torchmodal
cd torchmodal
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
pytest tests/                            # all tests must pass
ruff check torchmodal/ tests/ examples/  # must be clean
mypy torchmodal/                         # must be clean
```

`mypy` has been clean since 0.8.0 and CI gates on it; a new `# type: ignore`
needs a comment saying why.

## Conventions

- `from __future__ import annotations` at the top of every module.
- An explicit `__all__`.
- Google-style docstrings with `.. math::` for the formulae.
- Full type hints.
- A test for every new function.
- Line length 88 (`ruff` enforces it).
- Python 3.9 is the floor, so no `match`, no `X | Y` at runtime.

## Changing existing behaviour

Prefer additive changes: new functions, or new parameters whose defaults
reproduce today's output exactly. If a change does move an existing number, say
so explicitly in the pull request and in [`CHANGELOG.md`](CHANGELOG.md) — people
cite results produced with this code.

The repository has committed artefacts under `examples/coloring_results/` and
`examples/sudoku_results/`. If your change could affect them, regenerate and
diff before claiming it does not.

## Good first issues

Issues labelled
[`good first issue`](https://github.com/sulcantonin/torchmodal/labels/good%20first%20issue)
are genuinely small and self-contained. A few standing ones:

- Add a `nn.UntilGraph` module wrapper around `functional.until_graph`, matching
  the style of `nn.Necessity` / `nn.Possibility`.
- Add an `add_until_graph` node type to `inference.FormulaGraph`, leaving the
  existing `add_until` node's behaviour untouched.
- Add a frame-axiom regulariser for seriality (every world has a successor) to
  `losses.AxiomRegularization`, alongside the existing T / 4 / B axioms.
- Seed `torch` in the example scripts that only seed `numpy`, so their printed
  output is reproducible.
- Extend the docs site with a worked example page built from an existing script
  in `examples/`.

## Reporting a bug

A bug report that includes a short script reproducing the number you got and the
number you expected is worth five that describe the problem in prose. If the
symptom is "the constraint had no effect", run it through
`torchmodal.diagnostics.gradient_health` first and paste the report — that is
usually the whole diagnosis.
