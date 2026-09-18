# Support

## Where to ask

| I want to… | Go to |
|---|---|
| Report a bug | [Open a bug report](https://github.com/sulcantonin/torchmodal/issues/new?template=bug_report.yml) |
| Request a feature | [Open a feature request](https://github.com/sulcantonin/torchmodal/issues/new?template=feature_request.yml) |
| Ask how to model something | [Open a question issue](https://github.com/sulcantonin/torchmodal/issues/new?template=question.yml) |
| Read the API reference | [sulcantonin.github.io/torchmodal](https://sulcantonin.github.io/torchmodal/) |
| Understand a trap or caveat | [Limitations](https://sulcantonin.github.io/torchmodal/limitations/) |
| Contribute | [CONTRIBUTING.md](CONTRIBUTING.md) |

Issues are the right venue for everything. There is no mailing list or chat;
keeping the discussion in issues means the next person with the same question
can find the answer.

## Before you open an issue

Two checks resolve a large share of reports, because they cover the failure
mode this library is most prone to — a term that is silently pinned and
contributing nothing, which raises no error at all.

**If a constraint "has no effect":**

```python
from torchmodal.diagnostics import gradient_health

report = gradient_health(lambda: my_term(A), {"A": A})
print(report["issues"])     # dead terms, vacuous bounds, unreachable params
print(report["warnings"])   # saturated but still differentiable
```

**If a bound looks implausibly wide or an axiom is satisfied too easily:**

```python
from torchmodal.functional import box_width_entropy
from torchmodal.diagnostics import vacuity_report

box_width_entropy(A, bounds, tau=0.1)   # exactly how much width one □ adds
vacuity_report(lambda a: my_term(a), A) # true, or vacuously satisfied?
```

Pasting either report into the issue usually *is* the diagnosis.

## What a good report contains

- The version (`python -c "import torchmodal; print(torchmodal.__version__)"`)
  and your torch version — several documented behaviours differ between torch
  2.8 and 2.14.
- A short script that runs, with the number you got and the number you
  expected. A concrete reproduction is worth several paragraphs of prose.
- Whether you were in `mode="soft"` or `mode="exact"`, since they have
  genuinely different guarantees.

## Response expectations

This is a small research library maintained alongside other work. Bug reports
with a reproduction are usually looked at within a week; feature requests may
sit longer. Security-relevant reports should be sent to the maintainer address
in [CITATION.cff](CITATION.cff) rather than opened publicly.

## Known limitations

Several behaviours look like bugs and are not — a `contradiction` loss that is
exactly zero, a greatest fixpoint that collapses, an `until` that ignores its
relation. All are documented with measurements in
[Limitations](https://sulcantonin.github.io/torchmodal/limitations/) and pinned
in `tests/test_traps.py`. Worth a look before reporting one of them.
