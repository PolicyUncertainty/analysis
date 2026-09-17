# Summer revision changes

Economic (results-affecting) changes on the `summer` branch relative to `main`.
Computational/infrastructure changes are out of scope for this log.

## a) Experience-grid improvement

Replaced the pooled experience grid (normalized to `[0, 1]`, shared across types)
with a **sex-specific, real-value, age-dependent** grid, built by a two-regime rule
(`src/model_code/state_space/experience_grids.py`; see `docs/experience_grid.md`):

- Each `(sex, period)` grid spans `[0, cap]`, `cap = max_init_exp[sex] +
  min(period, last_working_period)` (real credited-years; men 16, women 15 initial).
- Below the very-long-insured (VLI) threshold: uniform. Above it: a dense bracket at
  the threshold (derived from the credited-periods factor and each sex's reachable
  quantum) plus equal-spaced bins before and after.
- Retired states reuse the same axis (pension points rescaled onto it), required by
  dcegm's shared-child consistency check.

**Effect:** an accuracy improvement at the VLI discontinuity. Verified against the
pre-revision grid for low-education men on the `run_cf_debias` scenario: **no
qualitative change** — retirement-timing, pension, and welfare (CV) effects are
preserved; only the VLI share is resolved more sharply (baseline ~16% → ~18%).

## b) Law-of-motion bugfix (dcegm)

dcegm submodule bump including *"Dev of right law of motion."* The sparsity **proxy**
is now a pure value-reuse pointer: the law of motion / budget is evaluated on the
correct **non-proxy** child at its true age, and only the value/policy is gathered
from the proxy slot.

**Effect:** previously, a proxied child's begin-of-period wealth used the *proxy's*
(last-period) age income — wrong for death and longer-retired continuations. The fix
evaluates them at the correct age, so results change in exactly those proxied
continuations (a correction, not a regression).

## Grid-change check: low-education men debias effects

`run_cf_debias` (informed − misinformed) for low-education men, old grid vs new
grid (both include the law-of-motion fix, so this isolates change a). Effects and
signs are preserved; only the very-long-insured share is resolved more sharply.

| debias effect (informed − misinformed) | old grid | new grid |
|---|---|---|
| retirement age (years) | −0.809 | −0.804 |
| retirement age, excl. disabled (years) | −1.013 | −1.005 |
| pensions | −0.958 | −0.973 |
| share below SRA (pp) | +11.0 | +9.4 |
| share very-long-insured (pp) | −11.1 | −13.5 |
| share disability pensions (pp) | −2.08 | −2.13 |
| compensated variation (level) | 0.1033 | 0.0938 |
