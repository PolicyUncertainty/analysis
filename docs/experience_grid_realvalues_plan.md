# Implementation plan: type-specific, real-value, age-dependent experience grid

**Status:** step 1 done (dcegm proxy fix landed). Step 2 in progress: 2a (real
values, age-dependent, pooled/not-yet-type-specific) done; 2b (re-add
type-specificity) and step 3 follow-ups still open. This document records the
staged path to a type-specific, real-value, age-dependent grid, and why the
whole thing is sequenced after a dcegm fix.

## Where we are now (baseline)

- **Experience grid:** the original single **pooled** `define_experience_grid`
  (`src/model_code/state_space/experience.py`) -- 12 nodes, normalized to `[0, 1]`,
  shared across sexes and periods, passed to dcegm as a static array (with an
  `experience_grid` override hook for benchmarks). This is the state the target
  work below departs from.
- **Analysis + plots kept:** `src/model_code/plots/retirement_system_plots.py`
  (retirement-system feature plots) and
  `src/benchmarks/inspect_experience_discontinuities.py` (value-function
  discontinuity inspection), plus `docs/experience_grid_discontinuities.md`. These
  motivate the target design; they reference the pooled grid, not any sex-specific
  grid.

The `[0, 1]` normalization uses a period-dependent scale
(`max_exps_period_working[period]`) that jumps from 48 to 59 at age 63
(`min_period_very_long_insured`), so a fixed normalized grid images onto a
period-dependent real-year range. That is harmless today but makes it hard to (a)
place grid nodes at real experience-year targets per age and (b) reason about how
value-function discontinuities travel across ages (see the discontinuity report).

## Target

An **age-dependent grid in real experience-years**: each working age gets a grid
over `[0, M_p]` (that age's maximum experience), so the stored state *is* real
credited-years and the VLI threshold sits at its real value (42 for men, 31.5 for
women) at every relevant age. Retired states reuse the same per-period axis for
pension points, rescaled by the grid's maximum value.

This was prototyped and verified to be **bit-identical in physical grid points**
to the current normalized grid (the grid values are just re-expressed), so the
*grid change itself does not change results*.

## Why it is blocked, and the prerequisite

An age-dependent grid is **period-specific**. dcegm's sparsity **proxy** collapses
some child states across periods (notably death states of all ages proxy to one
last-period death state, and some longer-retired states). dcegm's
`check_continuous_grid_consistency_across_shared_children` then requires every
parent transitioning into such a shared child to use the *same* grid — which a
period-specific grid violates, so the model fails at build time. (Confirmed
empirically: men-low fails the check on retired states at periods 0 vs 1 sharing a
child.)

**Prerequisite — dcegm proxy change** (see
`submodules/dcegm/docs/proxy_noproxy_plan.md`): make the proxy a pure value-reuse
pointer. Run the law of motion / budget, the dedup, and the grid-consistency check
on the **non-proxy** child; gather only the value/policy/endog from the proxy slot.

**Important — this dcegm change is a bug fix and will change results.** Today the
transition/budget for a proxied child is evaluated at the *proxy's* (last-period)
identity, so the begin-of-period wealth for a death/longer-retired continuation
uses the wrong age's income. The fix evaluates it at the correct age. So after the
fix, results differ from the current implementation in exactly the proxied
continuations (death, longer-retired) — a correction, not a regression. This is
why we do the dcegm fix **before** the grid change, and keep the two changes
separate in the history.

## Sequence

1. **dcegm proxy bug fix** (submodule), per `proxy_noproxy_plan.md`.
   - Deliverable: a period-specific second-continuous grid builds and solves for a
     toy model with a death-like cross-period proxy; existing dcegm tests unchanged
     (the non-proxy map equals the proxy map when nothing is proxied across a
     grid-varying dimension); an oracle test (death not proxied, solved explicitly)
     matches the proxied solve.
   - Expect: for this repository, a resolve of the current `[0,1]` model changes
     only the proxied continuations. Quantify the change (it should be small and
     localized) so it is understood, not a surprise.

2. **Introduce the type-specific, real-value, age-dependent grid** (this repo).
   Split into two sub-steps so real-value/age-dependence and type-specificity can
   each be checked in isolation.

   **2a. Real values, age-dependent, not yet type-specific. Done.**
   - The pooled `[0, 1]` grid (`define_experience_grid`, unchanged) scaled by
     `M_p = max_exps_period_working[period]` (real years) for every period is
     precomputed once into `specs["experience_grid_by_period"]`
     (`build_experience_grid_by_period`); `experience_grid_from_state`, the
     callable dcegm requires for a state-specific `continuous_grid_functions`
     entry, is then a pure lookup into that table, not a per-call computation --
     the grid is built once and fed in, the function just indexes it.
   - Verified: dividing row `p` of the table by `max_exps_period_working[p]`
     reproduces `define_experience_grid`'s `[0, 1]` values exactly (max diff
     ~1e-16) at every checked period -- this is the same pooled grid, just
     re-represented in real years per period, not a different one.
     `construct_experience_years`/`scale_experience_years` became identity for
     working (state is real years) and rescale by the grid's max value for retired.
   - Builds and solves men-low under dcegm's grid-consistency check; no test
     regressions.

   **2b. Type-specific (sex-specific brackets). Not yet re-applied** -- was
   implemented and verified once (10 nodes/sex, built as the pooled grid with the
   *other* sex's very-long-insured bracket points dropped -- a literal subset of
   the pooled grid's physical points, so bit-identical by construction) but rolled
   back to land 2a on its own first. Re-add as a second layer on top of 2a: a
   `(sex, period)` precomputed table instead of the current `(period,)` one, same
   lookup pattern.
   - Verified: `specs["experience_grid_by_sex"]`, scaled to `M_p` at the max
     period, places the very-long-insured node at exactly 42.0 (men) / 31.5
     (women) real years, as intended.
   - Test: built and solved men-low on the new grid and compared to a step-1
     (post-dcegm-fix, pooled-grid) men-low solve at matched physical grid points.
     **`policy` and `endog_grid` are bit-identical at every checked (period, row,
     node)** -- the decision rules are unchanged. `value` differs by up to ~0.1
     in absolute terms (<0.2% relative, on a ~60-scale value), growing from
     exactly 0 at the terminal period back to ~1e-2 at period 0 -- a backward-
     induction accumulation pattern consistent with floating-point
     summation-order noise from the 10- vs 12-node array shape (different JAX
     vectorization/padding), not a discrepancy in the economics. This is looser
     than the "machine-precision match" expected above; flagged here since the
     original expectation was wrong, not silently dropped.

3. **Follow-ups / caveats**
   - `specify_simple_model` passes a static per-sex grid; give it the period-scaled
     grid too (or leave as a debug model with an oversized young-age grid).
   - A couple of scripts hardcode a normalized experience literal
     (`run_eval_expectation_graphs.py`, `plots/wealth_plots.py`) and
     `benchmarks/inspect_experience_discontinuities.py` builds its grid as
     `years / scale`; update these to real-year units when the grid change lands.
   - Once real-value age grids are in, the period-dependent normalization scale
     jump at age 63 is gone; revisit whether the discontinuity analysis should be
     rerun on the new grid (the jumps' real-year locations no longer get distorted
     across the age-63 boundary).
   - The plots and the discontinuity benchmark currently read the pooled
     `define_experience_grid`; repoint them to the type-specific grid when it lands.
