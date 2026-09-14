# Implementation plan: type-specific, real-value, age-dependent experience grid

**Status:** planned. The repository currently uses the original single pooled
normalized `[0, 1]` experience grid. This document records the staged path to a
type-specific, real-value, age-dependent grid, and why the whole thing is
sequenced after a dcegm fix.

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
   The design below was prototyped and verified in-session, then reverted with the
   grid back to the pooled baseline; re-implement it:
   - **Type-specific** grid via dcegm `continuous_grid_functions`
     (`experience_grid_from_state`), sex-specific brackets (10 nodes/sex). Note:
     this alone (sex only, period-independent) already builds today -- it is the
     period dependence in the next bullet that needs the dcegm fix.
   - **Real values, age-dependent:** `experience_grid_from_state(sex, period, ...)`
     returns the per-sex reference grid scaled to `M_p =
     max_exps_period_working[period]` (real years);
     `construct_experience_years`/`scale_experience_years` become identity for
     working (state is real years) and rescale by the grid's max value for retired
     ("always rescale by the max value in the grid").
   - Verified in-session: the physical grid points are identical to the pooled/
     `[0,1]` representation (max diff 0.0), so the representation change alone is
     bit-identical; and a men-low solve of the sex-specific `[0,1]` grid matched
     the pooled grid's dynamics. The only numerical change to expect is from the
     dcegm proxy fix in step 1, not from the grid re-representation.
   - Test: solve men-low and compare to the step-1 (post-dcegm-fix) solve → expect
     machine-precision match (the grid change alone changes nothing numerically).

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
