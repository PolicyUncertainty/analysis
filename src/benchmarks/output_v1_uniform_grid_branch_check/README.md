# dcegm engine benchmark -- round 1: branch equivalence + uniform-grid method check

Produced by `run_benchmark.py` (4 cases, one `--case` run each) + `make_table.py`,
2026-08-31, on the HU HPC (H100 NVL).

## Setup

Four cases: dcegm branch (`main` vs `remove_endog`, the on-demand-law-of-motion
refactor) x upper-envelope method (`fues` vs `druedahl_jorgensen`), all at the full
production state space (`sex_type`/`edu_type` = `"all"`/`"all"`) but with
**synthetic, uniform grids**, not the production ones:

- `assets_end_of_period`: 30 points, `linspace(0, 10_000_000)` (raw currency units)
- `assets_begin_of_period` (dj only): 30 points, `linspace(0, 1_000)[1:]` (model
  units) -- i.e. starting at ~32,300 raw currency, with **no resolution below that**
- `experience`: 11 points, `linspace(0, 1)`

## Findings

1. **`main_fues` == `new_fues` and `main_dj` == `new_dj`, exactly**, in every
   per-period consumption and choice-share table. This directly confirms the
   on-demand law-of-motion refactor on `remove_endog` is behavior-preserving --
   the same thing the golden-value test we deleted from
   `submodules/dcegm/tests/test_two_occupation_model.py` was there to check, now
   validated empirically on our real model instead of a toy one.
2. **fues vs dj diverge substantially**, and systematically rather than as noise:
   full-time choice share ~51% under fues vs 62-66% under dj in early periods;
   period-0 mean consumption 3.23 (fues) vs 10.91 (dj); dj's per-period consumption
   path is visibly noisy/oscillating where fues's is smooth.
3. **dj is faster and uses less memory** (`new_dj`: 12.5s solve, 26.93GB peak vs
   `new_fues`: 24.6s, 46.16GB) -- but this is suspected to be partly an artifact of
   the grid setup, not a genuine dj advantage: dj evaluates directly on
   `assets_begin_of_period` (no endogenous refinement), and that grid has zero
   points below ~323,000 raw currency -- exactly the credit-constrained region
   where policy is most curved and production's real (non-uniform) grid
   concentrates most of its points. Coarser resolution there would both bias the
   solution and make dj cheaper to evaluate.

## Follow-up (round 2, see `../output_v2_production_grid_method_check/`)

Rerun `fues` and `druedahl_jorgensen` using production's actual (non-uniform,
manually-optimized) `assets_end_of_period` and `experience` grids for both methods,
plus a matching non-uniform `assets_begin_of_period` for dj (same points as
`assets_end_of_period`, in model units), to check whether the fues/dj gap above
shrinks once dj gets a fairly-resolved grid. Branch comparison is dropped from round
2 -- point 1 above already settled that question.
