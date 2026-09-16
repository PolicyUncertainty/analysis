# Experience grid

Guide to the experience state and its interpolation grid.

## The experience state

Experience is the second continuous state (interpolated alongside assets). Its
meaning depends on whether the individual is retired:

- **Working** (`lagged_choice != 0`): stored directly as **real credited-years**.
- **Retired** (`lagged_choice == 0`): stored as **pension points**, rescaled onto
  the same per-`(sex, period)` real-year axis. `scale_experience_years` maps pension
  points → stored value, `construct_experience_years` inverts it; both take `sex`
  and use the grid's per-`(sex, period)` cap as the rescaling constant. Working is
  the identity branch.

## The grid

One grid per `(sex, period)`, 14 nodes, over `[0, cap]`, supplied to dcegm by
`experience_grid_from_state(period, sex, model_specs)` (a lookup into
`model_specs["experience_grid_working_by_sex_period"]`, shape `(n_sexes, n_periods,
14)`). Built by `src/model_code/state_space/experience_grids.py`.

Only the grid *shape* is hardcoded (node count 14, bracket size 4). All
economically meaningful quantities are derived:

- **cap** `= max_exp_diff_period_working[sex] + min(period, max_ret_age - start_age)`
  — that sex's max initial experience plus years worked, frozen past the last
  working age. Nobody is placed beyond attainable experience. Max initial experience
  is per sex (men 16, women 15), so caps top out at 58 / 57.
- **VLI threshold** `= experience_threshold_very_long_insured[sex]` `= 45 / credited-
  periods-per-experience factor` (men 42, women 31.5). See the discontinuity note
  below.
- **bracket** `= threshold - quantum·[3, 2, 1, 0]`, quantum = each sex's reachable
  experience step (men whole years, full-time only; women half-years,
  `exp_increase_part_time`). Evaluates to men `39,40,41,42`; women
  `30,30.5,31,31.5`.

Two regimes per row (`experience_grid_row`):

- **(a) `cap <= threshold`**: uniform `linspace(0, cap, 14)`.
- **(b) `cap > threshold`**: `[0, bracket_start]` equal bins | dense bracket |
  `[threshold, cap]` equal bins. The non-bracket nodes split between the two regions
  in proportion to their lengths, so both share one spacing and the number of bins
  above the threshold grows with the cap (men 1→3, women 3→5 across ages).

`validate_working_grid_table` enforces the invariant: each row starts at 0, is
strictly increasing, and its top `== cap`.

## The VLI discontinuity (why the bracket exists)

German pensions grant deduction-free early retirement to the *besonders langjährig
Versicherte* — **≥ 45 credited periods**. Credited periods are modeled as
`factor[sex] × experience`, so eligibility is `experience >= threshold[sex]`. At the
threshold the value of retiring jumps (full vs. penalized pension points, see
`check_very_long_insured` / `early_retirement_paths.py`), so linear interpolation
needs nodes bracketing it — hence the dense bracket, spaced at each sex's reachable
quantum so nodes land where the jump can actually occur.

## Key constraint: grid keys on `(sex, period)` only

dcegm's shared-child consistency check requires every parent state-choice
transitioning to the same child to use the same experience grid. A not-yet-retired
individual who chooses retirement and an already-retired individual transition to
the *same* child, so the grid **cannot** depend on `lagged_choice`; retired states
therefore reuse the working axis (rescaled) rather than a separate pension-point
grid. `sex` and `period` are safe (`sex` is time-invariant; cross-period sharing is
handled by dcegm's proxy). Any future dependence must likewise agree across shared
children.

## Where it lives

- `src/model_code/state_space/experience_grids.py` — grid construction rule + cap
  validation.
- `src/model_code/state_space/experience.py` — `experience_grid_from_state`,
  `scale_experience_years` / `construct_experience_years`,
  `get_next_period_experience`, and the table/cap builders.
- `src/specs/experience_pp_specs.py` — per-sex `max_exp_diff_period_working`, VLI
  thresholds, and the precomputed grid + `experience_grid_cap_by_period` tables.

The value-function discontinuity evidence that motivates the bracket is summarized in
`docs/benchmarks.md`.

## Invariants when editing

- All state-choices must use the **same node count** (rectangular storage).
- Keep the grid a function of **`(sex, period)`** only (see the constraint above).
- The working grid's **top node must equal the cap** — the retired rescaling
  (`experience_grid_cap_by_period`) relies on it, so retired pension points stay
  on-grid.
