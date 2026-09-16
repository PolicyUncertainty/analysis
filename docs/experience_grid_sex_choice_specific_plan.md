# Plan: sex- and lagged_choice-specific experience grid (14 nodes)

**Status:** proposal only, not implemented or validated. Builds on
`experience_grid_discontinuities.md` and the dense-grid inspection in
`src/benchmarks/inspect_experience_discontinuities.py` /
`experience_discontinuities_report.md`.

## Motivation

Two populations currently share one grid but need different things:

- **Working, men vs. women:** the VLI reachability jump (see the discontinuity
  report) needs a bracket around each sex's threshold. Men can only work full
  time (`exp_increase_part_time` not applicable) so experience moves in
  **whole years**; women can also work part-time (`exp_increase_part_time =
  0.5`), so their experience moves in **half-year** steps. The bracket
  spacing should match each sex's actual reachable quantum, not a shared
  half-year default.
- **Retired (`lagged_choice == 0`):** the discontinuity inspection found the
  continuation value **smooth everywhere** in pension points — no threshold
  bracket needed. The relevant range is real pension points, `[0,
  max_pp_retirement]`, not the working-years scale.

## Proposed construction (4 grids × 14 nodes)

All working-grid targets are **real years, clipped per period to that
period's own cap** (`max_exps_period_working[period]`) — the construction
validated in this session (`build_grid_row`): every attainable target keeps
its exact location; unattainable targets collapse harmlessly near the cap.
Nobody is assigned experience beyond what's attainable at their age.

- **Men, working:** base spread `0, 6, 12, 18, 24, 30, 34, 37` (8 nodes) +
  whole-year bracket `39, 40, 41, 42` (4 nodes) + top anchors `50, 59` (2
  nodes).
- **Women, working:** base spread `0, 6, 12, 18, 24, 27, 35, 42` (8 nodes) +
  half-year bracket `30.0, 30.5, 31.0, 31.5` (4 nodes) + top anchors `50, 59`
  (2 nodes).
- **Retired (men and women, pending check below):** **equally spaced in real
  pension points**, `np.linspace(0, max_pp_retirement, 14)` — no bracket, no
  curvature-based bias (we have no evidence for where, if anywhere, extra
  density would pay off; equal spacing is the honest default until a
  grid-error sweep says otherwise). Converted to the stored per-period value
  via the model's own `scale_experience_years(is_retired=True)`, which is the
  correct rescaling here (`max_pp_retirement` is period-independent, unlike
  `max_exps_period_working`, so this rescaling is not the compression bug we
  fixed for working years).

## Open items before implementation

1. **Wiring gap:** `continuous_grid_functions["experience"]` currently keys
   only on `period`. Needs extending to also branch on `sex` and
   `lagged_choice` (closer to the rolled-back `experience_grid_by_sex`
   mechanism, rebuilt on the corrected real-year/clip construction).
2. **Unverified for women:** the reachability-jump pattern and the "smooth
   retired continuation value" finding were only checked for men. Should be
   re-run for women before trusting the mirrored bracket/retired proposals.
3. **Possible follow-up, separate from this grid change:** `max_exp_diff_period_working`
   (currently pooled) could be computed per sex, which would make
   `max_exps_period_working` sex-specific too (`(n_sexes, n_periods)`). That
   touches `experience_grid_from_state`, `construct_experience_years`,
   `scale_experience_years`, `get_next_period_experience`, and
   `create_max_pension_point` — scope as its own step, not bundled here.
4. **Validation:** before adopting, solve on each proposed grid and compare
   `policy`/`endog_grid` against a reference grid at matched physical points
   (the same check used for the type-specific step in
   `experience_grid_realvalues_plan.md`), not just visual inspection.
