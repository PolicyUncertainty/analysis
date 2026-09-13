# Discontinuities in the experience grid

This note documents where the value/policy function has kinks or jumps along the
**experience** dimension, why grid nodes are placed where they are in
`define_experience_grid` (`src/model_code/state_space/experience.py`), and what
this implies now that dcegm supports **type-specific** continuous-state grids.

## The experience state

Experience is the second continuous state (solved by interpolation, alongside
assets). It is stored **normalized to `[0, 1]`**: `1` corresponds to the maximum
experience attainable in a working state. Internally experience is counted in
**half-years** (part-time adds `exp_increase_part_time = 0.5`, full-time adds
`1`).

The normalization constant is the last (and largest) entry of
`max_exps_period_working`. With the current first-step estimates:

| quantity | value |
|---|---|
| `max_exp_diff_period_working` (max initial experience) | `16` |
| `max_ret_age - start_age` | `72 - 30 = 42` |
| max working experience | `42 + 16 = 58` |
| normalization constant `max_exps_period_working[-1]` | `59` |

So one half-year of experience is `0.5 / 59 ≈ 0.00847` in normalized units.

When an individual freshly retires, the experience slot is *reinterpreted* as
accumulated **pension points** (rescaled by `max_pp_retirement`); the working and
retired meanings share the same `[0, 1]` grid. The retirement decision itself is
where the experience discontinuities below bite.

## The dominant discontinuity: the "very long insured" threshold (sex-specific)

The German pension system grants early retirement **without deductions** to the
*besonders langjährig Versicherte* — individuals with **≥ 45 years of credited
periods**. In the model this is a hard threshold on experience
(`check_very_long_insured` in
`src/model_code/pension_system/early_retirement_paths.py`):

```
enough_years = experience_years >= experience_threshold_very_long_insured[sex]
```

- **Below** the threshold: early retirement pension points carry the early-
  retirement penalty (`early_retirement_factor * total_pension_points`).
- **At/above** the threshold (and retiring within 2 years of the SRA): the
  individual keeps the **full** `total_pension_points`.

This produces a **jump in the value of retiring** as experience crosses the
threshold, hence a kink in the value function and a shift in the optimal policy
at exactly that experience level. Linear interpolation over experience can only
represent the jump cleanly if there are grid nodes **bracketing** it — so the
grid places one node **exactly at the threshold** and one **half a year below**
(`threshold - 0.5`).

### The threshold is sex-specific

The 45-year credited-periods rule is converted to experience with a
sex-specific multiplier (`add_very_long_insured_specs`), then rounded up to the
next half-year. With the current estimates:

| sex | `45 / factor` | threshold (half-year units) | normalized (`÷ 59`) | bracket node `threshold − 0.5` (normalized) |
|---|---|---|---|---|
| men   | `45 / 1.0768 = 41.79` | `42.0` | `0.7119` | `41.5 → 0.7034` |
| women | `45 / 1.4341 = 31.38` | `31.5` | `0.5339` | `31.0 → 0.5254` |

It is **only sex-specific** — not education-specific — so each sex needs a
different pair of experience nodes. This is the primary reason to move to
**type-specific experience grids**.

## The current (pooled) grid

`define_experience_grid` builds one grid shared by all types:

1. Start from `np.linspace(0, 1, 11)` — 11 uniform nodes.
2. Append **all four** very-long-insured nodes (both sexes × {threshold,
   threshold − 0.5}): `[0.7119, 0.5339, 0.7034, 0.5254]`.
3. Drop the uniform nodes near `0.5`, `0.6`, and `0` (they collide with or crowd
   the appended threshold nodes).
4. Sort, then pin the ends/spacing: `grid[0] = 0`, `grid[1] = 0.15`,
   `grid[-3] = 0.85`.

Resulting grid (12 nodes):

```
[0.0, 0.15, 0.3, 0.4, 0.5254, 0.5339, 0.7, 0.7034, 0.7119, 0.85, 0.9, 1.0]
      ^women bracket^          ^men bracket^
```

The two nodes at `0.5254 / 0.5339` exist **only for women**, and the two at
`0.7034 / 0.7119` exist **only for men**. In a pooled solve every type carries
both pairs even though half of them sit at an experience level where *that*
type's value function is smooth — wasted resolution.

## Implication for type-specific grids

With dcegm handling per-type continuous grids (and `create_model_config` already
receiving `sex_type` / `edu_type`), the experience grid can be built per sex so
that each type keeps **only its own** very-long-insured bracket:

- **Men:** threshold pair `{0.7034, 0.7119}`; drop `{0.5254, 0.5339}`.
- **Women:** threshold pair `{0.5254, 0.5339}`; drop `{0.7034, 0.7119}`.

This frees two nodes per type, which can either shrink the grid (cheaper solve)
or be reinvested elsewhere in the experience range at the same node count. The
rest of the grid construction (uniform base, end pinning) is unaffected.

## Secondary, non-type-specific structure

These are smaller kinks that do **not** vary by type and are handled by the
uniform base grid, not by dedicated nodes:

- **Piecewise-linear pension points in experience.**
  `calc_pension_points_form_experience` linearly interpolates
  `pp_for_exp_by_sex_edu` between integer experience years, so the
  pension-point-per-experience mapping has a mild slope change at each integer
  year. Because the normalization constant (`59`) is common across types, these
  interior kinks fall at the same *normalized* experience for all types.
- **Boundaries `0` and `1`.** The endpoints are pinned exactly; `grid[1] = 0.15`
  and `grid[-3] = 0.85` just keep the spacing near the ends reasonable after the
  threshold nodes are spliced in.

Other retirement-path features (early-retirement penalty, disability fill-up,
late-retirement bonus, mothers' pension) vary with **age / retirement-age
difference or add constants**, not with experience, so they introduce no
additional experience-grid discontinuity.
