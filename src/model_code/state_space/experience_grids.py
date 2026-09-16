"""Age- and sex-specific working experience grids (real credited-years).

Each ``(sex, period)`` grid has ``N_EXPERIENCE_NODES`` nodes over ``[0, cap]``,
where ``cap = max_exp_diff_period_working[sex] + min(period, max_working_period)``
-- that sex's max initial experience plus years worked since, and never beyond the
experience attainable by the last working age. Two regimes, by a fixed rule:

  (a) ``cap <= threshold``: the very-long-insured (VLI) threshold is not yet
      attainable, so the grid is uniform -- ``linspace(0, cap, N)``.

  (b) ``cap > threshold``: a dense bracket sits at/just below the threshold (the
      VLI reachability jump, see ``docs/experience_grid.md``),
      with equally spaced bins *before* the bracket and equally spaced bins
      *after* the threshold up to the cap. The non-bracket nodes are split between
      the before and after regions in proportion to their lengths, so the two share
      one spacing and the number of bins above the threshold grows with the cap --
      it is not a fixed count.

Only the grid *shape* is hardcoded here (node count, bracket size). Everything with
economic content is derived from the model:

  * the **threshold** is ``experience_threshold_very_long_insured[sex]`` (= 45 /
    the estimated credited-periods-per-experience factor);
  * the **bracket** is that threshold minus multiples of each sex's reachable
    experience quantum (men accrue whole years -- full-time only; women can accrue
    half-years -- part-time), so the bracket steps match what each sex can land on;
  * the **cap** (and hence the implicit top of the grid) is
    ``max_exp_diff_period_working[sex] + min(period, max_working_period)``.

The retired experience axis reuses these same grids (see ``experience.py``).
"""

import numpy as np
from jax import numpy as jnp

N_EXPERIENCE_NODES = 14
N_BRACKET_NODES = 4  # dense nodes at/just below the VLI threshold


def vli_bracket(threshold, quantum):
    """Dense run of ``N_BRACKET_NODES`` ending exactly at ``threshold``, spaced by
    that sex's reachable experience ``quantum`` (whole years for men, half for women).
    """
    return threshold - quantum * np.arange(N_BRACKET_NODES - 1, -1, -1)


def _cap(max_init_exp, max_working_period, period):
    """Max experience attainable at this age: initial experience plus years worked,
    frozen once past the last working age (retired states accrue no experience)."""
    return max_init_exp + min(period, max_working_period)


def experience_grid_row(cap, threshold, bracket):
    """One ``(sex, period)`` grid row for the given cap, per the two-regime rule."""
    if cap <= threshold:
        # (a) threshold not attainable: uniform spacing to the cap.
        grid = np.round(np.linspace(0.0, cap, N_EXPERIENCE_NODES) * 2) / 2
    else:
        # (b) threshold attainable: equal bins | dense bracket | equal bins, with the
        # before-bracket region [0, bracket_start] and the above-threshold region
        # [threshold, cap] sharing one spacing. The non-bracket nodes are split
        # between the two regions in proportion to their lengths, so the region above
        # the threshold gets more bins the further the cap sits above it (one bin
        # while it is shorter than a before-bin, a second once it is longer, etc.)
        # rather than a fixed count.
        n_nonbracket = N_EXPERIENCE_NODES - len(bracket)
        bracket_start = bracket[0]
        span = bracket_start + (cap - threshold)
        n_before = int(round(n_nonbracket * bracket_start / span))
        n_before = min(max(n_before, 1), n_nonbracket - 1)
        n_after = n_nonbracket - n_before
        before = np.linspace(0.0, bracket_start, n_before, endpoint=False)
        after = np.linspace(threshold, cap, n_after + 1)[1:]
        grid = np.concatenate(
            [np.round(before * 2) / 2, np.asarray(bracket), np.round(after * 2) / 2]
        )
    # Guard strict monotonicity against rounding collisions, pin the ends.
    grid[0] = 0.0
    for i in range(1, len(grid)):
        if grid[i] <= grid[i - 1]:
            grid[i] = grid[i - 1] + 0.5
    grid[-1] = float(cap)
    return grid


def build_working_grid_table(
    max_init_exp_by_sex,
    max_working_period,
    thresholds,
    bracket_quantum_by_sex,
    n_periods,
):
    """Build the ``(n_sexes, n_periods, N)`` working grid table from the derived
    thresholds/brackets/caps and the fixed shape constants above."""
    n_sexes = len(max_init_exp_by_sex)
    table = np.zeros((n_sexes, n_periods, N_EXPERIENCE_NODES), dtype=float)
    for sex in range(n_sexes):
        threshold = float(thresholds[sex])
        bracket = vli_bracket(threshold, float(bracket_quantum_by_sex[sex]))
        for period in range(n_periods):
            cap = _cap(max_init_exp_by_sex[sex], max_working_period, period)
            table[sex, period] = experience_grid_row(cap, threshold, bracket)
    return table


def validate_working_grid_table(table, max_init_exp_by_sex, max_working_period):
    """Check the cap invariant: each grid is 0..cap, strictly increasing, N nodes.

    ``cap = max_init_exp[sex] + min(period, max_working_period)`` -- nobody is
    assigned experience beyond what is attainable at their age.
    """
    n_sexes, n_periods, n_nodes = table.shape
    if n_nodes != N_EXPERIENCE_NODES:
        raise ValueError(
            f"experience grid has {n_nodes} nodes, expected {N_EXPERIENCE_NODES}."
        )
    for sex in range(n_sexes):
        for period in range(n_periods):
            row = table[sex, period]
            cap = _cap(max_init_exp_by_sex[sex], max_working_period, period)
            if row[0] != 0.0:
                raise ValueError(f"grid[{sex},{period}] does not start at 0: {row}")
            if not np.all(np.diff(row) > 0):
                raise ValueError(
                    f"grid[{sex},{period}] is not strictly increasing: {row}"
                )
            if row[-1] > cap + 1e-9:
                raise ValueError(
                    f"grid[{sex},{period}] top {row[-1]} exceeds cap "
                    f"max_init_exp+min(period, max_working_period) = {cap}."
                )


def build_experience_grid_working_by_sex_period(specs):
    """Sex-specific, real-year, age-dependent working experience grid per period.

    Built by the two-regime rule in ``experience_grids.py`` (uniform to the cap
    below the VLI threshold; equal bins + dense bracket + equal bins above it). All
    economically meaningful inputs are derived from the model, not hardcoded:

    * ``cap = max_exp_diff_period_working[sex] + min(period, max_working_period)``,
      with ``max_working_period = max_ret_age - start_age`` (experience is frozen
      once past the last working age);
    * ``thresholds`` = ``experience_threshold_very_long_insured`` (= 45 / the
      estimated credited-periods factor);
    * the bracket quantum = each sex's reachable experience step -- full-time
      (whole year) for men, part-time (``exp_increase_part_time``) for women.

    The working "experience" state is stored as real years directly, so this grid is
    already in the stored units.
    """
    max_init_exp_by_sex = np.asarray(specs["max_exp_diff_period_working"])
    max_working_period = specs["max_ret_age"] - specs["start_age"]
    # Men (sex 0) accrue whole years (full-time only); women (sex 1) can accrue the
    # part-time step, so their bracket sits on half-years.
    bracket_quantum_by_sex = np.array([1.0, specs["exp_increase_part_time"]])
    table = build_working_grid_table(
        max_init_exp_by_sex=max_init_exp_by_sex,
        max_working_period=max_working_period,
        thresholds=specs["experience_threshold_very_long_insured"],
        bracket_quantum_by_sex=bracket_quantum_by_sex,
        n_periods=specs["n_periods"],
    )
    validate_working_grid_table(table, max_init_exp_by_sex, max_working_period)
    return jnp.asarray(table)  # (n_sexes, n_periods, N_EXPERIENCE_NODES)


def build_experience_grid_cap_by_period(working_table):
    """Per-(sex, period) cap of the working grid (its top node).

    Each sex's grid clips to ``min(max_exp_diff_period_working[sex] + period,
    top_anchor)``, so the top node is sex-specific. Retired states reuse the same
    per-(sex, period) axis (see ``experience_grid_from_state`` /
    ``scale_experience_years``), so this cap is also the rescaling constant that maps
    pension points onto the grid.
    """
    return jnp.asarray(np.asarray(working_table)[:, :, -1])  # (n_sexes, n_periods)
