"""Candidate assets_begin_of_period grids for druedahl_jorgensen.

Shared between check_dj_grid.py (accuracy vs. the fues reference) and
run_dj_grid_timing.py (solve cost), so both scripts sweep the identical candidate
set and their results can be read side by side as one accuracy-vs-cost picture.

Two independent shape families, both built purely from a wealth *sample* (see
grids.pool_positive_finite) -- not from assets_end_of_period:

  - "quantile": self-tuning, dense wherever the sample already has mass.
  - "power": one explicit concentration knob (``power``) pulling points toward the
    bottom of the sample's range, independent of point count.
"""

from benchmarks.grids import build_power_spaced_grid, build_quantile_grid

CANDIDATE_SPECS = [
    {"name": "quantile_10", "kind": "quantile", "n_points": 10},
    {"name": "quantile_15", "kind": "quantile", "n_points": 15},
    {"name": "quantile_20", "kind": "quantile", "n_points": 20},
    {"name": "quantile_28", "kind": "quantile", "n_points": 28},
    {"name": "quantile_40", "kind": "quantile", "n_points": 40},
    {"name": "power3_10", "kind": "power", "n_points": 10, "power": 3.0},
    {"name": "power3_15", "kind": "power", "n_points": 15, "power": 3.0},
    {"name": "power3_20", "kind": "power", "n_points": 20, "power": 3.0},
    {"name": "power3_28", "kind": "power", "n_points": 28, "power": 3.0},
    {"name": "power3_40", "kind": "power", "n_points": 40, "power": 3.0},
]


def build_candidate_grid(spec, endog_grid_sample):
    if spec["kind"] == "quantile":
        return build_quantile_grid(endog_grid_sample, spec["n_points"])
    if spec["kind"] == "power":
        lower = float(endog_grid_sample.min())
        upper = float(endog_grid_sample.max())
        return build_power_spaced_grid(lower, upper, spec["n_points"], spec["power"])
    raise ValueError(f"Unknown candidate kind: {spec['kind']!r}")
