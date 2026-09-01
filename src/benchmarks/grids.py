"""Grid builders for pinning down a good druedahl_jorgensen assets_begin_of_period
grid (see check_dj_grid.py / run_dj_grid_timing.py / dj_candidates.py).

These build grids purely from a *sample of wealth points* (e.g. fues's own solved
endogenous grid) -- deliberately independent of assets_end_of_period, since
begin-of-period wealth and end-of-period savings are different quantities and one
isn't a derivation of the other.
"""

import numpy as np


def pool_positive_finite(values):
    """Flatten ``values`` to a 1D array of finite, strictly positive entries.

    Used to turn a solved model's ``endog_grid`` (shape varies by period/discrete
    state, NaN-padded wherever the upper envelope trimmed points) into a plain
    sample of wealth points to build candidate grids from.
    """
    arr = np.asarray(values).ravel()
    return arr[np.isfinite(arr) & (arr > 0)]


def build_quantile_grid(sample, n_points):
    """``n_points`` evenly spaced quantiles of ``sample``.

    Self-tuning: dense wherever ``sample`` already has mass, without assuming any
    particular parametric shape for where that density should be.
    """
    sample = np.sort(np.asarray(sample))
    return np.unique(np.quantile(sample, np.linspace(0.0, 1.0, n_points)))


def build_power_spaced_grid(lower, upper, n_points, power):
    """``n_points`` between ``lower`` and ``upper``, spaced by ``(i / (n-1)) **
    power``. ``power=1`` is uniform; ``power > 1`` concentrates points near
    ``lower`` (i.e. near the credit constraint, if ``lower`` is ~0).
    """
    unit = np.linspace(0.0, 1.0, n_points) ** power
    return lower + (upper - lower) * unit
