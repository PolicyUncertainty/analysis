"""Parameterized grid builders for the dcegm engine benchmark.

Kept separate from ``model_code.wealth_and_budget.assets_grid`` /
``model_code.state_space.experience`` on purpose: those hold the hand-tuned,
non-uniform production grids, while the benchmark needs simple grids whose size can
be dialed up/down to stress-test runtime and GPU memory.
"""

import numpy as np


def build_assets_end_of_period_grid(n_points, max_wealth_raw=10_000_000):
    """End-of-period assets grid, in the same raw-currency units as the production
    grid (``model_code.wealth_and_budget.assets_grid.create_end_of_period_assets``).
    Gets divided by ``specs["wealth_unit"]`` downstream, exactly like the production
    grid.
    """
    return np.linspace(0, max_wealth_raw, n_points)


def build_assets_begin_of_period_grid(n_points, max_wealth_model_units=1_000):
    """Begin-of-period assets grid, already in model units (i.e. the same scale as
    ``assets_end_of_period`` after division by ``specs["wealth_unit"]``). Required by
    dcegm when ``upper_envelope["method"] == "druedahl_jorgensen"``. Strictly
    positive and increasing, since dcegm prepends 0 itself.
    """
    return np.linspace(0, max_wealth_model_units, n_points + 1)[1:]


def build_experience_grid(n_points):
    """Experience grid on [0, 1], matching the production grid's range."""
    return np.linspace(0.0, 1.0, n_points)
