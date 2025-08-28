import numpy as np

from estimation.msm.scripts.labor_supply_moments import (
    calc_labor_supply_choice,
    calc_labor_supply_variance,
)
from estimation.msm.scripts.labor_transition_moments import (
    calc_transition_to_work,
    calc_variance_labor_transitions,
)
from estimation.msm.scripts.wealth_moments import (
    calc_wealth_mean_variance,
    calc_wealth_moment,
)


def calc_all_moments(df, empirical=False, men_only=True):
    """
    Calculate all moments from the given DataFrame.
    """
    labor_supply_moments = calc_labor_supply_choice(df, men_only=men_only)
    labor_transitions_moments = calc_transition_to_work(df, men_only=men_only)
    median_wealth_moments = calc_wealth_moment(
        df, empirical=empirical, men_only=men_only
    )

    # Transform to numpy arrays and concatenate
    moments = np.concatenate(
        [
            labor_supply_moments.values,
            labor_transitions_moments.values,
            median_wealth_moments.values,
        ]
    )
    return moments


def calc_variance_of_moments(df, men_only=True):
    """
    Calculate the variance of all moments from the given DataFrame.
    """

    labor_supply_variance = calc_labor_supply_variance(df, men_only=men_only)
    labor_transition_variance = calc_variance_labor_transitions(df, men_only=men_only)
    wealth_mom_vars = calc_wealth_mean_variance(df, men_only=men_only)

    variances = np.concatenate(
        [
            labor_supply_variance.values,
            labor_transition_variance.values,
            wealth_mom_vars.values,
        ]
    )

    return variances
