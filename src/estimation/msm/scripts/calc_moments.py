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


def calc_all_moments(df, empirical=False):
    """
    Calculate all moments from the given DataFrame.
    """
    labor_supply_moments = calc_labor_supply_choice(df)
    labor_transitions_moments = calc_transition_to_work(df)
    median_wealth_moments = calc_wealth_moment(df, empirical=empirical)

    # Transform to numpy arrays and concatenate
    moments = np.concatenate(
        [
            labor_supply_moments.values,
            labor_transitions_moments.values,
            median_wealth_moments.values,
        ]
    )
    return moments


def calc_variance_of_moments(df):
    """
    Calculate the variance of all moments from the given DataFrame.
    """

    labor_supply_variance = calc_labor_supply_variance(df)
    labor_transition_variance = calc_variance_labor_transitions(df)
    wealth_mom_vars = calc_wealth_mean_variance(df)

    variances = np.concatenate(
        [
            labor_supply_variance.values,
            labor_transition_variance.values,
            wealth_mom_vars.values,
        ]
    )

    return variances
