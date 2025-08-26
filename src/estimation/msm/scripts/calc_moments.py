import numpy as np
import pandas as pd

from estimation.msm.scripts.wealth_moments import (
    calc_variance_of_the_empirical_mean,
    calc_wealth_moment,
)


def calc_all_moments(df, empirical=False):
    """
    Calculate all moments from the given DataFrame.
    """
    # labor_supply_moments = calc_labor_supply_choice(df)
    # labor_transitions_moments = calc_labor_transitions_by_age_bins(df)
    median_wealth_moments = calc_wealth_moment(df, empirical=empirical)

    # Transform to numpy arrays and concatenate
    moments = np.concatenate(
        [
            # labor_supply_moments.values,
            # labor_transitions_moments.values,
            median_wealth_moments.values,
        ]
    )
    return moments


def calc_variance_of_moments(df):
    """
    Calculate the variance of all moments from the given DataFrame.
    """
    wealth_mom_vars = calc_variance_of_the_empirical_mean(df)
    return wealth_mom_vars.values


def calc_labor_supply_choice(df):

    index = pd.MultiIndex.from_product(
        [
            [0],
            [0, 1],
            np.arange(0, 45),
            [0, 1, 2, 3],
        ],
        names=["sex", "education", "period", "choice"],
    )

    choice_shares = df.groupby(["sex", "education", "period"], observed=False)[
        "choice"
    ].value_counts(normalize=True)

    choice_shares_full = choice_shares.reindex(index, fill_value=0.0)
    return choice_shares_full


def calc_labor_transitions_by_age_bins(df):
    """
    Calculate the labor transitions by age bins.
    """
    # Define age bins
    age_bins = np.arange(0, 50, 5)
    df["period_bin"] = pd.cut(df["period"], bins=age_bins, right=False)

    df["has_worked"] = df["lagged_choice"].isin([2, 3]).astype(int)
    df["working_choice"] = df["choice"].isin([2, 3]).astype(int)

    # Group by age bins and calculate transitions
    transitions = df.groupby(
        ["sex", "education", "period_bin", "has_worked"], observed=False
    )["working_choice"].value_counts(normalize=True)

    index = pd.MultiIndex.from_product(
        [
            [0],
            [0, 1],
            age_bins[:-1],
            [0, 1],
            [0, 1],
        ],
        names=["sex", "education", "period_bin", "has_worked", "working_choice"],
    )
    transitions_full = transitions.reindex(index, fill_value=0.0)

    return transitions_full
