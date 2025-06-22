import numpy as np
import pandas as pd


def calc_all_moments(df):
    """
    Calculate all moments from the given DataFrame.
    """
    labor_supply_moments = calc_labor_supply_choice(df)
    labor_transitions_moments = calc_labor_transitions_by_age_bins(df)
    # median_wealth_moments = calc_median_wealth_by_age(df)

    # Transform to numpy arrays and concatenate
    moments = np.concatenate(
        [
            labor_supply_moments.values,
            labor_transitions_moments.values,
            # median_wealth_moments.values,
        ]
    )
    return moments


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


def calc_median_wealth_by_age(df):
    """
    Calculate the median wealth by age.
    """
    age_bins = np.arange(0, 50, 3)
    df["period_bin"] = pd.cut(df["period"], bins=age_bins, right=False)

    median_wealth = df.groupby(["sex", "education", "period_bin"], observed=False)[
        "assets_begin_of_period"
    ].median()

    # Create a full index for ages 0 to 79
    full_index = pd.MultiIndex.from_product(
        [
            [0],
            [0, 1],
            age_bins[:-1],
        ],
        names=["sex", "education", "period_bin"],
    )
    median_wealth_full = median_wealth.reindex(full_index, fill_value=np.nan)

    return median_wealth_full
