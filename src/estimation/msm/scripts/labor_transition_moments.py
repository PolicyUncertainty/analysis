import numpy as np
import pandas as pd


def calc_transition_to_work(df):
    """
    Calculate the labor transitions by age bins.
    """
    # Define age bins
    period_bins = np.arange(0, 50, 5)
    df["period_bin"] = pd.cut(df["period"], bins=period_bins, right=False)

    df["has_worked"] = df["lagged_choice"].isin([2, 3]).astype(int)
    df["working_choice"] = df["choice"].isin([2, 3]).astype(int)

    index_full = pd.MultiIndex.from_product(
        [
            [0, 1],
            [0, 1],
            period_bins[:-1],
            [0, 1],
        ],
        names=["sex", "education", "period_bin", "has_worked"],
    )

    # Group by age bins and calculate transitions
    transitions = df.groupby(
        ["sex", "education", "period_bin", "has_worked"], observed=False
    )["working_choice"].mean()

    transitions_full = transitions.reindex(index_full, fill_value=0.0)

    return transitions_full


def calc_variance_labor_transitions(df):
    """
    Calculate the variance of the mean wealth by age.
    """

    period_bins = np.arange(0, 50, 5)
    df["period_bin"] = pd.cut(df["period"], bins=period_bins, right=False)

    df["has_worked"] = df["lagged_choice"].isin([2, 3]).astype(int)
    df["working_choice"] = df["choice"].isin([2, 3]).astype(int)

    columns = ["sex", "education", "period_bin", "has_worked"]

    full_index = pd.MultiIndex.from_product(
        [
            [0, 1],
            [0, 1],
            period_bins[:-1],
            [0, 1],
        ],
        names=columns,
    )

    transition_variance = (
        df.groupby(columns, observed=False)["working_choice"]
        .var()
        .reindex(full_index, fill_value=0.0)
    )

    n_obs = df.groupby(columns, observed=False).size().reindex(full_index, fill_value=1)

    transition_variance /= n_obs
    # Replace all nans with zeros
    transition_variance = transition_variance.fillna(0.0)
    return transition_variance
