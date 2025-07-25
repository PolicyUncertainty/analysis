import numpy as np
import pandas as pd


def calc_all_moments(df, empirical=False):
    """
    Calculate all moments from the given DataFrame.
    """
    labor_supply_moments = calc_labor_supply_choice(df)
    labor_transitions_moments = calc_labor_transitions_by_age_bins(df)
    median_wealth_moments = calc_median_wealth_by_age(df, empirical=empirical)

    # Transform to numpy arrays and concatenate
    moments = np.concatenate(
        [
            labor_supply_moments.values,
            labor_transitions_moments.values,
            median_wealth_moments.values,
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


def calc_median_wealth_by_age(df, empirical):
    """
    Calculate the median wealth by age.
    """

    periods = np.arange(0, 71)
    # Create a full index for ages 0 to 79
    full_index = pd.MultiIndex.from_product(
        [
            [0],
            [0, 1],
            periods,
        ],
        names=["sex", "education", "period"],
    )

    if empirical:
        # Create mask for wealth smaller than the 95th percentile
        wealth_mask = df["assets_begin_of_period"] < df[
            "assets_begin_of_period"
        ].quantile(0.95)
        rolling_three = (
            df[wealth_mask]
            .groupby(["sex", "education", "period"], observed=False)[
                "assets_begin_of_period"
            ]
            .mean()
            .rolling(3)
            .mean()
            .reindex(full_index, fill_value=np.nan)
        )
        rolling_five = (
            df[wealth_mask]
            .groupby(["sex", "education", "period"], observed=False)[
                "assets_begin_of_period"
            ]
            .mean()
            .rolling(5)
            .mean()
            .reindex(full_index, fill_value=np.nan)
        )
        mean = (
            df[wealth_mask]
            .groupby(["sex", "education", "period"], observed=False)[
                "assets_begin_of_period"
            ]
            .mean()
            .reindex(full_index, fill_value=np.nan)
        )
        # Take rolling three as default. Assign for the first two the mean
        first_two = (slice(None), slice(None), [0, 1])
        rolling_three.loc[first_two] = mean.loc[first_two]

        # Assign for the last ten periods the rolling five
        last_twenty = (slice(None), slice(None), np.arange(50, 71))
        rolling_three.loc[last_twenty] = rolling_five.loc[last_twenty]
        wealth_mom = rolling_three
    else:
        wealth_mom = df.groupby(["sex", "education", "period"], observed=False)[
            "assets_begin_of_period"
        ].mean()

        wealth_mom = wealth_mom.reindex(full_index, fill_value=np.nan)

    return wealth_mom
