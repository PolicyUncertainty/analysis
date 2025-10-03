import numpy as np
import pandas as pd


def calc_wealth_moment(df, empirical, men_only=False):
    """
    Calculate the median wealth by age.
    """

    df["has_partner"] = (df["partner_state"] > 0).astype(int)

    # Create bins of 5 years from period 0 to 59
    period_bins = np.arange(0, 65, 5)
    df["period_bin"] = pd.cut(df["period"], bins=period_bins, right=False, labels=False)

    period_bin_idx = np.arange(len(period_bins) - 1)

    columns = ["sex", "education", "has_partner", "period_bin"]

    if men_only:
        index_col_first = [0]
    else:
        index_col_first = [0, 1]

    # Create a full index for ages 0 to 79
    full_index = pd.MultiIndex.from_product(
        [
            index_col_first,
            [0, 1],
            [0, 1],
            period_bin_idx,
        ],
        names=columns,
    )

    if empirical:
        # Create mask for wealth smaller than the 95th percentile
        # wealth_mask = df["assets_begin_of_period"] < df[
        #     "assets_begin_of_period"
        # ].quantile(0.95)
        # rolling_three = (
        #     df[wealth_mask]
        #     .groupby(["sex", "education", "period"], observed=False)[
        #         "assets_begin_of_period"
        #     ]
        #     .mean()
        #     .rolling(3)
        #     .mean()
        #     .reindex(full_index, fill_value=np.nan)
        # )
        # rolling_five = (
        #     df[wealth_mask]
        #     .groupby(["sex", "education", "period"], observed=False)[
        #         "assets_begin_of_period"
        #     ]
        #     .mean()
        #     .rolling(5)
        #     .mean()
        #     .reindex(full_index, fill_value=np.nan)
        # )
        mean = (
            df.groupby(columns, observed=False)["assets_begin_of_period"]
            .mean()
            .reindex(full_index, fill_value=np.nan)
        )

        # # Take rolling three as default. Assign for the first two the mean
        # first_two = (slice(None), slice(None), [0, 1])
        # rolling_three.loc[first_two] = mean.loc[first_two]
        #
        # # Assign for the last ten periods the rolling five
        # last_twenty = (slice(None), slice(None), np.arange(50, 60))
        # rolling_three.loc[last_twenty] = rolling_five.loc[last_twenty]
        # wealth_mom = rolling_three
        wealth_mom = mean
    else:
        wealth_mom = df.groupby(columns, observed=False)[
            "assets_begin_of_period"
        ].mean()

        wealth_mom = wealth_mom.reindex(full_index, fill_value=np.nan)

    wealth_mom = wealth_mom.fillna(0.0)
    return wealth_mom


def calc_wealth_mean_variance(df, men_only=False):
    """
    Calculate the variance of the mean wealth by age.
    """

    df["has_partner"] = (df["partner_state"] > 0).astype(int)

    # Create bins of 5 years from period 0 to 59
    period_bins = np.arange(0, 65, 5)
    df["period_bin"] = pd.cut(df["period"], bins=period_bins, right=False, labels=False)

    period_bin_idx = np.arange(len(period_bins) - 1)

    columns = ["sex", "education", "has_partner", "period_bin"]

    if men_only:
        index_col_first = [0]
    else:
        index_col_first = [0, 1]

    # Create a full index for ages 0 to 79
    full_index = pd.MultiIndex.from_product(
        [
            index_col_first,
            [0, 1],
            [0, 1],
            period_bin_idx,
        ],
        names=columns,
    )

    wealth_mask = df["assets_begin_of_period"] < df["assets_begin_of_period"].quantile(
        0.95
    )

    wealth_variance = (
        df[wealth_mask]
        .groupby(columns, observed=False)["assets_begin_of_period"]
        .var()
        .reindex(full_index, fill_value=np.nan)
    )

    n_obs = (
        df[wealth_mask]
        .groupby(columns, observed=False)
        .size()
        .reindex(full_index, fill_value=np.nan)
    )

    wealth_variance /= n_obs
    # Replace all nans with zeros
    wealth_variance = wealth_variance.fillna(0.0)
    return wealth_variance
