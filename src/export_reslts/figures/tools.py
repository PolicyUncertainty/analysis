import numpy as np


def create_realized_taste_shock(df):
    df["real_taste_shock"] = np.nan
    df.loc[df["choice"] == 0, "real_taste_shock"] = df.loc[
        df["choice"] == 0, "taste_shock_0"
    ]
    df.loc[df["choice"] == 1, "real_taste_shock"] = df.loc[
        df["choice"] == 1, "taste_shock_1"
    ]
    df.loc[df["choice"] == 2, "real_taste_shock"] = df.loc[
        df["choice"] == 2, "taste_shock_2"
    ]
    return df


def create_discounted_sum_utilities(df, beta, utility_col="real_util"):
    mean_utility = df.groupby("period")[utility_col].mean().sort_index().values

    max_period = df["period"].max()
    # reverse loop over range
    for i in range(max_period - 1, -1, -1):
        mean_utility[i] += mean_utility[i + 1] * beta

    return mean_utility
