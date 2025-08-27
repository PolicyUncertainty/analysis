from unicodedata import normalize

import numpy as np
import pandas as pd


def calc_labor_supply_choice(df):

    index = pd.MultiIndex.from_product(
        [
            [0, 1],
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


def calc_labor_supply_variance(df):

    state_cond = ["sex", "education", "period"]
    columns = state_cond + ["choice"]

    index = pd.MultiIndex.from_product(
        [
            [0, 1],
            [0, 1],
            np.arange(0, 45),
            [0, 1, 2, 3],
        ],
        names=columns,
    )

    choice_shares = (
        df.groupby(state_cond, observed=False)["choice"]
        .value_counts(normalize=True)
        .reindex(index, fill_value=0.0)
    )

    state_counts = df.groupby(state_cond).size()

    variance = choice_shares * (1 - choice_shares)
    for sex_var in [0, 1]:
        for edu_var in [0, 1]:
            for period in range(0, 45):
                variance.loc[(sex_var, edu_var, period)] = (
                    variance.loc[(sex_var, edu_var, period)]
                    / state_counts.loc[(sex_var, edu_var, period)]
                ).values

    return variance
