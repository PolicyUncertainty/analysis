import pickle

import pandas as pd
import scipy.optimize as opt
from export_results.tools import create_discounted_sum_utilities
from export_results.tools import create_realized_taste_shock


def calc_compensated_variation(df_base, df_cf, params):
    df_base = create_realized_taste_shock(df_base)
    df_cf = create_realized_taste_shock(df_cf)

    df_base["real_util"] = df_base["utility"] + df_base["real_taste_shock"]
    df_cf["real_util"] = df_cf["utility"] + df_cf["real_taste_shock"]

    df_base.reset_index(inplace=True)
    df_cf.reset_index(inplace=True)
    n_agents = df_base["agent"].nunique()
    cv = calc_adjusted_scale(df_base, df_cf, params, n_agents)
    return cv


def calc_adjusted_scale(df_base, df_count, params, n_agents):
    mu = params["mu"]
    beta = params["beta"]
    disc_sum_base = (
        df_base["real_util"] * (beta ** df_base["period"])
    ).sum() / n_agents

    df_count["cons_utility"] = (df_count["consumption"] ** (1 - mu)) / (1 - mu)

    df_count["non_cons_utility"] = df_count["real_util"] - df_count["cons_utility"]

    partial_adjustment = lambda scale_in: create_adjusted_difference(
        df_count, disc_sum_base, n_agents, params, scale_in
    )

    scale = opt.brentq(partial_adjustment, -1, 10)

    return scale


def create_adjusted_difference(df_count, disc_sum_base, n_agents, params, scale):
    mu = params["mu"]
    beta = params["beta"]
    adjusted_cons = df_count["consumption"] * (1 + scale)
    adjusted_cons_util = (adjusted_cons ** (1 - mu)) / (1 - mu)
    adjusted_real_util = adjusted_cons_util + df_count["non_cons_utility"]
    adjusted_disc_sum = (
        adjusted_real_util * (beta ** df_count["period"])
    ).sum() / n_agents
    print(scale, adjusted_disc_sum - disc_sum_base)
    return adjusted_disc_sum - disc_sum_base
