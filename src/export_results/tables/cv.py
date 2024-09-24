import pickle

import numpy as np
import pandas as pd
import scipy.optimize as opt
from export_results.tools import create_discounted_sum_utilities
from export_results.tools import create_realized_taste_shock


def calc_compensated_variation(df_base, df_cf, params, specs):
    df_base = create_real_utility(df_base)
    df_cf = create_real_utility(df_cf)

    df_base.reset_index(inplace=True)
    df_cf.reset_index(inplace=True)

    df_base = add_number_cons_scale(df_base, specs)
    df_cf = add_number_cons_scale(df_cf, specs)

    n_agents = df_base["agent"].nunique()
    cv = calc_adjusted_scale(df_base, df_cf, params, n_agents)
    return cv


def create_real_utility(df):
    df = create_realized_taste_shock(df)
    df.loc[:, "real_util"] = df["utility"] + df["real_taste_shock"]
    return df


def calc_adjusted_scale(df_base, df_count, params, n_agents):
    mu = params["mu"]

    disc_sum_base = create_disc_sum(df_base, params)
    disc_sum_count = create_disc_sum(df_count, params)

    df_count.loc[:, "cons_utility"] = (
        ((df_count["consumption"] / df_count["cons_scale"]) ** (1 - mu)) - 1
    ) / (1 - mu)

    df_count.loc[:, "non_cons_utility"] = (
        df_count["real_util"] - df_count["cons_utility"]
    )

    partial_adjustment = lambda scale_in: create_adjusted_difference(
        df_count, disc_sum_base, n_agents, params, scale_in
    )

    scale = opt.brentq(partial_adjustment, -1, 10)

    return 1 + scale


def create_disc_sum(df, params, reset_index=False):
    beta = params["beta"]
    if reset_index:
        df.reset_index(inplace=True)
    n_agents = df["agent"].nunique()
    df.loc[:, "disc_util"] = df["real_util"] * (beta ** df["period"])
    return df.groupby("agent")["disc_util"].sum().median()


def create_adjusted_difference(df_count, disc_sum_base, n_agents, params, scale):
    mu = params["mu"]
    beta = params["beta"]
    adjusted_cons = df_count["consumption"] * (1 + scale)
    adjusted_cons_util = (
        ((adjusted_cons / df_count["cons_scale"]) ** (1 - mu)) - 1
    ) / (1 - mu)
    adjusted_real_util = adjusted_cons_util + df_count["non_cons_utility"]
    adjusted_disc_sum = (
        adjusted_real_util * (beta ** df_count["period"])
    ).sum() / n_agents
    return adjusted_disc_sum - disc_sum_base


def add_number_cons_scale(df, specs):
    education = df["education"].values
    has_partner_int = (df["partner_state"].values > 0).astype(int)
    period = df["period"].values
    nb_children = specs["children_by_state"][0, education, has_partner_int, period]
    hh_size = 1 + has_partner_int + nb_children
    df.loc[:, "cons_scale"] = np.sqrt(hh_size)
    return df
