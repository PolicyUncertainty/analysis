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

    disc_sum_cf = create_disc_sum(df_count, params)

    cons_base = df_base["consumption"].values
    cons_scale_base = df_base["cons_scale"].values

    cons_utility_base = (((cons_base / cons_scale_base) ** (1 - mu)) - 1) / (1 - mu)

    no_cons_utility_base = df_base["real_util"].values - cons_utility_base

    disc_factor_base = params["beta"] ** df_count["period"].values

    partial_adjustment = lambda scale_in: create_adjusted_difference(
        cons_base=cons_base,
        cons_scale_base=cons_scale_base,
        no_cons_utility_base=no_cons_utility_base,
        disc_factor_base=disc_factor_base,
        disc_sum_cf=disc_sum_cf,
        n_agents=n_agents,
        params=params,
        scale=scale_in,
    )

    scale = opt.brentq(partial_adjustment, -0.5, 10)

    return scale


def create_adjusted_difference(
    cons_base,
    cons_scale_base,
    no_cons_utility_base,
    disc_factor_base,
    disc_sum_cf,
    n_agents,
    params,
    scale,
):
    adjusted_cons = cons_base * (1 + scale)
    adjusted_cons_util = cons_utility(adjusted_cons, cons_scale_base, params["mu"])

    adjusted_util = adjusted_cons_util + no_cons_utility_base
    adjusted_disc_sum = (adjusted_util * disc_factor_base).sum() / n_agents

    return adjusted_disc_sum - disc_sum_cf


def cons_utility(consumption, cons_scale, mu):
    return ((consumption / cons_scale) ** (1 - mu) - 1) / (1 - mu)


def create_disc_sum(df, params, reset_index=False):
    beta = params["beta"]
    if reset_index:
        df.reset_index(inplace=True)
    df.loc[:, "disc_util"] = df["real_util"] * (beta ** df["period"])

    return df.groupby("agent")["disc_util"].sum().mean()


def add_number_cons_scale(df, specs):
    education = df["education"].values
    has_partner_int = (df["partner_state"].values > 0).astype(int)
    period = df["period"].values
    nb_children = specs["children_by_state"][0, education, has_partner_int, period]
    hh_size = 1 + has_partner_int + nb_children
    df.loc[:, "cons_scale"] = np.sqrt(hh_size)
    return df
