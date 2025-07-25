import pickle

import numpy as np
import pandas as pd
import scipy.optimize as opt

from export_results.tools import create_realized_taste_shock


def calc_compensated_variation(df_base, df_cf, params, specs):
    """We assume the dfs have resetted index."""
    df_base = create_real_utility(df_base, specs)
    df_cf = create_real_utility(df_cf, specs)

    df_base = add_number_cons_scale(df_base, specs)
    df_cf = add_number_cons_scale(df_cf, specs)

    n_agents = df_base["agent"].nunique()
    cv = calc_adjusted_scale(df_base, df_cf, params, specs, n_agents)
    return cv


def create_real_utility(df, specs):
    df = create_realized_taste_shock(df, specs)
    df.loc[:, "real_util"] = df["utility"] + df["real_taste_shock"]
    return df


def calc_adjusted_scale(df_base, df_count, params, specs, n_agents):
    # First construct the discounted sum of utilities for the counterfactual scenario
    disc_sum_cf = create_disc_sum(df_count, specs)

    # Then we construct the relevant objects to be able to scale consumption,
    # such that it matches the discounted sum from above
    sex_values = df_base["sex"].values
    mu_vector = np.ones_like(sex_values) * params["mu"]

    # Generate not scaled utility by substracting from random utility 1 / (1 - mu)
    not_scaled_utility = df_base["real_taste_shock"].values - 1 / (1 - mu_vector)
    # Now scaled utility
    utility_to_scale = df_base["real_util"].values - not_scaled_utility

    # Generate the discount factor for the base dataframe
    disc_factor_base = specs["discount_factor"] ** df_base["period"].values

    partial_adjustment = lambda scale_in: create_adjusted_difference(
        utility_to_scale=utility_to_scale,
        not_scaled_utility=not_scaled_utility,
        disc_factor_base=disc_factor_base,
        disc_sum_cf=disc_sum_cf,
        n_agents=n_agents,
        mu_vector=mu_vector,
        scale=scale_in,
    )

    scale = opt.brentq(partial_adjustment, -0.5, 10)

    return scale


def create_adjusted_difference(
    utility_to_scale,
    not_scaled_utility,
    disc_factor_base,
    disc_sum_cf,
    n_agents,
    mu_vector,
    scale,
):
    scaled_utility = utility_to_scale * ((1 + scale) ** (1 - mu_vector))

    adjusted_utility = scaled_utility + not_scaled_utility
    adjusted_disc_sum = (adjusted_utility * disc_factor_base).sum() / n_agents

    return adjusted_disc_sum - disc_sum_cf


def create_disc_sum(df, specs):
    beta = specs["discount_factor"]
    df.loc[:, "disc_util"] = df["real_util"] * (beta ** df["period"])

    return df.groupby("agent")["disc_util"].sum().mean()


def add_number_cons_scale(df, specs):
    education = df["education"].values
    has_partner_int = (df["partner_state"].values > 0).astype(int)
    period = df["period"].values
    sex = df["sex"].values
    nb_children = specs["children_by_state"][sex, education, has_partner_int, period]
    hh_size = 1 + has_partner_int + nb_children
    df.loc[:, "cons_scale"] = np.sqrt(hh_size)
    return df
