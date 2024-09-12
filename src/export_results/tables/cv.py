import pickle

import pandas as pd
import scipy.optimize as opt
from export_results.tools import create_discounted_sum_utilities
from export_results.tools import create_realized_taste_shock


def calc_compensated_variation(path_dict):
    df_base = pd.read_pickle(
        path_dict["intermediate_data"] + "sim_data/data_real_scale_05.pkl"
    ).reset_index()
    df_count = pd.read_pickle(
        path_dict["intermediate_data"] + "sim_data/data_subj_scale_05.pkl"
    ).reset_index()

    df_base = create_realized_taste_shock(df_base)
    df_count = create_realized_taste_shock(df_count)

    df_base["real_util"] = df_base["utility"] + df_base["real_taste_shock"]
    df_count["real_util"] = df_count["utility"] + df_count["real_taste_shock"]

    params = pickle.load(open(path_dict["est_results"] + "est_params.pkl", "rb"))
    n_agents = df_base["agent"].nunique()
    cv = calc_adjusted_scale(df_base, df_count, params, n_agents)
    print(cv)


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
