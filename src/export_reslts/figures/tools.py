import numpy as np
import scipy.optimize as opt


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


def create_step_function_values(specs, base_policy_state, plot_span):
    new_value_periods = specs["policy_step_periods"] + 1

    step_function_vals = np.zeros(plot_span) + base_policy_state
    for i in range(1, plot_span):
        if np.isin(i, new_value_periods):
            step_function_vals[i] = step_function_vals[i - 1] + 1
        else:
            step_function_vals[i] = step_function_vals[i - 1]

    step_function_vals = step_function_vals * specs["SRA_grid_size"] + specs["min_SRA"]
    return step_function_vals


def calc_adjusted_difference(df_base, df_count, params, n_agents):
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
    scale = opt.brentq(partial_adjustment, -1, 1)

    return scale


def create_adjusted_difference(df_count, disc_sum_base, n_agents, params, scale):
    mu = params["mu"]
    beta = params["beta"]
    adjusted_cons = df_count["cons_utility"] * (1 + scale)
    adjusted_cons_util = (adjusted_cons ** (1 - mu)) / (1 - mu)
    adjusted_real_util = adjusted_cons_util + df_count["non_cons_utility"]
    adjusted_disc_sum = (
        adjusted_real_util * (beta ** df_count["period"])
    ).sum() / n_agents
    return adjusted_disc_sum - disc_sum_base
