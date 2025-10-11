import numpy as np
import scipy.optimize as opt


def calc_compensated_variation(df_base, df_cf, params, specs):
    """We assume the dfs have resetted index."""
    df_base = create_real_utility(df_base, specs)
    df_cf = create_real_utility(df_cf, specs)

    df_base = add_cons_scale_and_adult_hh_size(df_base, specs)
    df_cf = add_cons_scale_and_adult_hh_size(df_cf, specs)

    n_agents = df_base["agent"].nunique()
    cv = calc_adjusted_scale(df_base, df_cf, params, specs, n_agents)
    return cv


def calc_adjusted_scale(df_base, df_count, params, specs, n_agents):
    # First construct the discounted sum of utilities for the counterfactual scenario
    disc_sum_cf = create_disc_sum(df_count, specs)

    # Then we construct the relevant objects to be able to scale consumption,
    # such that it matches the discounted sum from above
    education = df_base["education"].values
    mu_vector = education * params["mu_high"] + (1 - education) * params["mu_low"]

    # Generate the part of the realized utility which is not to be sccaled, i.e. which remains
    # constant. This is the taste shock and the disutility paramters. First generate consumption utility.
    util_cons = df_base["hh_size"](
        (df_base["consumption"] / df_base["cons_scale"]) ** (1 - mu_vector)
    ) / (1 - mu_vector)
    # Then substract to get constant utility
    utility_base_stays_constant = df_base["real_util"].values - util_cons
    # If I now mutlitiply consumption with 1 + gamma, then bring the 1 + gamma out of the equation.
    # This is what will happen in the create_adjusted_difference function

    # Generate the discount factor for the base dataframe
    disc_factor_base = specs["discount_factor"] ** df_base["period"].values

    partial_adjustment = lambda scale_in: create_adjusted_difference(
        utility_to_scale=util_cons,
        not_scaled_utility=utility_base_stays_constant,
        disc_factor_base=disc_factor_base,
        disc_sum_cf=disc_sum_cf,
        n_agents=n_agents,
        mu_vector=mu_vector,
        scale=scale_in,
    )

    scale = opt.brentq(partial_adjustment, -0.5, 0.5)

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
    df.loc[:, "disc_util"] = df["real_util"] * (
        specs["discount_factor"] ** df["period"]
    )

    return df.groupby("agent")["disc_util"].sum().mean()


def add_cons_scale_and_adult_hh_size(df, specs):
    has_partner_int = (df["partner_state"].values > 0).astype(int)
    # education = df["education"].values
    # period = df["period"].values
    # sex = df["sex"].values
    # nb_children = specs["children_by_state"][sex, education, has_partner_int, period]
    hh_size = 1 + has_partner_int
    df.loc[:, "cons_scale"] = np.sqrt(hh_size)
    df.loc[:, "hh_size"] = hh_size
    return df


def create_realized_taste_shock(df, specs):
    df.loc[:, "real_taste_shock"] = np.nan
    for choice in range(specs["n_choices"]):
        df.loc[df["choice"] == choice, "real_taste_shock"] = df.loc[
            df["choice"] == choice, f"taste_shocks_{choice}"
        ]
    return df


def create_real_utility(df, specs):
    df = create_realized_taste_shock(df, specs)
    df.loc[:, "real_util"] = df["utility"] + df["real_taste_shock"]
    return df
