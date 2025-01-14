import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dcegm.wealth_correction import adjust_observed_wealth
from scipy.stats import pareto


def generate_start_states(path_dict, params, model, n_agents, seed):
    specs = model["options"]["model_params"]

    observed_data = pd.read_pickle(
        path_dict["intermediate_data"] + "structural_estimation_sample.pkl"
    )

    np.random.seed(seed)
    # Define start data and adjust wealth
    min_period = observed_data["period"].min()
    start_period_data = observed_data[observed_data["period"].isin([min_period])].copy()

    states_dict = {
        name: start_period_data[name].values
        for name in model["model_structure"]["discrete_states_names"]
    }
    states_dict["wealth"] = start_period_data["wealth"].values / specs["wealth_unit"]
    states_dict["experience"] = start_period_data["experience"].values

    start_period_data.loc[:, "adjusted_wealth"] = adjust_observed_wealth(
        observed_states_dict=states_dict,
        params=params,
        model=model,
    )

    # Generate container
    sex_agents = np.array([], np.uint8)
    education_agents = np.array([], np.uint8)
    for sex_var in range(specs["n_sexes"]):
        if sex_var == 0:
            n_agents_sex = n_agents - n_agents // 2
        else:
            n_agents_sex = n_agents // 2

        sex_vars = np.ones(n_agents_sex, np.uint8) * sex_var
        sex_agents = np.append(sex_agents, sex_vars)

        # Restrict start data
        start_data_sex = start_period_data[start_period_data["sex"] == sex_var]

        # Generate education level
        edu_shares = start_data_sex["education"].value_counts(normalize=True)
        n_agents_edu_types = np.round(edu_shares.sort_index() * n_agents_sex).astype(
            int
        )

        # Generate education array
        edu_agents_per_sex = np.repeat(edu_shares.index, n_agents_edu_types)
        education_agents = np.append(education_agents, edu_agents_per_sex)

    # Generate containers
    wealth_agents = np.empty(n_agents, np.float64)
    informed_agents = np.zeros(n_agents, np.uint8)
    exp_agents = np.empty(n_agents, np.float64)
    lagged_choice = np.empty(n_agents, np.uint8)
    partner_states = np.empty(n_agents, np.uint8)
    health_agents = np.empty(n_agents, np.float64)

    for sex_var in range(specs["n_sexes"]):
        for edu in range(specs["n_education_types"]):
            type_mask = (sex_agents == sex_var) & (education_agents == edu)
            start_period_data_edu = start_period_data[
                (start_period_data["sex"] == sex_var)
                & (start_period_data["education"] == edu)
            ]

            n_agents_edu = np.sum(type_mask)

            # Restrict dataset on education level

            wealth_start_edu = draw_start_wealth_dist(
                start_period_data_edu, n_agents_edu
            )
            wealth_agents[type_mask] = wealth_start_edu

            # Generate edu specific informed shares
            informed_share_edu = specs["initial_informed_shares"][edu]
            # Draw informed states according to inital distribution
            dist = np.array([1 - informed_share_edu, informed_share_edu])
            informed_draws_edu = np.random.choice(2, n_agents_edu, p=dist)
            informed_agents[type_mask] = informed_draws_edu

            # Generate type specific initial experience distribution
            exp_max_edu = start_period_data_edu["experience"].max()
            empirical_exp_probs = start_period_data_edu["experience"].value_counts(
                normalize=True
            )
            exp_probs = pd.Series(
                index=np.arange(0, exp_max_edu + 1), data=0, dtype=float
            )
            exp_probs.update(empirical_exp_probs)
            exp_agents[type_mask] = np.random.choice(
                exp_max_edu + 1, size=n_agents_edu, p=exp_probs.values
            )

            # Generate type specific initial lagged choice distribution
            empirical_lagged_choice_probs = start_period_data_edu[
                "lagged_choice"
            ].value_counts(normalize=True)
            lagged_choice_probs = pd.Series(
                index=np.arange(0, specs["n_choices"]), data=0, dtype=float
            )
            lagged_choice_probs.update(empirical_lagged_choice_probs)
            lagged_choice_edu = np.random.choice(
                specs["n_choices"], size=n_agents_edu, p=lagged_choice_probs.values
            )
            lagged_choice[type_mask] = lagged_choice_edu

            # Get type specific partner states
            empirical_partner_probs = start_period_data_edu[
                "partner_state"
            ].value_counts(normalize=True)
            partner_probs = pd.Series(
                index=np.arange(specs["n_partner_states"]), data=0, dtype=float
            )
            partner_probs.update(empirical_partner_probs)
            partner_states_edu = np.random.choice(
                specs["n_partner_states"], size=n_agents_edu, p=partner_probs.values
            )
            partner_states[type_mask] = partner_states_edu

            # Generate health states
            empirical_health_probs = start_period_data_edu["health"].value_counts(
                normalize=True
            )
            health_probs = pd.Series(
                index=np.arange(specs["n_health_states"]), data=0, dtype=float
            )
            health_probs.update(empirical_health_probs)
            health_states_edu = np.random.choice(
                specs["n_health_states"], size=n_agents_edu, p=health_probs.values
            )
            health_agents[type_mask] = health_states_edu

    # Transform it to be between 0 and 1
    exp_agents /= specs["max_init_experience"]

    # Set lagged choice to 1(unemployment) if experience is 0
    exp_zero_mask = exp_agents == 0
    lagged_choice[exp_zero_mask] = 1

    states = {
        "period": jnp.zeros_like(exp_agents, dtype=jnp.uint8),
        "experience": jnp.array(exp_agents, dtype=jnp.float64),
        "sex": jnp.array(sex_agents, dtype=jnp.uint8),
        "health": jnp.array(health_agents, dtype=jnp.uint8),
        "education": jnp.array(education_agents, dtype=jnp.uint8),
        "informed": jnp.array(informed_agents, dtype=jnp.uint8),
        "lagged_choice": jnp.array(lagged_choice, dtype=jnp.uint8),
        "policy_state": jnp.zeros_like(exp_agents, dtype=jnp.uint8) + 8,
        "job_offer": jnp.ones_like(exp_agents, dtype=jnp.uint8),
        "partner_state": jnp.array(partner_states, dtype=jnp.uint8),
    }
    return states, wealth_agents


def draw_start_wealth_dist(start_period_data_edu, n_agents_edu):
    # # From now use uniform from 30 to 70th quantile
    wealth_start_edu = np.random.uniform(
        start_period_data_edu["adjusted_wealth"].quantile(0.3),
        start_period_data_edu["adjusted_wealth"].quantile(0.7),
        n_agents_edu,
    )
    return wealth_start_edu
    # if edu == 1:
    #     # Filter out high outliers for high
    #     wealth_edu = wealth_edu[wealth_edu < np.quantile(wealth_edu, 0.85)]
    #
    # median = np.quantile(wealth_edu, 0.5)
    # fscale = min_unemployment_benefits - 0.01
    #
    # # # Adjust shape to ensure the median is as desired
    # # adjusted_shape = np.log(2) / np.log(median / fscale)
    #
    # # Estimate pareto wealth distribution. Take single unemployment benefits as minimum.
    # shape_param, loc_param, scale_param = pareto.fit(wealth_edu, fscale=fscale)
    #
    # wealth_agents[edu_agents == edu] = pareto.rvs(shape_param, loc=loc_param, scale=fscale, size=n_agents_edu)
    # breakpoint()
