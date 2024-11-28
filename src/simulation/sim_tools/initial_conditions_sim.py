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
    # Define start wealth
    min_period = observed_data["period"].min()
    start_period_data = observed_data[observed_data["period"].isin([min_period])].copy()

    # Generate education level
    edu_shares = start_period_data["education"].value_counts(normalize=True)
    n_agents_edu_types = np.round(edu_shares.sort_index() * n_agents).astype(int)
    edu_agents = np.repeat(edu_shares.index, n_agents_edu_types)

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

    wealth_agents = np.empty(n_agents, np.float64)
    informed_agents = np.zeros(n_agents, np.uint8)
    for edu in edu_shares.index:
        n_agents_edu = n_agents_edu_types.loc[edu]
        start_period_data_edu = start_period_data[start_period_data["education"] == edu]

        # # From now use uniform from 30 to 70th quantile
        wealth_agents[edu_agents == edu] = np.random.uniform(
            start_period_data_edu["adjusted_wealth"].quantile(0.3),
            start_period_data_edu["adjusted_wealth"].quantile(0.7),
            n_agents_edu,
        )

        # Generate edu specific informed shares
        informed_share_edu = specs["initial_informed_shares"][edu]
        # Draw informed states according to inital distribution
        dist = np.array([1 - informed_share_edu, informed_share_edu])
        informed_draws_edu = np.random.choice(2, n_agents_edu, p=dist)
        informed_agents[edu_agents == edu] = informed_draws_edu

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

    max_init_experience = specs["max_init_experience"]
    # Generate experience
    exp_agents = np.empty(n_agents, np.float64)
    for edu in edu_shares.index:
        n_agents_edu = n_agents_edu_types.loc[edu]
        start_period_data_edu = start_period_data[start_period_data["education"] == edu]
        exp_max_edu = start_period_data_edu["experience"].max()
        exp_dist = start_period_data_edu["experience"].value_counts(normalize=True)
        grid_probs = pd.Series(index=np.arange(0, exp_max_edu + 1), data=0, dtype=float)
        grid_probs.update(exp_dist)
        exp_agents[edu_agents == edu] = (
            np.random.choice(exp_max_edu + 1, size=n_agents_edu, p=grid_probs.values)
            / max_init_experience
        )

    # Generate lagged choice. Assign everywhere 1 except where experience is 0
    lagged_choice = np.ones_like(exp_agents)
    lagged_choice[exp_agents == 0] = 0

    # Draw random partner states
    partner_shares = start_period_data["partner_state"].value_counts(normalize=True)
    partner_probs = np.zeros(3, dtype=float)
    partner_probs[partner_shares.index] = partner_shares.values
    partner_states = np.random.choice(3, n_agents, p=partner_probs)

    states = {
        "period": jnp.zeros_like(exp_agents, dtype=jnp.uint8),
        "experience": jnp.array(exp_agents, dtype=jnp.float64),
        "education": jnp.array(edu_agents, dtype=jnp.uint8),
        "informed": jnp.array(informed_agents, dtype=jnp.uint8),
        "lagged_choice": jnp.array(lagged_choice, dtype=jnp.uint8),
        "policy_state": jnp.zeros_like(exp_agents, dtype=jnp.uint8) + 8,
        "job_offer": jnp.ones_like(exp_agents, dtype=jnp.uint8),
        "partner_state": jnp.array(partner_states, dtype=jnp.uint8),
    }
    return states, wealth_agents
