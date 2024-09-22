import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dcegm.wealth_correction import adjust_observed_wealth
from scipy.stats import pareto


def generate_start_states(path_dict, params, model, n_agents, seed):
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
        for name in model["model_structure"]["state_space_names"]
    }
    start_period_data.loc[:, "adjusted_wealth"] = adjust_observed_wealth(
        observed_states_dict=states_dict,
        wealth=start_period_data["wealth"].values,
        params=params,
        model=model,
    )

    model_params = model["options"]["model_params"]
    min_unemployment_benefits = (
        model_params["unemployment_benefits"]
        + model_params["unemployment_benefits_housing"]
    ) * 12
    wealth_agents = np.empty(n_agents, np.float64)
    for edu in edu_shares.index:
        n_agents_edu = n_agents_edu_types.loc[edu]
        start_period_data_edu = start_period_data[start_period_data["education"] == edu]
        wealth_edu = start_period_data_edu["adjusted_wealth"].values

        # # From now use uniform from 30 to 70th quantile
        # wealth_agents[edu_agents == edu] = np.random.uniform(
        #             start_period_data_edu["adjusted_wealth"].quantile(0.3),
        #             start_period_data_edu["adjusted_wealth"].quantile(0.7),
        #             n_agents_edu,
        # )
        if edu == 1:
            # Filter out high outliers for high
            wealth_edu = wealth_edu[wealth_edu < np.quantile(wealth_edu, 0.85)]

        median = np.quantile(wealth_edu, 0.5)
        fscale = min_unemployment_benefits - 0.01

        # Estimate pareto wealth distribution. Take single unemployment benefits as minimum.
        shape_param, loc_param, scale_param = pareto.fit(wealth_edu, fscale=fscale)

        # Adjust shape to ensure the median is as desired
        adjusted_shape = np.log(2) / np.log(median / fscale)
        wealth_agents[edu_agents == edu] = pareto.rvs(
            adjusted_shape, loc=loc_param, scale=fscale, size=n_agents_edu
        )

    # Generate experience
    exp_agents = np.empty(n_agents, np.int16)
    for edu in edu_shares.index:
        n_agents_edu = n_agents_edu_types.loc[edu]
        start_period_data_edu = start_period_data[start_period_data["education"] == edu]
        exp_max_edu = start_period_data_edu["experience"].max()
        exp_dist = start_period_data_edu["experience"].value_counts(normalize=True)
        grid_probs = pd.Series(index=np.arange(0, exp_max_edu + 1), data=0, dtype=float)
        grid_probs.update(exp_dist)
        exp_agents[edu_agents == edu] = np.random.choice(
            exp_max_edu + 1, size=n_agents_edu, p=grid_probs.values
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
        "period": jnp.zeros_like(exp_agents, dtype=jnp.int64),
        "experience": jnp.array(exp_agents, dtype=jnp.int64),
        "education": jnp.array(edu_agents, dtype=jnp.int64),
        "lagged_choice": jnp.array(lagged_choice, dtype=jnp.int64),
        "policy_state": jnp.zeros_like(exp_agents, dtype=jnp.int64) + 8,
        "retirement_age_id": jnp.zeros_like(exp_agents, dtype=jnp.int64),
        "job_offer": jnp.ones_like(exp_agents, dtype=jnp.int64),
        "partner_state": jnp.array(partner_states, dtype=jnp.int64),
    }
    return states, wealth_agents
