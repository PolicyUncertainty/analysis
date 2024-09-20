import jax.numpy as jnp
import numpy as np
import pandas as pd


def generate_start_states(observed_data, n_agents, seed, options):
    np.random.seed(seed)
    # Define start wealth
    start_period_data = observed_data[
        observed_data["period"] == observed_data["period"].min()
    ]

    # Generate education level
    edu_shares = start_period_data["education"].value_counts(normalize=True)
    n_agents_edu_types = np.round(edu_shares.sort_index() * n_agents).astype(int)
    edu_agents = np.repeat(edu_shares.index, n_agents_edu_types)

    wealth_agents = np.empty(n_agents, np.float64)
    for edu in edu_shares.index:
        n_agents_edu = n_agents_edu_types.loc[edu]
        start_period_data_edu = start_period_data[start_period_data["education"] == edu]
        # Draw between 25th and 75th percentile of wealth for each education level
        wealth_agents[edu_agents == edu] = np.random.uniform(
            start_period_data_edu["wealth"].quantile(0.25),
            start_period_data_edu["wealth"].quantile(0.75),
            n_agents_edu,
        )

    # This can be kicked out once we correct the wealth by lagged choice.
    wealth_agents = np.clip(
        wealth_agents, options["model_params"]["unemployment_benefits"] * 12, None
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
