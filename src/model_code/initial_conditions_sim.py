import jax.numpy as jnp
import numpy as np
import pandas as pd


def generate_start_states(observed_data, n_agents, seed, options):
    np.random.seed(seed)
    # Define start wealth
    start_period_data = observed_data[
        observed_data["period"] == observed_data["period"].min()
    ]
    # Get wealth quantiles
    median_start_wealth = start_period_data["wealth"].median()
    std_wealth = start_period_data["wealth"].std()
    # Draw num agents wealth from a normal with mean_start wealth and std
    wealth_agents = np.random.normal(
        loc=median_start_wealth, scale=std_wealth, size=n_agents
    )
    wealth_agents[wealth_agents < 0] = (
        options["model_params"]["unemployment_benefits"] * 12
    )

    exp_max = start_period_data["experience"].max()
    grid_probs = pd.Series(index=np.arange(0, exp_max + 1), data=0, dtype=float)
    # Initial experience
    exp_dist = start_period_data["experience"].value_counts(normalize=True)
    grid_probs.update(exp_dist)
    exp_agents = np.random.choice(exp_max + 1, size=n_agents, p=grid_probs.values)

    # Generate lagged choice. Assign everywhere 1 except where experience is 0
    lagged_choice = np.ones_like(exp_agents)
    lagged_choice[exp_agents == 0] = 0

    states = {
        "period": jnp.zeros_like(exp_agents, dtype=jnp.int16),
        "experience": jnp.array(exp_agents, dtype=jnp.int16),
        "lagged_choice": jnp.array(lagged_choice, dtype=jnp.int16),
        "policy_state": jnp.zeros_like(exp_agents, dtype=jnp.int16) + 8,
        "retirement_age_id": jnp.zeros_like(exp_agents, dtype=jnp.int16),
    }
    return states, wealth_agents
