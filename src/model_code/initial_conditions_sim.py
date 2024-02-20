import jax.numpy as jnp
import numpy as np


def generate_start_states(observed_data, n_agents, seed):
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
    wealth_agents[wealth_agents < 0] = 0

    # Initial experience
    exp_counts = (
        (start_period_data["experience"].value_counts(normalize=True) * n_agents)
        .round()
        .sort_index()
    )
    # Generate experience state vector
    exp_agents = np.array([])
    for i, exp in enumerate(exp_counts.index.values):
        exp_agents = np.append(exp_agents, np.full(int(exp_counts[i]), exp))

    missing = n_agents - exp_agents.size
    exp_agents = np.append(exp_agents, np.full(missing, 0)).astype(int)
    # Shuffle exp_agents
    np.random.shuffle(exp_agents)

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
