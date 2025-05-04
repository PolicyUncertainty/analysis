import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from dcegm.pre_processing.shared import create_array_with_smallest_int_dtype
from dcegm.wealth_correction import adjust_observed_wealth

from model_code.stochastic_processes.health_transition import (
    calc_disability_probability,
)
from model_code.stochastic_processes.job_offers import job_offer_process_transition


def generate_start_states_from_obs(
    path_dict, params, model, inital_SRA, seed, only_informed=False
):
    specs = model["options"]["model_params"]

    observed_data = pd.read_csv(path_dict["struct_est_sample"])

    # Define start data and adjust wealth
    min_period = observed_data["period"].min()
    start_period_data = observed_data[observed_data["period"].isin([min_period])].copy()

    # Read out states for wealth adjustment
    states_dict = {
        name: start_period_data[name].values
        for name in model["model_structure"]["discrete_states_names"]
    }
    # Transform experience for wealth adjustment
    periods = start_period_data["period"].values
    max_init_exp = model["options"]["model_params"]["max_exp_diffs_per_period"][periods]
    exp_denominator = periods + max_init_exp
    states_dict["experience"] = start_period_data["experience"].values / exp_denominator
    states_dict["wealth"] = start_period_data["wealth"].values / specs["wealth_unit"]

    states_dict["wealth"] = jnp.asarray(
        adjust_observed_wealth(
            observed_states_dict=states_dict,
            params=params,
            model=model,
        )
    )
    n_individuals = periods.shape[0]
    n_multiply_start_obs = specs["n_multiply_start_obs"]
    np.random.seed(seed)

    # Draw jax keys
    random_keys = jax.random.split(jax.random.PRNGKey(seed), (n_individuals, 3))

    def draw_informed_and_job_offers(
        sex, education, lagged_choice, job_offer_obs, health, keys
    ):
        job_offer_prob = job_offer_process_transition(
            params=params,
            sex=sex,
            options=specs,
            education=education,
            period=jnp.array(-1, dtype=int),
            choice=lagged_choice,
        ).T
        # Draw n_multiply_start_obs job offers
        job_offers_draw = jax.random.choice(
            key=keys[0], a=2, shape=(n_multiply_start_obs,), p=job_offer_prob
        )

        bool_observed_job_offfer = job_offer_obs > -1
        # Assign 1 if working is true
        observed_job_offer = jnp.ones_like(job_offers_draw) * job_offer_obs
        job_offers = (
            bool_observed_job_offfer * observed_job_offer
            + (1 - bool_observed_job_offfer) * job_offers_draw
        )

        # Draw informed state
        informed_share_edu = specs["informed_shares_in_ages"][
            specs["start_age"], education
        ]
        # Draw informed states according to inital distribution
        dist = jnp.array([1 - informed_share_edu, informed_share_edu])
        informed_draws_edu = jax.random.choice(
            key=keys[1], a=2, shape=(n_multiply_start_obs,), p=dist
        )

        # Health
        disability_prob = calc_disability_probability(
            params=params,
            sex=sex,
            period=jnp.array(-1, dtype=int),
            education=education,
            options=specs,
        )
        # 2 Disability, 1 is bad health
        prob_vector = jnp.array([1 - disability_prob, disability_prob])
        health_draw = jax.random.choice(
            key=keys[2], a=2, shape=(n_multiply_start_obs,), p=prob_vector
        )
        health_draw += 1
        # Good health is 0. So if observed this is 0, otherwise its the draw
        bool_observed_health = health > -1
        health = (1 - bool_observed_health) * health_draw

        return job_offers, informed_draws_edu, health

    # Create job offer and informed draws
    job_offers, informed_agents, health_agents = jax.vmap(
        draw_informed_and_job_offers, in_axes=(0, 0, 0, 0, 0, 0)
    )(
        states_dict["sex"],
        states_dict["education"],
        states_dict["lagged_choice"],
        states_dict["job_offer"],
        states_dict["health"],
        random_keys,
    )

    # Multiply all states in state dict
    states_dict = {
        name: np.repeat(states_dict[name], n_multiply_start_obs)
        for name in states_dict.keys()
    }
    # Read out wealth
    initial_wealth = states_dict["wealth"]
    states_dict.pop("wealth")

    # Assign job offers, informed agents and health
    # If only informed should be simoulated assign 1 everywhere
    if only_informed:
        states_dict["informed"] = jnp.ones_like(states_dict["period"])
    else:
        states_dict["informed"] = informed_agents.flatten()

    states_dict["job_offer"] = job_offers.flatten()
    states_dict["health"] = health_agents.flatten()

    # Generate start policy state from initial SRA
    initial_policy_state = np.floor(
        (inital_SRA - specs["min_SRA"]) / specs["SRA_grid_size"]
    )

    policy_state_agents = (
        jnp.ones_like(states_dict["health"]) * initial_policy_state
    ).astype(jnp.uint8)
    states_dict["policy_state"] = policy_state_agents

    initial_states = jax.tree.map(create_array_with_smallest_int_dtype, states_dict)

    # df = pd.DataFrame(initial_states)
    # df["wealth"] = initial_wealth
    # breakpoint()

    return initial_states, initial_wealth
