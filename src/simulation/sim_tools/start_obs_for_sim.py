import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from dcegm.pre_processing.shared import create_array_with_smallest_int_dtype

from model_code.stochastic_processes.health_transition import (
    calc_disability_probability,
)
from model_code.stochastic_processes.job_offers import job_offer_process_transition
from model_code.transform_data_from_model import load_scale_and_correct_data


def generate_start_states_from_obs(
    path_dict, params, model_class, inital_SRA, only_informed=False
):
    model_specs = model_class.model_specs
    model_structure = model_class.model_structure

    observed_data = load_scale_and_correct_data(
        path_dict=path_dict, model_class=model_class
    )

    # Generate start policy state from initial SRA
    initial_policy_state = np.floor(
        (inital_SRA - model_specs["min_SRA"]) / model_specs["SRA_grid_size"]
    )

    # Define start data and adjust wealth
    min_period = observed_data["period"].min()
    start_period_data = observed_data[observed_data["period"].isin([min_period])].copy()

    # Read out states for wealth adjustment
    states_dict = {
        name: start_period_data[name].values
        for name in model_structure["discrete_states_names"]
    }
    states_dict["assets_begin_of_period"] = start_period_data[
        "assets_begin_of_period"
    ].values
    states_dict["experience"] = start_period_data["experience"].values
    periods = start_period_data["period"].values

    n_individuals = periods.shape[0]
    n_multiply_start_obs = model_specs["n_multiply_start_obs"]

    seed = model_specs["seed"]
    np.random.seed(seed)

    # Draw jax keys
    random_keys = jax.random.split(jax.random.PRNGKey(seed), (n_individuals, 3))

    def draw_informed_and_job_offers(
        sex, education, lagged_choice, job_offer_obs, health, keys
    ):
        job_offer_prob = job_offer_process_transition(
            params=params,
            sex=sex,
            health=health,
            model_specs=model_specs,
            education=education,
            policy_state=initial_policy_state,
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
        informed_share_edu = model_specs["informed_shares_in_ages"][
            model_specs["start_age"], education
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
            model_specs=model_specs,
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

    # Assign job offers, informed agents and health
    # If only informed should be simulated assign 1 everywhere
    if only_informed:
        states_dict["informed"] = np.ones_like(states_dict["period"])
    else:
        states_dict["informed"] = informed_agents.flatten()

    states_dict["job_offer"] = job_offers.flatten()
    states_dict["health"] = health_agents.flatten()

    policy_state_agents = (
        np.ones_like(states_dict["health"]) * initial_policy_state
    ).astype(np.uint8)
    states_dict["policy_state"] = policy_state_agents

    initial_states = jax.tree.map(create_array_with_smallest_int_dtype, states_dict)

    return initial_states


def investigate_start_obs(
    path_dict,
    model_class,
    load=False,
):
    if load:
        initial_obs_table = pd.read_csv(
            path_dict["data_tables"] + "initial_obs_table.csv", index_col=[0, 1]
        )
        return initial_obs_table

    observed_data = load_scale_and_correct_data(
        path_dict=path_dict, model_class=model_class
    )

    # Define start data and adjust wealth
    min_period = observed_data["period"].min()
    start_period_data = observed_data[observed_data["period"].isin([min_period])].copy()
    # Get table of median and mean for continous variables
    # Get table of median and mean wealth
    median = start_period_data.groupby(["sex", "education"])[
        ["experience_years", "experience", "assets_begin_of_period", "wealth"]
    ].median()
    mean = start_period_data.groupby(["sex", "education"])[
        ["experience_years", "experience", "assets_begin_of_period", "wealth"]
    ].mean()
    rename_median = {col: col + "_median" for col in median.columns}
    rename_mean = {col: col + "_mean" for col in mean.columns}
    median = median.rename(columns=rename_median)
    mean = mean.rename(columns=rename_mean)
    initial_obs_table = median.merge(mean, left_index=True, right_index=True)
    initial_obs_table.to_csv(path_dict["data_tables"] + "initial_obs_table.csv")
    return initial_obs_table
