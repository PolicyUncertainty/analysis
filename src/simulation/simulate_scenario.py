import pickle

import jax.numpy as jnp
import pandas as pd
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods_for_model
from derive_specs import generate_specs_and_update_params
from model_code.initial_conditions_sim import generate_start_states
from model_code.specify_model import specify_model


def simulate_scenario(
    path_dict, params, specs, n_agents, seed, policy_state_func_scenario, expected_model
):
    # Generate dcegm model for project specs
    model, options = specify_model(
        project_specs=specs,
        model_data_path=path_dict["intermediate_data"],
        exog_trans_func=policy_state_func_scenario,
        load_model=True,
    )

    data_decision = pd.read_pickle(path_dict["intermediate_data"] + "decision_data.pkl")
    initial_states, wealth_agents = generate_start_states(
        data_decision, n_agents, seed, options
    )

    sim_dict = simulate_all_periods_for_model(
        states_initial=initial_states,
        resources_initial=wealth_agents,
        n_periods=options["model_params"]["n_periods"],
        params=params,
        seed=seed,
        endog_grid_solved=expected_model["endog_grid"],
        value_solved=expected_model["value"],
        policy_left_solved=expected_model["policy_left"],
        policy_right_solved=expected_model["policy_right"],
        choice_range=jnp.arange(options["model_params"]["n_choices"], dtype=jnp.int16),
        model=model,
    )
    df = create_simulation_df(sim_dict)
    return df
