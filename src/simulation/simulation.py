# Set paths of project
import sys
from pathlib import Path

analysis_path = str(Path(__file__).resolve().parents[2]) + "/"
sys.path.insert(0, analysis_path + "submodules/dcegm/src/")
sys.path.insert(0, analysis_path + "src/")

from set_paths import create_path_dict

path_dict = create_path_dict(analysis_path)
# Import jax and set jax to work with 64bit
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


import pandas as pd
import numpy as np
import pickle

est_params = pickle.load(open(path_dict["est_results"] + "est_params.pkl", "rb"))
from derive_specs import generate_specs_and_update_params

specs, start_params_all = generate_specs_and_update_params(path_dict, est_params)


# from estimation.tools import specify_model_and_options
# from model_code.initial_conditions_sim import generate_start_states
#
# model, options, start_params_all = specify_model_and_options(
#     project_paths=project_paths,
#     start_params_all=start_params_all,
#     load_model=True,
#     step="simulation",
# )
# from estimation.tools import create_likelihood
#
# # data_decision = pd.read_pickle(analysis_path + "output/decision_data.pkl")
# #
# # solved_model = process_data_and_model(
# #     data_decision=data_decision,
# #     project_paths=project_paths,
# #     start_params_all=start_params_all,
# #     load_model=True,
# #     output="solved_model",
# # )
# # pickle.dump(solved_model, open(file_dir_path + "results_and_data/solved_model.pkl", "wb"))
#
# # Load solved model
# solved_model = pickle.load(
#     open(file_dir_path + "results_and_data/solved_model.pkl", "rb")
# )
# choice_probs_observations, value, policy_left, policy_right, endog_grid = solved_model
# n_agents = 1000
# seed = 123
#
# data_decision = pd.read_pickle(analysis_path + "output/decision_data.pkl")
#
# initial_states, wealth_agents = generate_start_states(
#     data_decision, n_agents, seed, options
# )
#
# from dcegm.simulation.simulate import simulate_all_periods_for_model
#
# sim_dict = simulate_all_periods_for_model(
#     states_initial=initial_states,
#     resources_initial=wealth_agents,
#     n_periods=options["model_params"]["n_periods"],
#     params=start_params_all,
#     seed=seed,
#     endog_grid_solved=endog_grid,
#     value_solved=value,
#     policy_left_solved=policy_left,
#     policy_right_solved=policy_right,
#     choice_range=jnp.arange(options["model_params"]["n_choices"], dtype=jnp.int16),
#     model=model,
# )
# # Transform to df and save
# from dcegm.simulation.sim_utils import create_simulation_df
#
# df = create_simulation_df(sim_dict)
# df["resources_at_beginning"] = df["savings"] + df["consumption"]
# # Create income var by shifting period of 1 of individuals and then substract
# # savings from resoures at beginning of period
# df["income"] = df.groupby("agent")["resources_at_beginning"].shift(-1) - df[
#     "savings"
# ] * (1 + start_params_all["interest_rate"])
# df.to_pickle(file_dir_path + "results_and_data/simulated_data.pkl")
