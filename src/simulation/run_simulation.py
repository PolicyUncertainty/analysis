# %%
# Set paths of project
from set_paths import create_path_dict

path_dict = create_path_dict()
# Import jax and set jax to work with 64bit
import jax

jax.config.update("jax_enable_x64", True)

import pickle

from simulation.policy_state_scenarios.step_function import (
    create_update_function_for_scale,
    create_update_function_for_slope,
    realized_policy_step_function,
)
from model_code.stochastic_processes.policy_states_belief import (
    expected_SRA_probs_estimation,
)
from model_code.stochastic_processes.policy_states_belief import (
    update_specs_exp_ret_age_trans_mat,
)
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario

# %%
# Set specifications
n_agents = 10000
seeed = 123
params = pickle.load(open(path_dict["est_results"] + "est_params.pkl", "rb"))

# %%
###################################################################
# Uncertainty counterfactual
###################################################################
update_specs_for_step_function_scale_1 = create_update_function_for_scale(path_dict, 1)
# # Create estimated model
data_sim = solve_and_simulate_scenario(
    path_dict=path_dict,
    params=params,
    solve_update_specs_func=update_specs_exp_ret_age_trans_mat,
    solve_policy_trans_func=expected_SRA_probs_estimation,
    simulate_update_specs_func=update_specs_for_step_function_scale_1,
    simulate_policy_trans_func=realized_policy_step_function,
    solution_exists=True,
    file_append_sol="subj",
    model_exists=True,
)
data_sim.to_pickle(path_dict["intermediate_data"] + "sim_data/data_subj_scale_1.pkl")
del data_sim
data_sim = solve_and_simulate_scenario(
    path_dict=path_dict,
    params=params,
    solve_update_specs_func=update_specs_for_step_function_scale_1,
    solve_policy_trans_func=realized_policy_step_function,
    simulate_update_specs_func=update_specs_for_step_function_scale_1,
    simulate_policy_trans_func=realized_policy_step_function,
    solution_exists=False,
    file_append_sol="scale_1",
    model_exists=True,
)
data_sim.to_pickle(path_dict["intermediate_data"] + "sim_data/data_real_scale_1.pkl")
del data_sim

# #########################################
# # Alternative policy trajectories
# #########################################
# Baseline of no increase
data_sim = solve_and_simulate_scenario(
    path_dict=path_dict,
    params=params,
    solve_update_specs_func=update_specs_exp_ret_age_trans_mat,
    solve_policy_trans_func=expected_SRA_probs_estimation,
    simulate_update_specs_func=create_update_function_for_slope(0),
    simulate_policy_trans_func=realized_policy_step_function,
    solution_exists=True,
    file_append_sol="subj",
    model_exists=True,
)
data_sim.to_pickle(path_dict["intermediate_data"] + "sim_data/data_no_increase.pkl")
del data_sim

# Increase of 0.05
data_sim = solve_and_simulate_scenario(
    path_dict=path_dict,
    params=params,
    solve_update_specs_func=update_specs_exp_ret_age_trans_mat,
    solve_policy_trans_func=expected_SRA_probs_estimation,
    simulate_update_specs_func=create_update_function_for_slope(0.05),
    simulate_policy_trans_func=realized_policy_step_function,
    solution_exists=True,
    file_append_sol="subj",
    model_exists=True,
)
data_sim.to_pickle(path_dict["intermediate_data"] + "sim_data/data_incr_005.pkl")
del data_sim

# Increase of 0.05
data_sim = solve_and_simulate_scenario(
    path_dict=path_dict,
    params=params,
    solve_update_specs_func=update_specs_exp_ret_age_trans_mat,
    solve_policy_trans_func=expected_SRA_probs_estimation,
    simulate_update_specs_func=create_update_function_for_slope(0.1),
    simulate_policy_trans_func=realized_policy_step_function,
    solution_exists=True,
    file_append_sol="subj",
    model_exists=True,
)
data_sim.to_pickle(path_dict["intermediate_data"] + "sim_data/data_incr_01.pkl")
del data_sim

# New counterfactual:
# Baseline SRA stays at 67
# Comparision with vorschlag rat: Alle 10 ein halbes jahr
# Comparision mit: Alle 10 ein Jahr

# %%
#
# data_sim = solve_and_simulate_scenario(
#     path_dict=path_dict,
#     params=params,
#     solve_update_specs_func=create_update_function_for_slope(0),
#     solve_policy_trans_func=realized_policy_step_function,
#     simulate_update_specs_func=create_update_function_for_slope(0),
#     simulate_policy_trans_func=realized_policy_step_function,
#     solution_exists=False,
#     file_append_sol="00",
#     model_exists=True,
# )
# data_sim.to_pickle(path_dict["intermediate_data"] + "sim_data/data_no_inc_no_unc.pkl")
# del data_sim
#
# data_sim = solve_and_simulate_scenario(
#     path_dict=path_dict,
#     params=params,
#     solve_update_specs_func=create_update_function_for_slope(0.05),
#     solve_policy_trans_func=realized_policy_step_function,
#     simulate_update_specs_func=create_update_function_for_slope(0.05),
#     simulate_policy_trans_func=realized_policy_step_function,
#     solution_exists=False,
#     file_append_sol="005",
#     model_exists=True,
# )
# data_sim.to_pickle(path_dict["intermediate_data"] + "sim_data/data_005_no_unc.pkl")
# del data_sim
