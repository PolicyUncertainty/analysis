# %%
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
import pickle

jax.config.update("jax_enable_x64", True)
from simulation.policy_state_scenarios.step_function import (
    update_specs_for_step_function_scale_1,
    update_specs_for_step_function_scale_05,
    update_specs_for_step_function_scale_2,
    realized_policy_step_function,
)
from model_code.model_solver import specify_and_solve_model
from model_code.policy_states_belief import expected_SRA_probs_estimation
from model_code.policy_states_belief import update_specs_exp_ret_age_trans_mat
from simulation.simulate_scenario import solve_and_simulate_scenario

# %%
# Set specifications
n_agents = 10000
seeed = 123
params = pickle.load(open(path_dict["est_results"] + "em_result.pkl", "rb"))
import inspect

for i in inspect.getmembers(params):
    # to remove private and protected
    # functions
    if not i[0].startswith("_"):
        # To remove other methods that
        # doesnot start with a underscore
        if not inspect.ismethod(i[1]):
            print(i)
# %%
###################################################################
# Uncertainty counterfactual
###################################################################
# Create estimated model
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
    solution_exists=True,
    file_append_sol="scale_1",
    model_exists=True,
)
data_sim.to_pickle(path_dict["intermediate_data"] + "sim_data/data_real_scale_1.pkl")
del data_sim


###################################################################
# Counterfactual double alpha
###################################################################

data_sim = solve_and_simulate_scenario(
    path_dict=path_dict,
    params=params,
    solve_update_specs_func=update_specs_exp_ret_age_trans_mat,
    solve_policy_trans_func=expected_SRA_probs_estimation,
    simulate_update_specs_func=update_specs_for_step_function_scale_2,
    simulate_policy_trans_func=realized_policy_step_function,
    solution_exists=True,
    file_append_sol="subj",
    model_exists=True,
)
data_sim.to_pickle(path_dict["intermediate_data"] + "sim_data/data_subj_scale_2.pkl")
del data_sim

data_sim = solve_and_simulate_scenario(
    path_dict=path_dict,
    params=params,
    solve_update_specs_func=update_specs_for_step_function_scale_2,
    solve_policy_trans_func=realized_policy_step_function,
    simulate_update_specs_func=update_specs_for_step_function_scale_2,
    simulate_policy_trans_func=realized_policy_step_function,
    solution_exists=True,
    file_append_sol="scale_2",
    model_exists=True,
)
data_sim.to_pickle(path_dict["intermediate_data"] + "sim_data/data_real_scale_2.pkl")
del data_sim


###################################################################
# Counterfactual half alpha
###################################################################

data_sim = solve_and_simulate_scenario(
    path_dict=path_dict,
    params=params,
    solve_update_specs_func=update_specs_exp_ret_age_trans_mat,
    solve_policy_trans_func=expected_SRA_probs_estimation,
    simulate_update_specs_func=update_specs_for_step_function_scale_05,
    simulate_policy_trans_func=realized_policy_step_function,
    solution_exists=True,
    file_append_sol="subj",
    model_exists=True,
)
data_sim.to_pickle(path_dict["intermediate_data"] + "sim_data/data_subj_scale_05.pkl")
del data_sim

data_sim = solve_and_simulate_scenario(
    path_dict=path_dict,
    params=params,
    solve_update_specs_func=update_specs_for_step_function_scale_05,
    solve_policy_trans_func=realized_policy_step_function,
    simulate_update_specs_func=update_specs_for_step_function_scale_05,
    simulate_policy_trans_func=realized_policy_step_function,
    solution_exists=False,
    file_append_sol="scale_05",
    model_exists=True,
)
data_sim.to_pickle(path_dict["intermediate_data"] + "sim_data/data_real_scale_05.pkl")
del data_sim
# %%
