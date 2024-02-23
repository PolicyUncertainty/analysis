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
from simulation.simulate_scenario import simulate_scenario
from simulation.policy_state_scenarios.step_function import (
    update_specs_for_step_function,
    realized_policy_step_function,
)

# Create estimated model
from model_code.model_solver import solve_model
from model_code.policy_states_belief import expected_SRA_probs_estimation
from model_code.policy_states_belief import exp_ret_age_transition_matrix


est_params = pickle.load(open(path_dict["est_results"] + "est_params.pkl", "rb"))

model_solution_est = solve_model(
    path_dict=path_dict,
    params=est_params,
    update_spec_for_policy_state=exp_ret_age_transition_matrix,
    policy_state_trans_func=expected_SRA_probs_estimation,
    file_append="est",
    load_model=True,
    load_solution=True,
)

data_sim = simulate_scenario(
    path_dict=path_dict,
    params=est_params,
    n_agents=1000,
    seed=123,
    update_spec_for_policy_state=update_specs_for_step_function,
    policy_state_func_scenario=realized_policy_step_function,
    expected_model=model_solution_est,
)
data_sim.to_pickle(path_dict["intermediate_data"] + "simulated_data.pkl")
