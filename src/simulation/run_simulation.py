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
from simulation.simulate_scenario import simulate_scenario
from simulation.policy_state_scenarios.step_function import (
    update_specs_for_step_function,
    realized_policy_step_function,
)

# Create estimated model
from model_code.model_solver import specify_and_solve_model
from model_code.policy_states_belief import expected_SRA_probs_estimation
from model_code.policy_states_belief import update_specs_exp_ret_age_trans_mat

params = pickle.load(open(path_dict["est_results"] + "est_params.pkl", "rb"))
# params = {
#     # Utility parameters
#     "mu": 0.8,
#     "dis_util_work": 1.5411056147726503,
#     "dis_util_unemployed": 1.972168129756152,
#     "bequest_scale": 1e-12,
#     # Taste shock scale
#     "lambda": 1.0,
#     # Interest rate and discount factor
#     "interest_rate": 0.03,
#     "beta": 0.95,
# }
# %%

model_solution_est, model, options, params = specify_and_solve_model(
    path_dict=path_dict,
    params=params,
    update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
    policy_state_trans_func=expected_SRA_probs_estimation,
    file_append="est_2024_02_26",
    load_model=True,
    load_solution=True,
)

data_sim_1 = simulate_scenario(
    path_dict=path_dict,
    params=params,
    n_agents=1000,
    seed=123,
    update_spec_for_policy_state=update_specs_for_step_function,
    policy_state_func_scenario=realized_policy_step_function,
    expected_model=model_solution_est,
)
data_sim_1.to_pickle(path_dict["intermediate_data"] + "sim_data_1_unc.pkl")
import matplotlib.pyplot as plt

#
# model_solution_step_func, _, _, _ = specify_and_solve_model(
#     path_dict=path_dict,
#     params=est_params,
#     update_spec_for_policy_state=update_specs_for_step_function,
#     policy_state_trans_func=realized_policy_step_function,
#     file_append="step",
#     load_model=True,
#     load_solution=True,
# )
#
# data_sim_2 = simulate_scenario(
#     path_dict=path_dict,
#     params=est_params,
#     n_agents=1000,
#     seed=123,
#     update_spec_for_policy_state=update_specs_for_step_function,
#     policy_state_func_scenario=realized_policy_step_function,
#     expected_model=model_solution_step_func,
# )
# data_sim_2.to_pickle(path_dict["intermediate_data"] + "sim_data_1_no_unc.pkl")

# %%
# Plot choice shares by age
data_sim_1.groupby(["age"]).choice.value_counts(normalize=True).unstack().plot(
    title="Choice shares by age"
)

# %%fig_1 = (

# plot average income by age and choice
data_sim_1.groupby(["age", "choice"])["labor_income"].mean().unstack().plot(
    title="Average income by age and choice"
)
# %%
# plot average consumption by age and choice
data_sim_1.groupby(["age", "choice"])["consumption"].mean().unstack().plot(
    title="Average consumption by age and choice"
)
# %%
# plot average periodic savings by age and choice
data_sim_1.groupby(["age", "choice"])["savings_dec"].mean().unstack().plot(
    title="Average periodic savings by age and choice"
)

# %%
# plot average utility by age and choice
data_sim_1.groupby(["age", "choice"])["utility"].mean().unstack().plot(
    title="Average utility by age and choice"
)
# %%
# plot average wealth by age and choice
data_sim_1.groupby(["age", "choice"])["wealth_at_beginning"].mean().unstack().plot(
    title="Average wealth by age and choice"
)
data_sim_1.groupby(["age"])["wealth_at_beginning"].mean().plot(
    title="Average wealth by age"
)

plt.show()
# %%
