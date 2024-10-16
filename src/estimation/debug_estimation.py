# %%
# Set paths of project
from set_paths import create_path_dict

paths_dict = create_path_dict()

import jax
import yaml
import pickle as pkl
import numpy as np

jax.config.update("jax_enable_x64", True)

from set_paths import create_path_dict

path_dict = create_path_dict()
from estimation.estimate_setup import create_job_offer_params_from_start

# %%
params = pkl.load(open(path_dict["est_params"], "rb"))


from model_code.stochastic_processes.policy_states_belief import (
    expected_SRA_probs_estimation,
)
from model_code.stochastic_processes.policy_states_belief import (
    update_specs_exp_ret_age_trans_mat,
)

from model_code.specify_model import specify_model

# # Generate model_specs
# model, params = specify_model(
#     path_dict=paths_dict,
#     update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
#     policy_state_trans_func=expected_SRA_probs_estimation,
#     params=params,
#     load_model=True,
# )

from model_code.specify_model import specify_and_solve_model

solution, model, params = specify_and_solve_model(
    path_dict=paths_dict,
    file_append="subj",
    params=params,
    update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
    policy_state_trans_func=expected_SRA_probs_estimation,
    load_model=True,
    load_solution=True,
)


from estimation.estimate_setup import create_ll_from_paths, load_and_prep_data

data_decision, states_dict = load_and_prep_data(
    path_dict, params, model, drop_retirees=True
)

individual_likelihood, weights = create_ll_from_paths(
    params, paths_dict, load_model=False
)
ll_contribution = individual_likelihood(params)
# breakpoint()
data_decision["ll_contribution"] = -ll_contribution
# df_full = data_decision[data_decision["full_observed_state"]]
# # df_full.reset_index(inplace=True, drop=True)
# df_full_working = df_full[df_full["choice"] == 1]
# from model_code.utility_functions import utility_func
# breakpoint()

# from dcegm.interface import get_state_choice_index_per_state
# from dcegm.interpolation.interp1d import interp_value_on_wealth
#
# def plot_value(value_solved, endog_grid_solved, var_name, var_grid, state_dict, model, choices):
#
#     fig, ax = plt.subplots()
#     for choice in choices:
#         state_choice_dict = {**state_dict, "choice": choice}
#         value_all = np.zeros_like(var_grid)
#         for var_id, var in enumerate(var_grid):
#             if "wealth" in state_dict:
#                 wealth = state_dict["wealth"]
#                 state_dict[var_name] = var
#             elif var_name == "wealth":
#                 wealth = var
#             else:
#                 raise ValueError("Wealth not in state_dict or var_name")
#
#             state_choice_indexes = get_state_choice_index_per_state(
#                 states=state_dict,
#                 map_state_choice_to_index=model["model_structure"]["map_state_choice_to_index"],
#                 discrete_states_names=model["model_structure"]["discrete_states_names"],
#             )
#             value_state = jnp.take(
#                 value_solved, state_choice_indexes, axis=0, mode="fill", fill_value=jnp.nan
#             )
#             endog_grid_state = jnp.take(endog_grid_solved, state_choice_indexes, axis=0)
#
#             value_all[var_id] = interp_value_on_wealth(
#                 wealth=wealth,
#                 endog_grid=endog_grid_state[choice, :],
#                 value=value_state[choice, :],
#                 compute_utility=model["model_funcs"]["compute_utility"],
#                 state_choice_vec=state_choice_dict,
#                 params=params,
#             )
#
#         ax.plot(var_grid, value_all, label=f"Choice {choice}")
#     ax.legend()
#     plt.show()
#
# exp_grid = np.arange(0, 45, 1)
# discrete_state_to_plot = {"period": 30, "lagged_choice": 1, "policy_state": 1, "job_offer": 1, "wealth": 50, "education": 0}
# plot_value(solution["value"], solution["endog_grid"], "experience", exp_grid, discrete_state_to_plot, model, [0, 1])
# breakpoint()
#
#
