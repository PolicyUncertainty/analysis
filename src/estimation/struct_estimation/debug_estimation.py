# %%
# Set paths of project
from set_paths import create_path_dict

paths_dict = create_path_dict()

import jax
import pickle as pkl
import numpy as np

jax.config.update("jax_enable_x64", True)

from set_paths import create_path_dict

path_dict = create_path_dict()
from specs.derive_specs import generate_derived_and_data_derived_specs

specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)

model_name = "both"

# %%
params = pkl.load(open(path_dict["est_results"] + f"est_params_{model_name}.pkl", "rb"))
# params = load_and_set_start_params(path_dict)


from model_code.policy_processes.policy_states_belief import (
    expected_SRA_probs_estimation,
)
from model_code.policy_processes.policy_states_belief import (
    update_specs_exp_ret_age_trans_mat,
)

# from model_code.specify_model import specify_model
#
# load_model = input("Load model? (y/n): ") == "y"
#
# # Generate model_specs
# model, params = specify_model(
#     path_dict=paths_dict,
#     update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
#     policy_state_trans_func=expected_SRA_probs_estimation,
#     params=params,
#     load_model=load_model,
# )
from model_code.specify_model import specify_and_solve_model

solution, model, params = specify_and_solve_model(
    path_dict=paths_dict,
    file_append=model_name,
    params=params,
    expected_alpha=False,
    load_model=True,
    load_solution=True,
)

from estimation.struct_estimation.estimate_setup import load_and_prep_data

data_decision, states_dict = load_and_prep_data(
    path_dict, params, model, drop_retirees=True
)

data_decision = data_decision[data_decision["age"] <= specs["max_ret_age"]]

# from estimation.struct_estimation.estimate_setup import (
#     load_and_prep_data,
#     est_class_from_paths,
# )
#
# est_class = est_class_from_paths(
#     path_dict=path_dict,
#     start_params_all=params,
#     slope_disutil_method=False,
#     file_append="subj",
#     use_weights=False,
#     load_model=load_model,
#     save_results=False,
# )
#
# ll_value_individual, model_solution = est_class.ll_func(params)
#
# data_decision["ll_contribution"] = -ll_value_individual
#
# from model_code.unobserved_state_weighting import create_unobserved_state_specs
#
# unobserved_state_specs = create_unobserved_state_specs(data_decision, model)
#
# from export_results.figures.observed_model_fit import choice_probs_for_choice_vals
#
# for choice in range(specs["n_choices"]):
#     choice_vals = np.ones_like(data_decision["choice"].values) * choice
#
#     choice_probs_observations = choice_probs_for_choice_vals(
#         choice_vals=choice_vals,
#         states_dict=states_dict,
#         model=model,
#         unobserved_state_specs=unobserved_state_specs,
#         params=params,
#         est_model=model_solution,
#     )
#
#     choice_probs_observations = np.nan_to_num(choice_probs_observations, nan=0.0)
#     data_decision[f"choice_{choice}"] = choice_probs_observations


# df_full = data_decision[data_decision["full_observed_state"]]
# # df_full.reset_index(inplace=True, drop=True)
# df_full_working = df_full[df_full["choice"] == 1]
# from model_code.utility_functions import utility_func
# breakpoint()

from dcegm.interface import (
    policy_and_value_for_state_choice_vec,
)
import matplotlib.pyplot as plt


def plot_value(
    value_solved,
    endog_grid_solved,
    policy_solved,
    var_name,
    var_grid,
    state_dict,
    model,
    choices,
):
    fig, ax = plt.subplots(ncols=2)
    for choice in choices:
        state_choice_dict = {**state_dict, "choice": choice}
        value_all = np.zeros_like(var_grid, dtype=float)
        consumption = np.zeros_like(var_grid, dtype=float)
        for var_id, var in enumerate(var_grid):
            if var_name == "wealth":
                wealth = var
                second_cont = state_dict["experience"]
            elif var_name == "experience":
                second_cont = var
                wealth = state_dict["wealth"]
            else:
                raise ValueError("Wealth not in state_dict or var_name")

            policy, value = policy_and_value_for_state_choice_vec(
                endog_grid_solved=endog_grid_solved,
                value_solved=value_solved,
                policy_solved=policy_solved,
                params=params,
                model=model,
                state_choice_vec=state_choice_dict,
                wealth=wealth,
                compute_utility=model["model_funcs"]["compute_utility"],
                second_continous=second_cont,
            )
            value_all[var_id] = value
            consumption[var_id] = policy

        ax[0].plot(var_grid, value_all, label=f"Choice {choice}")
        ax[1].plot(var_grid, consumption, label=f"Choice {choice}")
    ax[0].legend()
    ax[0].set_title("Value")
    ax[0].set_xlabel(var_name)
    ax[1].legend()
    ax[1].set_title("Consumption")
    ax[1].set_xlabel(var_name)
    plt.show()


import jax.numpy as jnp

discrete_state_to_plot = {
    "period": 69,
    "lagged_choice": 0,
    "policy_state": 29,
    "job_offer": 0,
    "education": 1,
    "health": 0,
    "sex": 0,
    "informed": 1,
    "partner_state": jnp.array(1),
    "experience": 0.325406,
    # "wealth": 50,
}
exp_grid = np.arange(0, 1, 0.1, dtype=float)
wealth_grid = np.arange(40, 80, 0.5, dtype=float)
plot_value(
    solution["value"],
    solution["endog_grid"],
    solution["policy"],
    "wealth",
    wealth_grid,
    discrete_state_to_plot,
    model,
    [0],
)
