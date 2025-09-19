import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from estimation.struct_estimation.scripts.estimate_setup import generate_print_func
from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)
from model_code.specify_model import specify_and_solve_model, specify_model
from model_code.specify_simple_model import specify_and_solve_simple_model
from set_paths import create_path_dict
from set_styles import set_plot_defaults
from specs.derive_specs import generate_derived_and_data_derived_specs

# paths, specs, create directories
path_dict = create_path_dict(define_user=True)
specs = generate_derived_and_data_derived_specs(path_dict)
show_plots = False
save_plots = True

# Set plot defaults
set_plot_defaults()

# Load model parameters - try estimated params first, fall back to start params
model_name = specs["model_name"]
estimated_params_path = path_dict["struct_results"] + f"est_params_{model_name}.pkl"

if os.path.exists(estimated_params_path):
    params = pickle.load(open(estimated_params_path, "rb"))
    params_source = "estimated"
else:
    print(f"WARNING: Estimated parameters file '{estimated_params_path}' not found.")
    print(
        "Using start values from estimation/struct_estimation/start_params_and_bounds/start_params.yaml"
    )

    # Load start parameters
    params = load_and_set_start_params(path_dict)
    params_source = "start_values"

generate_print_func(
    params.keys(), specs, print_men_examples=True, print_women_examples=True
)(params)


# Model solution plots (require solved model)
from model_code.plots.plot_law_of_motion import plot_ret_experience_multi

plot_ret_experience_multi(path_dict, specs, show=show_plots, save=save_plots)

# from model_code.plots.weights import plot_weights
#
# model = specify_model(
#     path_dict,
#     specs,
#     subj_unc=True,
#     custom_resolution_age=None,
#     sim_specs=None,
#     load_model=True,
#     debug_info=None,
#     sex_type="all",
#     edu_type="all",
#     util_type="add",
# )
# plot_weights(model, params, specs, path_dict)

from model_code.plots.retirement_probs_illustration import (
    plot_ret_probs_for_state,
    plot_solution,
    plot_work_probs_for_state,
)

try:
    model_solved = specify_and_solve_model(
        path_dict=path_dict,
        file_append=model_name,
        params=params,
        subj_unc=True,
        custom_resolution_age=None,
        load_model=False,
        load_solution=True,
        sim_specs=None,
        sex_type="all",
        edu_type="all",
        util_type="add",
    )

    plot_solution(model_solved=model_solved, specs=specs, path_dict=path_dict)
    plot_ret_probs_for_state(
        model_solved=model_solved, specs=specs, path_dict=path_dict
    )
    plot_work_probs_for_state(
        model_solved=model_solved, specs=specs, path_dict=path_dict
    )

    if params_source == "start_values":
        print(f"Model plots generated using start parameter values.")

except Exception as e:
    print(f"ERROR: Could not solve model: {str(e)}")
    print("Skipping model solution plots.")

# Income plots (uses model specifications - can run without solved model)
from model_code.plots.income_plots import plot_incomes

plot_incomes(path_dict, show=show_plots, save=save_plots)

# Wealth plots (uses model specifications - can run without solved model)
from model_code.plots.wealth_plots import plot_budget_of_unemployed

plot_budget_of_unemployed(path_dict, specs, show=show_plots, save=save_plots)

# Utility plots (require model parameters, if estimated parameters not available, uses start params. if start params incomplete, skips utility plots)
from model_code.plots.utility_plots import plot_bequest, plot_cons_scale, plot_utility

try:
    plot_utility(path_dict, params, specs, show=show_plots, save=save_plots)
    plot_bequest(path_dict, params, specs, show=show_plots, save=save_plots)
except KeyError as e:
    if params_source == "start_values":
        print(
            f"WARNING: Some utility plots skipped - missing parameters, e.g., {e} in start_params.yaml"
        )
    else:
        raise e


plot_cons_scale(path_dict, specs, show=show_plots, save=save_plots)

print("Model plotting completed.")
if params_source == "start_values":
    print(
        "Note: Some plots used start parameter values instead of estimated parameters."
    )
