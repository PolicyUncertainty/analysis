# %% Set paths of project
import pickle

import jax
import matplotlib.pyplot as plt
import numpy as np

from estimation.struct_estimation.map_params_to_current import (
    map_period_to_age,
    new_to_current,
)
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)


model_name = "disability"
load_df = True
load_solution = True
load_sim_model = True
load_sol_model = True


params = pickle.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

# params = map_period_to_age(params)

# params = new_to_current(path_dict)
#
which_plots = input(
    "Which plots do you want to show?\n \n"
    " - [a]ll\n"
    " - [c]hoices\n"
    " - [w]ealth\n"
    " - [i]ncome\n"
    " - [s]tates\n"
    " - [wc]hoices and wealth\n"
)
print(jax.devices())
# which_plots = "wc"

from simulation.figures.simulated_model_fit import (
    plot_choice_shares_single,
    plot_quantiles,
    plot_states,
)

if which_plots in ["a", "c", "wc"]:
    plot_choice_shares_single(
        path_dict=path_dict,
        specs=specs,
        params=params,
        file_name="sim_choices",
        model_name=model_name,
        load_df=load_df,
        load_solution=load_solution,
        load_sol_model=load_sol_model,
        load_sim_model=load_sim_model,
    )
    # After running, we can set all to true
    load_df = True if load_df is not None else load_df
    load_solution = True if load_solution is not None else load_solution
    load_sim_model = True
    load_sol_model = True

if which_plots in ["a", "w", "wc"]:
    plot_quantiles(
        path_dict=path_dict,
        specs=specs,
        params=params,
        model_name=model_name,
        quantiles=[0.5],
        sim_col_name="wealth_at_beginning",
        obs_col_name="adjusted_wealth",
        file_name="average_wealth",
        load_df=load_df,
        load_solution=load_solution,
        load_sol_model=load_sol_model,
        load_sim_model=load_sim_model,
    )
    # After running, we can set all to true
    load_df = True if load_df is not None else load_df
    load_solution = True if load_solution is not None else load_solution
    load_sim_model = True
    load_sol_model = True


if which_plots in ["a", "s"]:
    plot_states(
        path_dict,
        specs,
        params,
        model_name,
        load_df=load_df,
        load_solution=load_solution,
        load_sol_model=load_sol_model,
        load_sim_model=load_sim_model,
    )
    # After running, we can set all to true
    load_df = True if load_df is not None else load_df
    load_solution = True if load_solution is not None else load_solution
    load_sim_model = True
    load_sol_model = True

if which_plots in ["a", "i"]:
    plot_quantiles(
        path_dict=path_dict,
        specs=specs,
        params=params,
        model_name=model_name,
        quantiles=[0.5],
        sim_col_name="total_income",
        obs_col_name="hh_net_income",
        file_name=None,
        load_df=load_df,
        load_solution=load_solution,
        load_sol_model=load_sol_model,
        load_sim_model=load_sim_model,
    )


plt.show()
