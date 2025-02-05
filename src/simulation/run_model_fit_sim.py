# %% Set paths of project
import pickle

import matplotlib.pyplot as plt
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)


model_name = "new"
load_df = True
load_solution = True
load_sim_model = True
load_sol_model = True


params = pickle.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

which_plots = input(
    "Which plots do you want to show?\n \n"
    " - [a]ll\n"
    " - [c]hoices\n"
    " - [w]ealth\n"
)

from simulation.figures.simulated_model_fit import (
    plot_average_wealth,
    plot_choice_shares_single,
)

if which_plots in ["a", "s"]:
    # plot_states(path_dict, specs)
    plot_choice_shares_single(
        path_dict=path_dict,
        specs=specs,
        params=params,
        model_name=model_name,
        load_df=load_df,
        load_solution=load_solution,
        load_sol_model=load_sol_model,
        load_sim_model=load_sim_model,
    )
    # After first run all loading is true
    load_df = True
    load_solution = True
    load_sim_model = True
    load_sol_model = True

if which_plots in ["a", "w"]:
    plot_average_wealth(
        path_dict=path_dict,
        specs=specs,
        params=params,
        model_name=model_name,
        load_df=load_df,
        load_solution=load_solution,
        load_sol_model=load_sol_model,
        load_sim_model=load_sim_model,
    )
plt.show()
