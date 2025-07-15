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
load_sol_model = True


params = pickle.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

which_plots = input(
    "Which plots do you want to show?\n \n"
    " - [a]ll\n"
    " - [c]onsumption\n"
    " - [u]tility\n"
    " - [w]ealth\n"
    " - [s]avings\n"
)


from simulation.figures.illustrate_sim_data import plot_sim_vars

if which_plots in ["a", "c"]:
    plot_sim_vars(
        path_dict,
        specs,
        params,
        model_name,
        plot_dead=False,
        sim_var="consumption",
        load_df=load_df,
        load_solution=load_solution,
        load_sol_model=load_sol_model,
    )
    # After running, we can set all to true
    load_df = True
    load_solution = True
    load_sim_model = True
    load_sol_model = True

if which_plots in ["a", "u"]:
    plot_sim_vars(
        path_dict,
        specs,
        params,
        model_name,
        plot_dead=False,
        sim_var="utility",
        load_df=load_df,
        load_solution=load_solution,
        load_sol_model=load_sol_model,
    )
    # After running, we can set all to true
    load_df = True
    load_solution = True
    load_sim_model = True
    load_sol_model = True

if which_plots in ["a", "w"]:
    plot_sim_vars(
        path_dict,
        specs,
        params,
        model_name,
        plot_dead=False,
        sim_var="wealth_at_beginning",
        load_df=load_df,
        load_solution=load_solution,
        load_sol_model=load_sol_model,
    )
    # After running, we can set all to true
    load_df = True
    load_solution = True
    load_sim_model = True
    load_sol_model = True

if which_plots in ["a", "s"]:
    plot_sim_vars(
        path_dict,
        specs,
        params,
        model_name,
        plot_dead=False,
        sim_var="savings_dec",
        load_df=load_df,
        load_solution=load_solution,
        load_sol_model=load_sol_model,
    )
    # After running, we can set all to true
    load_df = True
    load_solution = True
    load_sim_model = True
    load_sol_model = True


plt.show()
