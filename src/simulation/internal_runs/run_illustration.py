# %% Set paths of project
import pickle

from model_code.specify_model import specify_model
from set_paths import create_path_dict
from simulation.sim_tools.calc_exp_results import investigate_start_obs
from simulation.sim_tools.start_obs_for_sim import (
    generate_start_states_from_obs,
)
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)


model_name = specs["model_name"]
util_type = specs["util_type"]

load_df = None
load_solution = None
load_sol_model = True


model = specify_model(
    path_dict,
    specs,
    custom_resolution_age=None,
    subj_unc=True,
    load_model=load_sol_model,
    util_type=util_type,
)
generate_start_states_from_obs(
    path_dict=path_dict,
    params=pickle.load(
        open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
    ),
    model_class=model,
    inital_SRA=65,
    only_informed=False,
)
initial_obs_table = investigate_start_obs(
    model_class=model,
    path_dict=path_dict,
)
# params = pickle.load(
#     open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
# )

# which_plots = input(
#     "Which plots do you want to show?\n \n"
#     " - [a]ll\n"
#     " - [c]onsumption\n"
#     " - [u]tility\n"
#     " - [w]ealth\n"
#     " - [s]avings\n"
# )
# which_plots = "s"
#
# from simulation.figures.savings_rate import plot_savings
#
# plot_savings(
#     path_dict=path_dict,
#     specs=specs,
#     params=params,
#     model_name=model_name,
#     file_name="savings_rate_illustration",
#     load_df=load_df,
#     load_solution=load_solution,
#     load_sol_model=load_sol_model,
#     util_type=util_type,
# )


#
# if which_plots in ["a", "c"]:
#     plot_sim_vars(
#         path_dict,
#         specs,
#         params,
#         model_name,
#         plot_dead=False,
#         sim_var="consumption",
#         load_df=load_df,
#         load_solution=load_solution,
#         load_sol_model=load_sol_model,
#     )
#     # After running, we can set all to true
#     load_df = True
#     load_solution = True
#     load_sim_model = True
#     load_sol_model = True
#
# if which_plots in ["a", "u"]:
#     plot_sim_vars(
#         path_dict,
#         specs,
#         params,
#         model_name,
#         plot_dead=False,
#         sim_var="utility",
#         load_df=load_df,
#         load_solution=load_solution,
#         load_sol_model=load_sol_model,
#     )
#     # After running, we can set all to true
#     load_df = True
#     load_solution = True
#     load_sim_model = True
#     load_sol_model = True
#
# if which_plots in ["a", "w"]:
#     plot_sim_vars(
#         path_dict,
#         specs,
#         params,
#         model_name,
#         plot_dead=False,
#         sim_var="wealth_at_beginning",
#         load_df=load_df,
#         load_solution=load_solution,
#         load_sol_model=load_sol_model,
#     )
#     # After running, we can set all to true
#     load_df = True
#     load_solution = True
#     load_sim_model = True
#     load_sol_model = True
#
# if which_plots in ["a", "s"]:
#     plot_sim_vars(
#         path_dict,
#         specs,
#         params,
#         model_name,
#         plot_dead=False,
#         sim_var="net_hh_income",
#         load_df=load_df,
#         load_solution=load_solution,
#         load_sol_model=load_sol_model,
#     )
#     # After running, we can set all to true
#     load_df = True
#     load_solution = True
#     load_sim_model = True
#     load_sol_model = True
#
#
# plt.show()
