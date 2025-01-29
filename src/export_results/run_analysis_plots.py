# %% Set paths of project
import pickle

import matplotlib.pyplot as plt
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)
model_name = "new"

# kind_string = input("Execute [pre]- or [post]-estimation plots? (pre/post)\n")

# if kind_string == "pre":
#     from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
#         load_and_set_start_params,
#     )
#
#     params = load_and_set_start_params(path_dict)
# elif kind_string == "post":
#     params = pickle.load(
#         open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
#     )
# else:
#     raise ValueError("Either pre or post estimation plots.")

from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)

params = load_and_set_start_params(path_dict)

# %%###################################
# Health characteristics
######################################
# exec_health = input("Show health transition plots? (y/n) ") == "y"
# if exec_health:
from export_results.figures.expected_health import (
    plot_healthy_unhealthy,
    plot_health_transition_prob,
)

#
# plot_healthy_unhealthy(path_dict, specs)
# plt.show()
# plot_health_transition_prob(specs)
# plt.show()
# plt.close("all")
#
# # %%###################################
# # Mortality characteristics
# ######################################
# exec_mortality = input("Execute Mortality characteristics? (y/n) ") == "y"
# if exec_mortality:
from export_results.figures.plot_mortality import (
    plot_mortality,
)

plot_mortality(path_dict, specs)
plt.show()
plt.close("all")

exit()
# %%###################################
# Family characteristics
######################################
# exec_family = input("Execute family characteristics? (y/n) ") == "y"
# if exec_family:
# from export_results.figures.family_params import (
#     plot_children,
#     plot_marriage_and_divorce,
# )
#
# plot_children(path_dict, specs)
# plot_marriage_and_divorce(path_dict, specs)
# plt.show()
# plt.close("all")


# %% ########################################
# # Utility plots
# ##########################################
# exec_utility = input("Show utility plots? (y/n) ") == "y"
# if exec_utility:
#     from export_results.figures.utility import (
#         plot_utility,
#         plot_cons_scale,
#         plot_bequest,
#     )
#
#     plot_utility(params, specs)
#     plot_cons_scale(specs)
#     plot_bequest(params, specs)
#     plt.show()
#     plt.close("all")


# %% ########################################
# Job offer plots
# ##########################################
# exec_job_offer = input("Show job offer plots? (y/n) ") == "y"
from export_results.figures.job_offer_plots import plot_job_transitions

# if exec_job_offer:
plot_job_transitions(
    path_dict=path_dict,
    specs=specs,
    # params=params
)
plt.show()
plt.close("all")


# %% ########################################
# # Budget plots
# ##########################################
exec_budget = input("Execute budget plots? (y/n) ") == "y"
from export_results.figures.income_plots import (
    plot_incomes,
    plot_total_income,
    plot_partner_wage,
    plot_child_benefits,
)
from export_results.figures.wealth_plots import plot_budget_of_unemployed

if exec_budget:
    # plot_incomes(path_dict)
    # plot_partner_wage(path_dict, specs)
    # plot_total_income(specs)
    # plot_child_benefits(specs)
    plot_budget_of_unemployed(specs)
    plt.show()
    plt.close("all")


# %% ########################################
# # SRA plots
# ##########################################
show_SRA = input("Show SRA plots? (y/n)") == "y"
from export_results.figures.expected_policy_plots import plot_markov_process

if show_SRA:
    plot_markov_process(path_dict)
    plt.show()
    plt.close("all")

# %% ##########################################
# # Policy state plot
# ##########################################
# from export_results.figures.policy_states import plot_SRA_2007_reform
#
# plot_SRA_2007_reform(path_dict)
# plt.show()
# plt.close("all")

#
# from export_results.figures.pension_npv import plot_pension_npv_by_age
#
# plot_pension_npv_by_age(path_dict)
# plt.show()
# plt.close("all")
# %%
