# %% Set paths of project
import matplotlib.pyplot as plt
import pandas as pd
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)

kind_string = input("Execute [pre]- or [post]-estimation plots? (pre/post) ")

if kind_string == "pre":
    from estimation.struct_estimation.start_params.set_start_params import (
        load_and_set_start_params,
    )

    params = load_and_set_start_params(path_dict)
elif kind_string == "post":
    params = pd.read_pickle(path_dict["est_params"])

else:
    raise ValueError("Either pre or post estimation plots.")


# %%###################################
# Family chracteristics
######################################
exec_family = input("Exectue family characteristics? (y/n) ") == "y"
if exec_family:
    from export_results.figures.family_params import (
        plot_children,
        plot_marriage_and_divorce,
    )

    plot_children(path_dict, specs)
    plot_marriage_and_divorce(path_dict, specs)
    plt.show()
    plt.close("all")


# %% ########################################
# # Utility plots
# ##########################################
exec_utility = input("Execute utility plots? (y/n) ") == "y"
if exec_utility:
    from export_results.figures.utility import plot_utility, plot_cons_scale

    plot_utility(params, specs)
    plot_cons_scale(specs)
    plt.show()
    plt.close("all")


# %% ########################################
# Job offer plots
# ##########################################
exec_job_offer = input("Execute job offer plots? (y/n) ") == "y"
from export_results.figures.job_offer_plots import plot_job_separation

if exec_job_offer:
    plot_job_separation(path_dict, params)
    plt.show()
    plt.close("all")


# %% ########################################
# # Budget plots
# ##########################################
exec_budget = input("Execute budget plots? (y/n)") == "y"
from export_results.figures.income_plots import (
    plot_incomes,
    plot_total_income,
    plot_partner_wage,
    plot_child_benefits,
)

if exec_budget:
    plot_incomes(path_dict)
    plot_partner_wage(path_dict, specs)
    plot_total_income(specs)
    plot_child_benefits(specs)
    plt.show()
    plt.close("all")


# %% ########################################
# # SRA plots
# ##########################################
show_SRA = input("Execute SRA plots? (y/n)") == "y"
from export_results.figures.expected_SRA_plots import plot_markov_process

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
