# Set paths of project
import sys
from pathlib import Path

analysis_path = str(Path(__file__).resolve().parents[2]) + "/"
sys.path.insert(0, analysis_path + "submodules/dcegm/src/")
sys.path.insert(0, analysis_path + "src/")
import matplotlib.pyplot as plt
from set_paths import create_path_dict

path_dict = create_path_dict(analysis_path)

show_any_plots = input("Show any plots? (y/n): ") == "y"

# ##########################################
# # Policy state plot
# ##########################################
if show_any_plots:
    show_policy_state_plot = input("Show policy state plot? (y/n): ") == "y"
else:
    show_policy_state_plot = False
from export_reslts.figures.policy_states import plot_SRA_2007_reform

plot_SRA_2007_reform(path_dict)
if show_policy_state_plot:
    plt.show()
plt.close("all")

# ##########################################
# # Expected SRA
# ##########################################
if show_any_plots:
    show_expected_SRA_plot = input("Show expected SRA plot? (y/n): ") == "y"
else:
    show_expected_SRA_plot = False
from export_reslts.figures.expected_SRA_plots import plot_markov_process

plot_markov_process(path_dict)
if show_expected_SRA_plot:
    plt.show()
plt.close("all")


# ##########################################
# # Wage plots
# ##########################################
if show_any_plots:
    show_wage_plots = input("Show wage plots? (y/n): ") == "y"
else:
    show_wage_plots = False
from export_reslts.figures.income_plots import plot_incomes

plot_incomes(path_dict)
if show_wage_plots:
    plt.show()
plt.close("all")


# ##########################################
# # Counterfactual 1 plots
# ##########################################
if show_any_plots:
    show_cf_1_plots = input("Show counterfactual 1 plots? (y/n): ") == "y"
else:
    show_cf_1_plots = False
from export_reslts.figures.counterfactual_no_unc import (
    plot_average_savings,
    plot_full_time,
    plot_values_by_age,
    trajectory_plot,
)

trajectory_plot(path_dict)
# plot_average_savings(path_dict)
# plot_full_time(path_dict)
# plot_values_by_age(path_dict)
if show_cf_1_plots & show_any_plots:
    plt.show()

plt.close("all")


###########################################
# Counterfactual 2 plots
###########################################
if show_any_plots:
    show_cf_2_plots = input("Show counterfactual 2 plots? (y/n): ") == "y"
else:
    show_cf_2_plots = False

from export_reslts.figures.counterfactual_bias import (
    plot_savings_over_age,
    plot_step_functions,
)

plot_savings_over_age(path_dict)
plot_step_functions(path_dict)
if show_cf_2_plots:
    plt.show()
plt.close("all")

# ##########################################
# # Model fit plots
# ##########################################
if show_any_plots:
    show_model_fit_plots = input("Show model fit plots? (y/n): ") == "y"
else:
    show_model_fit_plots = False
from export_reslts.figures.observed_model_fit import observed_model_fit

observed_model_fit(path_dict)
if show_model_fit_plots:
    plt.show()
plt.close("all")

##########################################
# Model fit plots simulated
##########################################
if show_any_plots:
    show_model_fit_plots = input("Show model fit plots? (y/n): ") == "y"
else:
    show_model_fit_plots = False

from export_reslts.figures.simulated_model_fit import (
    plot_average_wealth,
    plot_choice_shares,
    plot_choice_shares_single,
    illustrate_simulated_data,
)

plot_average_wealth(path_dict)
plot_choice_shares(path_dict)
plot_choice_shares_single(path_dict)
illustrate_simulated_data(path_dict)
if show_model_fit_plots:
    plt.show()
plt.close("all")
