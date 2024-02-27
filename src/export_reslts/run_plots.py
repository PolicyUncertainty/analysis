# Set paths of project
import sys
from pathlib import Path

analysis_path = str(Path(__file__).resolve().parents[2]) + "/"
sys.path.insert(0, analysis_path + "submodules/dcegm/src/")
sys.path.insert(0, analysis_path + "src/")
import matplotlib.pyplot as plt
from set_paths import create_path_dict

path_dict = create_path_dict(analysis_path)


##########################################
# Counterfactual 1 plots
##########################################
show_cf_1_plots = input("Show counterfactual 1 plots? (y/n): ") == "y"
from export_reslts.figures.policy_state_trajectories import trajectory_plot
from export_reslts.figures.sim_1_plots import (
    plot_average_savings,
    plot_full_time,
    plot_values_by_age,
)

trajectory_plot(path_dict)
plot_average_savings(path_dict)
plot_full_time(path_dict)
plot_values_by_age(path_dict)
if show_cf_1_plots:
    plt.show()

plt.close("all")
##########################################
# Wage plots
##########################################
show_wage_plots = input("Show wage plots? (y/n): ") == "y"
from export_reslts.figures.plot_wage import plot_wages

plot_wages(path_dict)
if show_wage_plots:
    plt.show()
plt.close("all")

##########################################
# Model fit plots
##########################################
show_model_fit_plots = input("Show model fit plots? (y/n): ") == "y"

from export_reslts.figures.model_fit_sim import (
    plot_average_wealth,
    plot_choice_shares,
    plot_choice_shares_single,
)

plot_average_wealth(path_dict)
plot_choice_shares(path_dict)
plot_choice_shares_single(path_dict)
if show_model_fit_plots:
    plt.show()
plt.close("all")
