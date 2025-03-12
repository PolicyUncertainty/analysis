# %% Set paths of project
import matplotlib.pyplot as plt
import pandas as pd

from set_paths import create_path_dict
from simulation.figures.sra_increase import plot_aggregate_results

path_dict = create_path_dict()

model_name = "partner_est"

# Exclude the first rpw
plot_aggregate_results(path_dict, model_name)
plt.show()

#
# show_any_plots = input("Show any plots? (y/n): ") == "y"
#
# # %% ##########################################
# # # Counterfactual 1 plots
# # ##########################################
# if show_any_plots:
#     show_cf_1_plots = input("Show counterfactual 1 plots? (y/n): ") == "y"
# else:
#     show_cf_1_plots = False
# from export_results.figures.counterfactual_no_unc import (
#     plot_average_savings,
#     plot_full_time,
#     trajectory_plot,
# )
#
# # trajectory_plot(path_dict)
# plot_average_savings(path_dict)
# plt.show()
# # plot_full_time(path_dict)
# # if show_cf_1_plots & show_any_plots:
# #     plt.show()
#
# plt.close("all")
#
#
# # %%##########################################
# # Counterfactual 2 plots
# ###########################################
# if show_any_plots:
#     show_cf_2_plots = input("Show counterfactual 2 plots? (y/n): ") == "y"
# else:
#     show_cf_2_plots = False
#
# from export_results.figures.counterfactual_bias import (
#     plot_savings_over_age,
#     plot_step_functions,
# )
#
# plot_savings_over_age(path_dict)
# plot_step_functions(path_dict)
# if show_cf_2_plots:
#     plt.show()
# plt.close("all")
#
# # %%
