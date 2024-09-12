# %% Set paths of project
import matplotlib.pyplot as plt
from set_paths import create_path_dict

path_dict = create_path_dict()

show_any_plots = input("Show any plots? (y/n): ") == "y"

# %%##########################################
# # Model fit plots
# ##########################################
if show_any_plots:
    show_model_fit_plots = input("Show model fit plots? (y/n): ") == "y"
else:
    show_model_fit_plots = False
from export_results.figures.observed_model_fit import observed_model_fit

observed_model_fit(path_dict)
if show_model_fit_plots:
    plt.show()
plt.close("all")

##########################################
# Model fit plots simulated
##########################################
if show_any_plots:
    show_sim_plots = input("Show simulated model fit plots? (y/n): ") == "y"
else:
    show_sim_plots = False

from export_results.figures.simulated_model_fit import (
    plot_average_wealth,
    # plot_choice_shares,
    # plot_choice_shares_single,
    # illustrate_simulated_data,
)

plot_average_wealth(path_dict)
# plot_choice_shares(path_dict)
# plot_choice_shares_single(path_dict)
# illustrate_simulated_data(path_dict)
if show_model_fit_plots:
    plt.show()
plt.close("all")

# %%
