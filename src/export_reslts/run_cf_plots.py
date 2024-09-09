# %% Set paths of project
import matplotlib.pyplot as plt
from set_paths import create_path_dict

path_dict = create_path_dict()

show_any_plots = input("Show any plots? (y/n): ") == "y"

# %% ##########################################
# # Counterfactual 1 plots
# ##########################################
if show_any_plots:
    show_cf_1_plots = input("Show counterfactual 1 plots? (y/n): ") == "y"
else:
    show_cf_1_plots = False
from export_reslts.figures.counterfactual_no_unc import (
    plot_average_savings,
    plot_full_time,
    trajectory_plot,
)

# trajectory_plot(path_dict)
plot_average_savings(path_dict)
plot_full_time(path_dict)
if show_cf_1_plots & show_any_plots:
    plt.show()

plt.close("all")


# %%##########################################
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

# %%
