import matplotlib.pyplot as plt
from set_paths import create_path_dict

path_dict = create_path_dict()


# %% ########################################
# # Data plots
# ##########################################
from export_results.figures.data_plots import plot_data_choices

# show_data = input("Execute data plots? (y/n)") == "y"
# if show_data:
plot_data_choices(path_dict)
plt.show()
plt.close("all")
