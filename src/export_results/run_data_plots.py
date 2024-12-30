# This script is used to run the data_plots.py script to generate plots of the structural estimation dataset.
# %%
import matplotlib.pyplot as plt
from set_paths import create_path_dict

path_dict = create_path_dict()

# %%
from export_results.figures.data_plots import plot_data_choices

show_choices = input("Show data choice plots? (y/n) ") == "y"
if show_choices:
    plot_data_choices(path_dict)
    plt.show()
    plt.close("all")
show_lagged_choices = input("Show lagged data choice plots? (y/n) ") == "y"
if show_lagged_choices:
    plot_data_choices(path_dict, lagged=True)
    plt.show()
    plt.close("all")
# %%
from export_results.figures.data_plots import plot_state_by_age_and_type

show_state = input("Show state plots? (y/n) ") == "y"
if show_state:
    state_vars = [
        "mean experience",
        "mean wealth",
        "mean health",
        "median wealth",
    ]
    plot_state_by_age_and_type(path_dict, state_vars=state_vars)
    plt.show()
    plt.close("all")
