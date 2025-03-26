# This script is used to run the data_plots.py script to generate plots of the structural estimation dataset.
# %%
import matplotlib.pyplot as plt
from set_paths import create_path_dict

path_dict = create_path_dict()
from specs.derive_specs import generate_derived_and_data_derived_specs

specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)

which_plots = input(
    "Which plots do you want to show?\n \n"
    " - [a]ll\n"
    " - [r]etirement timing\n"
    " - [f]resh retiree classificiation\n"
    " - [c]hoices\n"
    " - [s]tates\n"
    " - [w]ealth\n"
    
)

# %% ########################################
# Retirement timing relative to SRA
from process_data.aux_and_plots.retirement_timing import plot_retirement_timing_data

if which_plots in ["a", "r"]:
    plot_retirement_timing_data(path_dict, specs)


# %%
from process_data.aux_and_plots.data_plots import plot_data_choices

if which_plots in ["a", "c"]:
    plot_data_choices(path_dict)
    plot_data_choices(path_dict, lagged=True)
# %%
from process_data.aux_and_plots.data_plots import plot_state_by_age_and_type

if which_plots in ["a", "s"]:
    state_vars = [
        "mean experience",
        "mean wealth",
        "mean health",
        "median wealth",
    ]
    plot_state_by_age_and_type(path_dict, state_vars=state_vars)

from process_data.aux_and_plots.wealth import plot_average_wealth_by_type

if which_plots in ["a", "w"]:
    plot_average_wealth_by_type(path_dict)

plt.show()
plt.close("all")

if which_plots in ["a", "f"]:
    from process_data.aux_and_plots.retiree_classification import plot_retiree_classification
    path_dict = create_path_dict(define_user=True)
    plot_retiree_classification(path_dict)
    plt.show()
    plt.close("all")
