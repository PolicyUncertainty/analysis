from set_paths import create_path_dict
from set_styles import set_plot_defaults
from specs.derive_specs import generate_derived_and_data_derived_specs

# paths, specs, create directories
path_dict = create_path_dict(define_user=True)
specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)
show_plots = False
save_plots = True

# Set plot defaults
set_plot_defaults()

# Retirement timing plots
from process_data.data_plots.retirement_timing import plot_retirement_timing_data
plot_retirement_timing_data(path_dict, specs, show=show_plots, save=save_plots)

# Choice distribution plots
from process_data.data_plots.choices import plot_data_choices
plot_data_choices(path_dict, specs, lagged=False, show=show_plots, save=save_plots)
plot_data_choices(path_dict, specs, lagged=True, show=show_plots, save=save_plots)
from process_data.data_plots.savings_rate import plot_savings_rate
plot_savings_rate(path_dict, specs, covariate="sex", show=show_plots, save=save_plots, window=5)

# State variable plots
from process_data.data_plots.states import plot_state_by_age_and_type
state_vars = [
    "mean experience",
    "mean wealth", 
    "mean health",
    "median wealth",
]
plot_state_by_age_and_type(path_dict, state_vars, specs, show=show_plots, save=save_plots)

# Wealth plots
from process_data.data_plots.wealth import plot_average_wealth_by_type
plot_average_wealth_by_type(path_dict, specs, show=show_plots, save=save_plots)

# Income plots (requires estimated parameters)
try:
    from process_data.data_plots.income import plot_income
    plot_income(path_dict, specs, show=show_plots, save=save_plots)
except FileNotFoundError as e:
    print(f"WARNING: Could not generate income plots - missing file: {e}")
    print("Income plots require estimated model parameters.")

print("Data plotting completed.")

# OLD INTERACTIVE CODE (soon to be removed):
# %%
# import matplotlib.pyplot as plt
# 
# which_plots = input(
#     "Which plots do you want to show?\n \n"
#     " - [a]ll\n"
#     " - [r]etirement timing\n"
#     # " - [f]resh retiree classificiation\n"
#     " - [c]hoices\n"
#     " - [s]tates\n"
#     " - [w]ealth\n"
#     " - [i]ncome\n"
# )
# 
# # %% ########################################
# # Retirement timing relative to SRA
# from process_data.data_plots.retirement_timing import plot_retirement_timing_data
# 
# if which_plots in ["a", "r"]:
#     plot_retirement_timing_data(path_dict, specs)
# 
# 
# # %%
# from process_data.data_plots.choices import plot_data_choices
# 
# if which_plots in ["a", "c"]:
#     plot_data_choices(path_dict)
#     plot_data_choices(path_dict, lagged=True)
# # %%
# from process_data.data_plots.states import plot_state_by_age_and_type
# 
# if which_plots in ["a", "s"]:
#     state_vars = [
#         "mean experience",
#         "mean wealth",
#         "mean health",
#         "median wealth",
#     ]
#     plot_state_by_age_and_type(path_dict, state_vars=state_vars)
# 
# from process_data.data_plots.wealth import plot_average_wealth_by_type
# 
# if which_plots in ["a", "w"]:
#     plot_average_wealth_by_type(path_dict)
# 
#     plt.show()
#     plt.close("all")
# 
# if which_plots in ["a", "i"]:
#     from process_data.data_plots.income import plot_income
# 
#     plot_income(path_dict, specs)
#     plt.show()
#     plt.close("all")
