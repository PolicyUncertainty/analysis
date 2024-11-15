# %% Set paths of project
import pickle

import matplotlib.pyplot as plt
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()

specs = generate_derived_and_data_derived_specs(path_dict)

kind_string = input("Execute [pre]- or [post]-estimation plots? (pre/post) ")

if kind_string == "pre":
    from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
        load_and_set_start_params,
    )

    params = load_and_set_start_params(path_dict)
    pt_ratio_low = specs["av_annual_hours_pt"][0] / specs["av_annual_hours_ft"][0]
    pt_ratio_high = specs["av_annual_hours_pt"][1] / specs["av_annual_hours_ft"][1]
    params["dis_util_unemployed_low"] = params["dis_util_not_retired_low"]
    params["dis_util_pt_work_low"] = (
        params["dis_util_not_retired_low"]
        + pt_ratio_low * params["dis_util_working_low"]
    )
    params["dis_util_ft_work_low"] = (
        params["dis_util_not_retired_low"] + params["dis_util_working_low"]
    )

    params["dis_util_unemployed_high"] = params["dis_util_not_retired_high"]
    params["dis_util_pt_work_high"] = (
        params["dis_util_not_retired_high"]
        + pt_ratio_high * params["dis_util_working_high"]
    )
    params["dis_util_ft_work_high"] = (
        params["dis_util_not_retired_high"] + params["dis_util_working_high"]
    )

elif kind_string == "post":
    params = pickle.load(open(path_dict["est_params"], "rb"))

else:
    raise ValueError("Either pre or post estimation plots.")

show_any_plots = input("Show any plots? (y/n): ") == "y"

# %%##########################################
# # Model fit plots
# ##########################################
if show_any_plots:
    show_model_fit_plots = input("Show model fit plots? (y/n): ") == "y"
else:
    show_model_fit_plots = False
from export_results.figures.observed_model_fit import observed_model_fit

observed_model_fit(path_dict, specs, params)
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
if show_sim_plots:
    plt.show()
plt.close("all")

# %%
