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
elif kind_string == "post":
    params = pickle.load(open(path_dict["est_params"], "rb"))
else:
    raise ValueError("Either pre or post estimation plots.")

which_plot = input("Which plot to show? ([a]ll/[f]it]/[s]im): ")


# %%##########################################
# # Model fit plots
# ##########################################

if which_plot in ["a", "f"]:
    from export_results.figures.observed_model_fit import observed_model_fit

    observed_model_fit(path_dict, specs, params)
    plt.show()
plt.close("all")

##########################################
# Model fit plots simulated
##########################################

if which_plot in ["a", "s"]:
    from export_results.figures.simulated_model_fit import (
        plot_states,
        plot_average_wealth,
        plot_choice_shares_single,
    )

    # plot_states(path_dict, specs)
    plot_choice_shares_single(path_dict, specs, params)
    plt.show()
    #
    # plot_average_wealth(path_dict, specs)
    # plt.show()


# %%
