# %% Set paths of project
import pickle

import matplotlib.pyplot as plt
import pandas as pd

from estimation.struct_estimation.scripts.estimate_setup import generate_print_func
from estimation.struct_estimation.scripts.observed_model_fit import (
    plot_life_cycle_choice_probs,
    plot_retirement_fit,
)
from estimation.struct_estimation.scripts.print_ll_info import (
    print_choice_probs_by_group,
)
from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)
from model_code.specify_model import specify_and_solve_model
from model_code.transform_data_from_model import (
    calc_choice_probs_for_df,
    load_scale_and_correct_data,
)
from set_paths import create_path_dict, get_model_resutls_path
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)

# Set run specs
model_name = specs["model_name"]
print(f"Running model: {model_name}")
load_sol_model = True
load_solution = None
load_data_from_sol = True


# check if folder of model objects exits:
model_folder = get_model_resutls_path(path_dict, model_name)

if load_data_from_sol:
    data_decision = pd.read_csv(
        model_folder["model_results"] + "data_with_probs.csv", index_col=0
    )
    # df = data_decision[
    #     (data_decision["lagged_choice"] != 0) & (data_decision["sex"] == 0)
    # ]

else:
    params = pickle.load(
        open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
    )
    generate_print_func(params.keys(), specs)(params)

    model_solved = specify_and_solve_model(
        path_dict=path_dict,
        params=params,
        subj_unc=True,
        custom_resolution_age=None,
        file_append=model_name,
        load_model=load_sol_model,
        load_solution=load_solution,
        sim_specs=None,
        debug_info="all",
    )

    data_decision = load_scale_and_correct_data(
        path_dict=path_dict, model_class=model_solved
    )

    data_decision = calc_choice_probs_for_df(
        df=data_decision, params=params, model_solved=model_solved
    )
    data_decision.to_csv(model_folder["model_results"] + "data_with_probs.csv")

plot_life_cycle_choice_probs(
    specs=specs,
    data_decision=data_decision,
    save_folder=path_dict["plots"],
)

plot_retirement_fit(
    specs=specs,
    data_decision=data_decision,
    save_folder=path_dict["plots"],
)

print_choice_probs_by_group(df=data_decision, specs=specs, path_dict=path_dict)

plt.show()
plt.close("all")

# %%
