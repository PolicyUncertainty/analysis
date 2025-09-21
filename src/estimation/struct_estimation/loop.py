# Set paths of project
import pickle as pkl

from set_paths import create_path_dict

paths_dict = create_path_dict(define_user=False)
from estimation.struct_estimation.scripts.estimate_setup import estimate_model
from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)

LOAD_SOL_MODEL = True
SAVE_RESULTS = False
USE_WEIGHTS = False

# Load start params
start_params_all = load_and_set_start_params(paths_dict)


# %% Set paths of project
import pickle as pkl

from estimation.struct_estimation.scripts.observed_model_fit import create_fit_plots
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)

# Set run specs
model_name = "all_men_3_cobb"
print(f"Running model: {model_name}")
load_sol_model = False
load_solution = None
load_data_from_sol = False


for tase_shock_scale in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
    start_params_all["taste_shock_scale_men"] = tase_shock_scale

    model_name = f"{tase_shock_scale:.2f}_scale"
    # estimation_results = estimate_model(
    #     paths_dict,
    #     params_to_estimate_names=params_to_estimate_names,
    #     file_append=model_name,
    #     load_model=LOAD_SOL_MODEL,
    #     start_params_all=start_params_all,
    #     use_weights=USE_WEIGHTS,
    #     last_estimate=None,
    #     save_results=SAVE_RESULTS,
    # )
    # print(estimation_results)

    create_fit_plots(
        path_dict=path_dict,
        specs=specs,
        params=start_params_all,
        model_name=model_name,
        load_sol_model=load_sol_model,
        load_solution=load_solution,
        load_data_from_sol=load_data_from_sol,
    )


# %%
