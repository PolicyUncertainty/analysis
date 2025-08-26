# Set paths of project
import pickle as pkl

from set_paths import create_path_dict

paths_dict = create_path_dict(define_user=False)
from estimation.msm.scripts.msm_estimation_setup import estimate_model
from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)

params_to_estimate_names = ["bequest_scale", "mu_bequest_high", "mu_bequest_low", "mu"]
model_name = "whole"
LOAD_LAST_ESTIMATE = True
LOAD_SOL_MODEL = True
SAVE_RESULTS = True
USE_WEIGHTS = False

if LOAD_LAST_ESTIMATE:
    # last_estimate = pkl.load(
    #     open(paths_dict["struct_results"] + f"est_params_msm_first.pkl", "rb")
    # )
    from estimation.struct_estimation.map_params_to_current import (
        merge_men_and_women_params,
    )

    last_estimate = merge_men_and_women_params(
        path_dict=paths_dict, ungendered_model_name=model_name
    )
else:
    last_estimate = None


# Load start params
start_params_all = load_and_set_start_params(paths_dict)

estimation_results = estimate_model(
    paths_dict,
    params_to_estimate_names=params_to_estimate_names,
    file_append=model_name,
    start_params_all=start_params_all,
    load_model=LOAD_SOL_MODEL,
    # use_weights=USE_WEIGHTS,
    last_estimate=None,
    weighting_method="diagonal",
    # save_results=SAVE_RESULTS,
)
print(estimation_results)


# %%
