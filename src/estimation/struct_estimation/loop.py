# Set paths of project
import pickle as pkl

from set_paths import create_path_dict

paths_dict = create_path_dict(define_user=False)
from estimation.struct_estimation.scripts.estimate_setup import estimate_model
from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)

params_to_estimate_names = [
    # "mu_men",
    # Men Full-time - 4 parameters
    "disutil_ft_work_high_good_men",
    "disutil_ft_work_high_bad_men",
    "disutil_ft_work_low_good_men",
    "disutil_ft_work_low_bad_men",
    # Men unemployment - 2 parameters
    "disutil_unemployed_high_men",
    "disutil_unemployed_low_men",
    # Taste shock men - 1 parameter
    # "taste_shock_scale_men",
]

LOAD_SOL_MODEL = True
SAVE_RESULTS = True
USE_WEIGHTS = False

# Load start params
start_params_all = load_and_set_start_params(paths_dict)


for tase_shock_scale in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
    start_params_all["taste_shock_scale_men"] = tase_shock_scale

    model_name = f"{tase_shock_scale:.2f}_scale"
    estimation_results = estimate_model(
        paths_dict,
        params_to_estimate_names=params_to_estimate_names,
        file_append=model_name,
        load_model=LOAD_SOL_MODEL,
        start_params_all=start_params_all,
        use_weights=USE_WEIGHTS,
        last_estimate=None,
        save_results=SAVE_RESULTS,
    )
    print(estimation_results)


# %%
