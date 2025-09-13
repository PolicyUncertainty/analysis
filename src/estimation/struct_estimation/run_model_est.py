# Set paths of project
import pickle as pkl
import sys

from estimation.struct_estimation.start_params_and_bounds.param_lists import (
    high_men_disutil_firing,
    high_women_disutil_firing,
    low_men_disutil_firing,
    low_women_disutil_firing,
)
from set_paths import create_path_dict

paths_dict = create_path_dict(define_user=False)
from estimation.struct_estimation.scripts.estimate_setup import estimate_model
from estimation.struct_estimation.scripts.observed_model_fit import create_fit_plots
from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)

param_lists = {
    "men": {"low": low_men_disutil_firing, "high": high_men_disutil_firing},
    "women": {"low": low_women_disutil_firing, "high": high_women_disutil_firing},
}

model_name = sys.argv[1]
if "_men" in model_name:
    sex_type = "men"
elif "_women" in model_name:
    sex_type = "women"
else:
    sex_type = "all"

# Determine education type
if "low" in model_name:
    edu_type = "low"
elif "high" in model_name:
    edu_type = "high"
else:
    edu_type = "all"

params_to_estimate_names = param_lists[sex_type][edu_type]


if "add" in model_name:
    util_type = "add"
elif "cobb" in model_name:
    util_type = "cobb"
else:
    raise ValueError("unknown model")

print(f"Running estimation for model: {model_name}", flush=True)

LOAD_LAST_ESTIMATE = False
LOAD_SOL_MODEL = False
SAVE_RESULTS = True
USE_WEIGHTS = False

if LOAD_LAST_ESTIMATE:
    last_estimate = pkl.load(
        open(paths_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
    )
else:
    last_estimate = None

# Load start params
start_params_all = load_and_set_start_params(paths_dict)

# Run estimation
estimation_results, end_params = estimate_model(
    paths_dict,
    params_to_estimate_names=params_to_estimate_names,
    file_append=model_name,
    start_params_all=start_params_all,
    load_model=LOAD_SOL_MODEL,
    use_weights=USE_WEIGHTS,
    last_estimate=last_estimate,
    save_results=SAVE_RESULTS,
    sex_type=sex_type,
    edu_type=edu_type,
    util_type=util_type,
    slow_version=True,
    scale_opt=True,
    multistart=True,
)
print(estimation_results)

# %% Set paths of project
from specs.derive_specs import generate_derived_and_data_derived_specs

specs = generate_derived_and_data_derived_specs(paths_dict)


create_fit_plots(
    path_dict=paths_dict,
    specs=specs,
    params=end_params,
    model_name=model_name,
    load_sol_model=True,
    load_solution=None,
    load_data_from_sol=False,
    sex_type=sex_type,
    edu_type=edu_type,
)


# %%
