# Set paths of project
import pickle as pkl
import sys

from estimation.struct_estimation.start_params_and_bounds.param_lists import (
    men_disability_old_age_params,
    # men_disutil_firing,
    men_SRA_firing,
    men_disutil_params,
    men_disability_params,
    men_disutil_params_edu,
    men_job_offer_old_age_params,
    men_job_offer_params,
    women_disutil_firing,
    women_disutil_params,
    women_job_offer_old_age_params,
women_disability_old_age_params,
    women_job_offer_params,
    women_disability_params
)
from set_paths import create_path_dict

paths_dict = create_path_dict(define_user=False)
from estimation.struct_estimation.scripts.estimate_setup import estimate_model
from estimation.struct_estimation.scripts.observed_model_fit import create_fit_plots
from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)

model_name = "last_old_women"
params_to_estimate_names = women_disutil_params
sex_type = "women"
edu_type = "all"
util_type = "add"

LOAD_LAST_ESTIMATE = False
LOAD_SOL_MODEL = True
SAVE_RESULTS = True
USE_WEIGHTS = False

print(f"Running estimation for model: {model_name}", flush=True)

if LOAD_LAST_ESTIMATE:
    last_estimate = pkl.load(
        open(paths_dict["struct_results"] + f"est_params_fire_old_done.pkl", "rb")
    )
else:
    last_estimate = None

# Load start params
start_params_all = load_and_set_start_params(paths_dict)
# last_estimate["disutil_children_ft_work_low"] = 0.1
# last_estimate["disutil_children_ft_work_high"] = 0.1

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
    old_only=True,
    print_men_examples=True,
    print_women_examples=True,
    slow_version=False,
    scale_opt=True,
    multistart=True,
)
print(estimation_results)

# # %% Set paths of project
# from specs.derive_specs import generate_derived_and_data_derived_specs

# specs = generate_derived_and_data_derived_specs(paths_dict)


# create_fit_plots(
#     path_dict=paths_dict,
#     specs=specs,
#     params=end_params,
#     model_name=model_name,
#     load_sol_model=True,
#     load_solution=None,
#     load_data_from_sol=False,
#     sex_type=sex_type,
#     edu_type=edu_type,
# )


# %%
# 