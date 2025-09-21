# Set paths of project
import pickle as pkl
import sys

from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

paths_dict = create_path_dict(define_user=False)
import pandas as pd

from estimation.struct_estimation.scripts.estimate_setup import estimate_model
from estimation.struct_estimation.scripts.initialize_ll_function_only import (
    initialize_est_class,
)
from estimation.struct_estimation.scripts.observed_model_fit import create_fit_plots
from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)
from model_code.specify_model import specify_model
from model_code.transform_data_from_model import load_scale_and_correct_data

LOAD_SOL_MODEL = True
OLD_ONLY = True

# Load start params
start_params_all = load_and_set_start_params(paths_dict)
specs = generate_derived_and_data_derived_specs(paths_dict)

# test_name = "test_1"
# data_decision = pd.read_csv(test_name + ".csv")
# mask_m = data_decision["sex"] == 0
# ll_list = ["ll_0.1", "ll_0.2", "ll_0.3", "ll_0.4"]
# breakpoint()
param_to_loop = "disutil_ft_work_good_men"

params_to_estimate_names = [param_to_loop]


est_class = initialize_est_class(
    paths_dict,
    params_to_estimate_names=params_to_estimate_names,
    file_append=test_name,
    load_model=LOAD_SOL_MODEL,
    start_params_all=start_params_all,
    use_weights=False,
    save_results=True,
    print_men_examples=True,
    print_women_examples=True,
    use_observed_data=True,
    sim_data=None,
    old_only=OLD_ONLY,
)

model = specify_model(
    path_dict=paths_dict,
    specs=specs,
    subj_unc=True,
    custom_resolution_age=None,
    load_model=LOAD_SOL_MODEL,
    sim_specs=None,
)
data_decision = load_scale_and_correct_data(path_dict=paths_dict, model_class=model)
if OLD_ONLY:
    data_decision = data_decision[data_decision["age"] >= 55]

data_decision = data_decision[data_decision["lagged_choice"] != 0]

for disutil in [0.1, 0.2, 0.3, 0.4]:
    data_decision[f"ll_{disutil}"] = est_class.ll_func(
        {param_to_loop: disutil},
    )

data_decision.to_csv(test_name + ".csv")


# %%
