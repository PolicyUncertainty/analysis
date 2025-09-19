# Set paths of project
import pickle as pkl
import sys

from estimation.struct_estimation.map_params_to_current import merge_params
from estimation.struct_estimation.start_params_and_bounds.param_lists import (
    high_men_disutil_firing,
    high_women_disutil_firing,
    low_men_disutil_firing,
    low_women_disutil_firing,
)
from set_paths import create_path_dict

paths_dict = create_path_dict(define_user=False)
from estimation.struct_estimation.scripts.estimate_setup import (
    estimate_model,
    generate_print_func,
)
from estimation.struct_estimation.scripts.observed_model_fit import create_fit_plots
from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)

start_params_all = load_and_set_start_params(paths_dict)

param_lists = {
    "men": {"low": low_men_disutil_firing, "high": high_men_disutil_firing},
    "women": {"low": low_women_disutil_firing, "high": high_women_disutil_firing},
}

model_name = "low_women_3_cobb"
params_dict = {}
params_dict["default"] = start_params_all
for edu_type in ["low", "high"]:
    for s_type in ["men", "women"]:
        name = f"{edu_type}_{s_type}"
        params_dict[name] = {}
        params_dict[name]["names"] = param_lists[s_type][edu_type]
        params = pkl.load(
            open(
                paths_dict["struct_results"]
                + f"est_params_{edu_type}_{s_type}_3_cobb.pkl",
                "rb",
            )
        )
        params_dict[name]["params"] = params


params = merge_params(params_dict)
from specs.derive_specs import generate_derived_and_data_derived_specs

specs = generate_derived_and_data_derived_specs(paths_dict)
generate_print_func(params.keys(), specs)(params)
pkl.dump(
    params,
    open(
        paths_dict["struct_results"] + f"est_params_3_cobb.pkl",
        "wb",
    ),
)
