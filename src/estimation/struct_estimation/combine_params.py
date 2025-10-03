import pickle as pkl

from estimation.struct_estimation.map_params_to_current import merge_params
from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)
from set_paths import create_path_dict

path_dict = create_path_dict(define_user=False)

params_dict = {}
params_dict["default"] = load_and_set_start_params(path_dict)
params_dict["women"] = {}
params_dict["men"] = {}
# Load start params
params_dict["women"]["params"] = pkl.load(
    open(path_dict["struct_results"] + f"est_params_very_old_women_2.pkl", "rb")
)
params_dict["women"]["names"] = [
    key for key in params_dict["default"].keys() if "_women" in key or "children" in key
]

params_dict["men"]["params"] = pkl.load(
    open(path_dict["struct_results"] + f"est_params_very_old_men_1.pkl", "rb")
)
params_dict["men"]["names"] = [
    key for key in params_dict["default"].keys() if "_men" in key
]
params = merge_params(params_dict)
pkl.dump(
    params,
    open(path_dict["struct_results"] + f"est_params_merge_final.pkl", "wb"),
)
