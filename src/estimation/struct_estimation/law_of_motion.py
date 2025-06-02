# %% Set paths of project
import pickle

import numpy as np

from estimation.struct_estimation.map_params_to_current import map_period_to_age
from model_code.specify_model import specify_model
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)

# Set run specs
model_name = "start"
load_sol_model = True

if model_name == "start":
    from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
        load_and_set_start_params,
    )

    params = load_and_set_start_params(path_dict)
else:
    params = pickle.load(
        open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
    )
    params = map_period_to_age(params)


model = specify_model(
    path_dict=path_dict,
    subj_unc=True,
    custom_resolution_age=None,
    load_model=load_sol_model,
    sim_specs=None,
)

# First validate all exogenous processes
model.validate_exogenous(
    params=params,
)

# Now the law of motions for the continuous states
continuous_grids_state_space = model.compute_law_of_motions(
    params=params,
)
assert (~np.isnan(continuous_grids_state_space["second_continuous"])).all()
assert (~np.isnan(continuous_grids_state_space["assets_begin_of_period"])).all()

# last_nan_idx = np.where(nan_grid)[0][-1]
# state_space_dict = model.model_structure["state_space_dict"]
# nan_state = {
#     key: state_space_dict[key][last_nan_idx] for key in state_space_dict.keys()
# }
# exp = model.model_config["continuous_states_info"]["second_continuous_grid"][5]
# from model_code.state_space.experience import get_next_period_experience
#
#
# exp_new = get_next_period_experience(
#     period=nan_state["period"],
#     lagged_choice=nan_state["lagged_choice"],
#     policy_state=nan_state["policy_state"],
#     sex=nan_state["sex"],
#     education=nan_state["education"],
#     experience=exp,
#     informed=nan_state["informed"],
#     health=nan_state["health"],
#     model_specs=model.model_specs,
# )
