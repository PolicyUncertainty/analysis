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
model_name = "stable"
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

model = specify_model(
    path_dict=path_dict,
    specs=specs,
    subj_unc=True,
    custom_resolution_age=None,
    load_model=load_sol_model,
    sim_specs=None,
)

# # First validate all exogenous processes
# model.validate_exogenous(
#     params=params,
# )
#
# Now the law of motions for the continuous states
continuous_grids_state_space = model.compute_law_of_motions(
    params=params,
)
# assert (~np.isnan(continuous_grids_state_space["second_continuous"])).all()
# assert (~np.isnan(continuous_grids_state_space["assets_begin_of_period"])).all()

state_idx_to_inspect = 174
exp_id_last_period = 10
state_space_dict = model.model_structure["state_space_dict"]
state_to_inspect = {
    key: state_space_dict[key][state_idx_to_inspect] for key in state_space_dict.keys()
}
exp = model.model_config["continuous_states_info"]["second_continuous_grid"][
    exp_id_last_period
]
from model_code.state_space.experience import get_next_period_experience

exp_new = get_next_period_experience(
    period=state_to_inspect["period"],
    lagged_choice=state_to_inspect["lagged_choice"],
    policy_state=state_to_inspect["policy_state"],
    sex=state_to_inspect["sex"],
    education=state_to_inspect["education"],
    experience=exp,
    informed=state_to_inspect["informed"],
    health=state_to_inspect["health"],
    model_specs=model.model_specs,
)
