# %% Set paths of project
import pickle

from estimation.struct_estimation.map_params_to_current import map_period_to_age
from estimation.struct_estimation.scripts.observed_model_fit import (
    load_and_prep_data_for_model_fit,
)
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
    debug_info="all",
)

data_decision, states_dict = load_and_prep_data_for_model_fit(
    paths_dict=path_dict, specs=specs, params=params, model_class=model
)


def select_state_by_id(id):
    state = {key: data_decision.loc[id, key] for key in states_dict.keys()}
    return state


states_47 = select_state_by_id(47)
states_47["informed"] = 1
states_47["job_offer"] = 1
child_states = model.get_child_states(state=states_47, choice=3)

# from model_code.stochastic_processes.health_transition import health_transition
#
# health_transition(
#     sex=0,
#     health=2,
#     education=0,
#     period=0,
#     params=params,
#     options=model["options"]["model_params"],
# )
