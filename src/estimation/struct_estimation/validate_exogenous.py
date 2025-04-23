# %% Set paths of project
import pickle

from model_code.specify_model import specify_model
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)

# Set run specs
model_name = "disability"
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


model, params = specify_model(
    path_dict=path_dict,
    params=params,
    subj_unc=True,
    custom_resolution_age=None,
    sim_alpha=None,
    annoucement_age=None,
    annoucement_SRA=None,
    load_model=load_sol_model,
    model_type="solution",
)

from dcegm.interface import validate_exogenous_processes

validate_exogenous_processes(
    model=model,
    params=params,
)
#
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
