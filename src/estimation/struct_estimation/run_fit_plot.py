# %% Set paths of project
import pickle

import matplotlib.pyplot as plt

from estimation.struct_estimation.map_params_to_current import new_to_current
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)

# Set run specs
model_name = "wo_jo"
load_sol_model = True
load_solution = True

if model_name == "start":
    from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
        load_and_set_start_params,
    )

    params = load_and_set_start_params(path_dict)
else:
    params = pickle.load(
        open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
    )

from estimation.struct_estimation.scripts.observed_model_fit import observed_model_fit

observed_model_fit(
    paths_dict=path_dict,
    specs=specs,
    params=params,
    model_name=model_name,
    load_sol_model=load_sol_model,
    load_solution=load_solution,
)
plt.show()
plt.close("all")

# %%
