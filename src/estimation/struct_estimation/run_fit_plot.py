# %% Set paths of project
import pickle as pkl

from estimation.struct_estimation.scripts.observed_model_fit import create_fit_plots
from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)

# Set run specs
model_name = "25_women"
print(f"Running model: {model_name}")
load_sol_model = True
load_solution = None
load_data_from_sol = False

# Load start params
params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

create_fit_plots(
    path_dict=path_dict,
    specs=specs,
    params=params,
    model_name=model_name,
    load_sol_model=load_sol_model,
    load_solution=load_solution,
    load_data_from_sol=load_data_from_sol,
    sex_type="all",
    edu_type="all",
    util_type="add",
)


# %%
