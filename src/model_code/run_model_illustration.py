# %% Set paths of project
import pickle

import matplotlib.pyplot as plt
import numpy as np

from model_code.plots.plot_law_of_motion import plot_ret_experience
from model_code.plots.retirement_probs_illustration import (
    plot_ret_probs_for_asset_level,
)
from model_code.specify_model import specify_and_solve_model
from model_code.specify_simple_model import specify_and_solve_simple_model
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)

# Set run specs
model_name = "ucl"
load_model = True
load_solution = True

params = pickle.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)


which_plots = "l"
plot_ret_experience(specs)

model_solved = specify_and_solve_simple_model(
    path_dict=path_dict,
    file_append=model_name,
    params=params,
    load_model=load_model,
    load_solution=load_solution,
)

if which_plots == "r":
    plot_ret_probs_for_asset_level(model_solved=model_solved, specs=specs, assets=9)
