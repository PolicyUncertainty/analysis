# %%
# Set paths of project
import os

import pandas as pd

from set_paths import create_path_dict

path_dict = create_path_dict()

from specs.derive_specs import generate_derived_and_data_derived_specs

specs = generate_derived_and_data_derived_specs(path_dict)

# Import jax and set jax to work with 64bit
import jax

jax.config.update("jax_enable_x64", True)

import pickle as pkl

import numpy as np

from simulation.sim_tools.calc_exp_results import calc_exp_results
from simulation.tables.baseline_expectation import generate_baseline_expectation_table

# %%
# Set specifications
seeed = 123
model_name = specs["model_name"]
util_type = specs["util_type"]

load_model = True  # informed state as type
load_unc_solution = None  # baseline solution conntainer


# Load params
params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)
edu_append = ["low", "high"]
for sex_var, sex_label in enumerate(specs["sex_labels"]):
    for edu_var, edu_label in enumerate(specs["education_labels"]):

        res_df = calc_exp_results(
            path_dict=path_dict,
            specs=specs,
            sex=sex_var,
            education=edu_var,
            params=params,
            model_name=model_name,
            load_solution=load_unc_solution,
            load_sol_model=load_model,
            util_type=util_type,
        )
        file_append = sex_label + edu_append[edu_var]

        sim_results_folder = path_dict["sim_results"] + model_name + "/"
        os.makedirs(sim_results_folder, exist_ok=True)

        res_df.to_csv(sim_results_folder + f"baseline_margins_{file_append}.csv")
