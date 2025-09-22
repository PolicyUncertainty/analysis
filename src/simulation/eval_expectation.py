# %%
# Set paths of project
import pandas as pd

from set_paths import create_path_dict

path_dict = create_path_dict()

from specs.derive_specs import generate_derived_and_data_derived_specs

specs = generate_derived_and_data_derived_specs(path_dict)

import pickle as pkl

# Import jax and set jax to work with 64bit
import jax
import numpy as np

from simulation.sim_tools.calc_aggregate_results import (
    calc_average_retirement_age,
    expected_lifetime_income,
)
from simulation.sim_tools.simulate_exp import simulate_exp

# %%
# Set specifications
model_name = specs["model_name"]
load_sol_model = True  # informed state as type
load_unc_solution = True  # baseline solution conntainer

# Load params
params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

for subj_unc in [True, False]:
    for informed, informed_label in enumerate(["Uninformed", "Informed"]):
        initial_state = {
            "period": 0,
            "education": 0,
            "lagged_choice": 1,
            "sex": 0,
            "policy_state": 8,
            "health": 0,
            "informed": informed,
            "assets_begin_of_period": 5,
            "experience": 0.5,
            "partner_state": 0,
            "job_offer": 1,
        }

        df = simulate_exp(
            initial_state=initial_state,
            n_multiply=10_000,
            path_dict=path_dict,
            params=params,
            subj_unc=subj_unc,
            custom_resolution_age=None,
            model_name=model_name,
            solution_exists=load_unc_solution,
            sol_model_exists=load_sol_model,
        )
        avg_ret_age = calc_average_retirement_age(df)
        total_inc = expected_lifetime_income(df, specs)
        print(f"{informed_label} expected lifetime income: {total_inc}")
        print(f"{informed_label} expected average retirement age: {avg_ret_age}")
