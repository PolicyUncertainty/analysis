# %%
# Set paths of project
import os

import pandas as pd

from set_paths import create_path_dict
from simulation.sim_tools.calc_aggregate_results import add_overall_results
from simulation.sim_tools.cv import calc_compensated_variation
from simulation.sim_tools.simulate_exp import simulate_exp
from simulation.sim_tools.start_obs_for_sim import investigate_start_obs

path_dict = create_path_dict()

from specs.derive_specs import generate_derived_and_data_derived_specs

specs = generate_derived_and_data_derived_specs(path_dict)

# Import jax and set jax to work with 64bit
import jax

jax.config.update("jax_enable_x64", True)

import pickle as pkl

import numpy as np

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

initial_obs_table = investigate_start_obs(
    path_dict=path_dict, model_class=None, load=True
)

for sex_var, sex_label in enumerate(specs["sex_labels"]):
    for edu_var, edu_label in enumerate(specs["education_labels"]):
        fixed_states = {
            "period": 0,
            "sex": sex_var,
            "education": edu_var,
            "lagged_choice": 1,
            "health": 0,
            "assets_begin_of_period": initial_obs_table.loc[
                (sex_var, edu_var), "assets_begin_of_period_median"
            ],
            "experience": initial_obs_table.loc[
                (sex_var, edu_var), "experience_median"
            ],
            "partner_state": 1,
            "job_offer": 1,
            "policy_state": 8,  # SRA = 67
        }

        # Initialize empty DataFrame
        res_df = pd.DataFrame()

        df_base = None
        for subj_unc in [False, True]:
            model_solution = None
            for informed_label in ["Informed", "Uninformed"]:
                informed = int(informed_label == "Informed")
                col_prefix = f"{informed_label}_unc_{subj_unc}"
                print(
                    "Eval expectation: ",
                    subj_unc,
                    informed_label,
                    flush=True,
                )
                state = {
                    **fixed_states,
                    "informed": informed,
                }

                df, model_solution = simulate_exp(
                    initial_state=state,
                    n_multiply=1_000,
                    path_dict=path_dict,
                    params=params,
                    subj_unc=subj_unc,
                    custom_resolution_age=None,
                    model_name=model_name,
                    solution_exists=load_unc_solution,
                    sol_model_exists=load_model,
                    model_solution=model_solution,
                    util_type=util_type,
                )
                if (informed_label == "Informed") and (not subj_unc):
                    df_base = df.copy()
                    res_df.loc[col_prefix, "_cv"] = 0.0
                else:
                    res_df.loc[col_prefix, "_cv"] = calc_compensated_variation(
                        df_base=df_base,
                        df_cf=df,
                        params=params,
                        specs=specs,
                    )

                # Use col_prefix as index, empty string as pre_name
                res_df = add_overall_results(
                    result_df=res_df,
                    df_scenario=df,
                    index=col_prefix,
                    pre_name="",
                    specs=specs,
                )
                if subj_unc:
                    res_df.loc[col_prefix, "_sra_at_63"] = 68.23

        file_append = sex_label + edu_append[edu_var]

        sim_results_folder = path_dict["sim_results"] + model_name + "/"
        os.makedirs(sim_results_folder, exist_ok=True)

        res_df.to_csv(sim_results_folder + f"baseline_margins_{file_append}.csv")
