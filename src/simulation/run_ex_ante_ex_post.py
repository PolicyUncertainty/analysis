# %%
# Set paths of project
import os

import pandas as pd
from pandas.io.sql import table_exists

from set_paths import create_path_dict
from simulation.sim_tools.calc_aggregate_results import add_overall_results
from simulation.sim_tools.cv import calc_compensated_variation
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario
from simulation.sim_tools.start_obs_for_sim import generate_start_states_from_obs

path_dict = create_path_dict()

from specs.derive_specs import generate_derived_and_data_derived_specs

specs = generate_derived_and_data_derived_specs(path_dict)

# Import jax and set jax to work with 64bit
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import pickle as pkl

import numpy as np

# %%
# Set specifications
seed = 123
model_name = specs["model_name"]
util_type = specs["util_type"]

load_unc_model = True
load_unc_solution = None  # baseline solution conntainer

# Be aware: Starting at 67 is hardcoded everywhere
initial_SRA = 67

# Load params
params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

# Initialize empty DataFrame
res_df = pd.DataFrame()

expected_SRA = initial_SRA + 33 * specs["sra_belief_alpha"]

df_base = None
model_solution = None
start_states = None
for sample_split in ["full", "informed", "uninformed"]:
    if sample_split != "full":
        start_states = generate_start_states_from_obs(
            path_dict=path_dict,
            specs=specs,
            sample_split=sample_split,
            seed=seed,
            model_class=model_solution,
        )
        informed = int(sample_split == "informed")
        start_states["informed"] = jnp.ones_like(start_states["informed"]) * informed
    for seed in [234, 456]:
        # The order of this loop matters for the cv calculation
        for reform_scenario in ["no_reform", "reform"]:
            if reform_scenario == "no_reform":
                SRA_at_63 = 67.0
            else:
                SRA_at_63 = expected_SRA

            ex_post_table_column_prefix = f"{reform_scenario}_unc_true"

            df_post, model_solution = solve_and_simulate_scenario(
                path_dict=path_dict,
                params=params,
                subj_unc=True,
                custom_resolution_age=None,
                SRA_at_start=67,
                SRA_at_retirement=SRA_at_63,
                announcement_age=None,
                model_name=model_name,
                initial_states=start_states,
                df_exists=None,
                only_informed=None,  # Initial states important only
                solution_exists=load_unc_solution,
                sol_model_exists=load_unc_model,
                model_solution=model_solution,
                util_type=util_type,
            )
            # df_post.to_csv(
            #     f"df_post_{reform_scenario}_{sample_split}.csv"
            # )
            if reform_scenario == "no_reform":
                df_base = df_post.copy()
                cv = 0.0
            else:
                cv = calc_compensated_variation(
                    df_base=df_base,
                    df_cf=df_post,
                    params=params,
                    specs=specs,
                )
            res_df.loc[f"{sample_split}_ex_post", "cv"] = cv

        df_ante, _ = solve_and_simulate_scenario(
            path_dict=path_dict,
            params=params,
            subj_unc=True,
            custom_resolution_age=None,
            SRA_at_start=None,  # Not needed as we simulate expectations with initial states
            SRA_at_retirement=None,  # Not needed as we simulate expectations
            announcement_age=None,
            model_name=model_name,
            simulate_expectations=True,
            initial_states=start_states,
            df_exists=None,
            only_informed=None,  # Initial states important only
            solution_exists=load_unc_solution,
            sol_model_exists=load_unc_model,
            model_solution=model_solution,
            util_type=util_type,
        )
        # df_ante.to_csv(
        #     f"df_ante_{sample_split}.csv"
        # )

        cv_ex_ante = calc_compensated_variation(
            df_base=df_base,
            df_cf=df_ante,
            params=params,
            specs=specs,
        )
        res_df.loc[f"{sample_split}_ex_ante", "cv"] = cv_ex_ante

        print(res_df, flush=True)

sim_results_folder = path_dict["sim_results"] + model_name + "/"
os.makedirs(sim_results_folder, exist_ok=True)
res_df.to_csv(sim_results_folder + f"ex_post_ex_ante.csv")
print("Ex-post and ex-ante simulation with realized SRA margins done.")
