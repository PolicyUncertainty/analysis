# %%
# Set paths of project
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

from export_results.tables.cv import calc_compensated_variation
from simulation.sim_tools.calc_aggregate_results import calc_overall_results
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario

# %%
# Set specifications
n_agents = 10000
seeed = 123
model_name = "partner_est"
load_solution = True
load_sol_model = True
load_sim_model = True
load_df = True


# Load params
params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

# Initialize alpha values and replace 0.04 with subjective alpha
sra_at_63 = np.arange(67, 70 + specs["SRA_grid_size"], specs["SRA_grid_size"])

# Create result df
result_df = pd.DataFrame(dtype=float)
# Assign sras
result_df["sra_at_63"] = sra_at_63
for i, sra in enumerate(sra_at_63):
    print("Start simulation for sra: ", sra)

    # Create estimated model
    df = solve_and_simulate_scenario(
        path_dict=path_dict,
        params=params,
        subj_unc=True,
        custom_resolution_age=None,
        annoucement_age=None,
        SRA_at_retirement=sra,
        SRA_at_start=67,
        model_name=model_name,
        df_exists=load_df,
        solution_exists=load_solution,
        sol_model_exists=load_sol_model,
        sim_model_exists=load_sim_model,
    )
    # After the first run we can always set models and solutions to True
    load_sol_model = True
    load_sim_model = True
    load_solution = True

    if i == 0:
        df_base = df.reset_index().copy()

    else:
        results_row = calc_overall_results(df_base=df_base, df_cf=df.reset_index())

        for key, value in results_row.items():
            result_df.loc[i, key] = value

        result_df.loc[i, "cv"] = calc_compensated_variation(
            df_base=df_base,
            df_cf=df.reset_index(),
            params=params,
            specs=specs,
        )

# Save results
result_df.to_csv(path_dict["sim_results"] + f"sra_increase_aggregate_{model_name}.csv")
