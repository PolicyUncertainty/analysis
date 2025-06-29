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
seeed = 123
model_name = specs["model_name"]
load_sol_model = True
load_unc_solution = False
load_no_unc_solution = False
load_df = None


# Load params
params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

# Initialize alpha values and replace 0.04 with subjective alpha
sra_at_63 = np.arange(67, 70 + specs["SRA_grid_size"], specs["SRA_grid_size"])

# Create result dfs and assign sra. Uncertainty
result_df_unc = pd.DataFrame(dtype=float)
result_df_unc["sra_at_63"] = sra_at_63

# No uncertainty
result_df_no_unc = pd.DataFrame(dtype=float)
result_df_no_unc["sra_at_63"] = sra_at_63

# Debias
result_df_debias = pd.DataFrame(dtype=float)
result_df_debias["sra_at_63"] = sra_at_63
for i, sra in enumerate(sra_at_63):
    print("Start simulation for sra: ", sra)

    # Create estimated model
    df_unc, _ = solve_and_simulate_scenario(
        path_dict=path_dict,
        params=params,
        subj_unc=True,
        custom_resolution_age=None,
        announcement_age=None,
        SRA_at_retirement=sra,
        SRA_at_start=67,
        model_name=model_name,
        df_exists=load_df,
        solution_exists=load_unc_solution,
        sol_model_exists=load_sol_model,
    )

    df_unc = df_unc.reset_index()
    # After the first run we can always set models and solutions to True
    load_sol_model = True
    load_unc_solution = True if load_unc_solution is not None else load_unc_solution

    if i == 0:
        df_base_unc = df_unc.copy()

    else:
        results_row = calc_overall_results(df_base=df_base_unc, df_cf=df_unc)

        for key, value in results_row.items():
            result_df_unc.loc[i, key] = value

        result_df_unc.loc[i, "cv"] = calc_compensated_variation(
            df_base=df_base_unc,
            df_cf=df_unc,
            params=params,
            specs=specs,
        )

    # Create estimated model
    df_no_unc, _ = solve_and_simulate_scenario(
        path_dict=path_dict,
        params=params,
        subj_unc=False,
        custom_resolution_age=None,
        announcement_age=None,
        SRA_at_retirement=sra,
        SRA_at_start=sra,
        model_name=model_name,
        df_exists=load_df,
        solution_exists=load_no_unc_solution,
        sol_model_exists=load_sol_model,
    )

    df_no_unc = df_no_unc.reset_index()

    # After the first run we can always set models and solutions to True
    load_sol_model = True
    load_no_unc_solution = (
        True if load_no_unc_solution is not None else load_no_unc_solution
    )

    if i == 0:
        df_base_no_unc = df_no_unc.copy()

    else:
        results_row = calc_overall_results(df_base=df_base_no_unc, df_cf=df_no_unc)

        for key, value in results_row.items():
            result_df_no_unc.loc[i, key] = value

        result_df_no_unc.loc[i, "cv"] = calc_compensated_variation(
            df_base=df_base_no_unc,
            df_cf=df_no_unc,
            params=params,
            specs=specs,
        )

    df_debias, _ = solve_and_simulate_scenario(
        path_dict=path_dict,
        params=params,
        subj_unc=False,
        custom_resolution_age=None,
        announcement_age=None,
        SRA_at_retirement=sra,
        SRA_at_start=sra,
        only_informed=True,
        model_name=model_name,
        df_exists=load_df,
        solution_exists=load_no_unc_solution,
        sol_model_exists=load_sol_model,
    )
    # After the first run we can always set models and solutions to True
    load_sol_model = True
    load_no_unc_solution = (
        True if load_no_unc_solution is not None else load_no_unc_solution
    )

    df_debias = df_debias.reset_index()

    if i == 0:
        df_base_debias = df_debias.copy()

    else:
        results_row = calc_overall_results(df_base=df_base_debias, df_cf=df_debias)

        for key, value in results_row.items():
            result_df_debias.loc[i, key] = value

        result_df_debias.loc[i, "cv"] = calc_compensated_variation(
            df_base=df_base_debias,
            df_cf=df_debias,
            params=params,
            specs=specs,
        )

# Save results
result_df_unc.to_csv(
    path_dict["sim_results"] + f"sra_increase_aggregate_unc_{model_name}.csv"
)
result_df_no_unc.to_csv(
    path_dict["sim_results"] + f"sra_increase_aggregate_no_unc_{model_name}.csv"
)
result_df_debias.to_csv(
    path_dict["sim_results"] + f"sra_increase_aggregate_debias_{model_name}.csv"
)
