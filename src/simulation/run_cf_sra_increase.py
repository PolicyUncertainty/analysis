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

from simulation.sim_tools.calc_aggregate_results import calc_overall_results
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario
from simulation.tables.cv import calc_compensated_variation


# %%
def write_results_to_dataframe(i, df_base, df_cf, result_df, params, specs):
    """
    Helper function to calculate and write results to result dataframe

    Args:
        i: iteration index
        df_base: baseline dataframe
        df_cf: counterfactual dataframe
        result_df: result dataframe to write to
        params: model parameters
        specs: model specifications
    """
    results_row = calc_overall_results(df_base=df_base, df_cf=df_cf)
    for key, value in results_row.items():
        result_df.loc[i, key] = value
    # result_df.loc[i, "cv"] = calc_compensated_variation(
    #     df_base=df_base,
    #     df_cf=df_cf,
    #     params=params,
    #     specs=specs,
    # )


def process_gender_results(
    i, df, result_dfs, df_base, params, specs
):
    """
    Process results for all gender categories (men, women, overall)

    Args:
        i: iteration index
        df: full dataframe from simulation
        result_dfs: dict of result dataframes {"men": df_men, "women": df_women, "overall": df_overall}
        df_bases: dict of baseline dataframes
        params: model parameters
        specs: model specifications
        scenario_name: name for debugging/logging
    """

    # Calculate and write results for each gender category
    write_results_to_dataframe(
        i, df_base[], df_men, result_dfs["men"], params, specs
    )
    write_results_to_dataframe(
        i, df_bases["women"], df_women, result_dfs["women"], params, specs
    )
    write_results_to_dataframe(
        i, df_bases["overall"], df_overall, result_dfs["overall"], params, specs
    )


# Save all results
def save_results(result_dfs, path_dict, model_name):
    """Save all result dataframes to CSV files"""
    for scenario in result_dfs:
        for gender in result_dfs[scenario]:
            filename = f"sra_increase_aggregate_{scenario}_{gender}_{model_name}.csv"
            result_dfs[scenario][gender].to_csv(path_dict["sim_results"] + filename)
            print(f"Saved: {filename}")


# Initialize result dataframes and baseline storage
def create_result_dfs(sra_at_63):
    """Create result dataframes for all scenarios and gender categories"""
    result_dfs = {}
    for scenario in ["unc", "no_unc"]:  # , "debias"]:
        result_dfs[scenario] = {}
        for gender in ["men", "women", "overall"]:
            df = pd.DataFrame(dtype=float)
            df["sra_at_63"] = sra_at_63
            result_dfs[scenario][gender] = df
    return result_dfs


# %%
# Set specifications
seeed = 123
model_name = specs["model_name"]
load_sol_model = True
load_solutions = False
load_df = None

# Load params
params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)


# Initialize alpha values and replace 0.04 with subjective alpha
sra_at_63 = np.arange(67, 70 + 1, 1)

result_dfs = create_result_dfs(sra_at_63)

# Initialize baseline storage

for unc_scenario in [True, False]:
    df_bases = {}
    if unc_scenario:
        df_label = "unc"
    else:
        df_label = "no_unc"

    model_sol = None

    for i, sra in enumerate(sra_at_63):
        print("Start simulation for sra: ", sra)
        if unc_scenario:
            SRA_at_start = 67
        else:
            SRA_at_start = sra

        # ========== UNCERTAINTY SCENARIO ==========
        df, model_sol = solve_and_simulate_scenario(
            path_dict=path_dict,
            params=params,
            subj_unc=unc_scenario,
            custom_resolution_age=None,
            announcement_age=None,
            SRA_at_retirement=sra,
            SRA_at_start=SRA_at_start,
            model_name=model_name,
            df_exists=load_df,
            solution_exists=load_solutions,
            sol_model_exists=load_sol_model,
            model_solution=model_sol,
        )

        df = df.reset_index()
        process_gender_results(
            i, df, result_dfs[df_label], df_bases, params, specs
        )

    # # ========== DEBIAS SCENARIO (COMMENTED OUT) ==========
    # df_debias, _ = solve_and_simulate_scenario(
    #     path_dict=path_dict,
    #     params=params,
    #     subj_unc=False,
    #     custom_resolution_age=None,
    #     announcement_age=None,
    #     SRA_at_retirement=sra,
    #     SRA_at_start=sra,
    #     only_informed=True,
    #     model_name=model_name,
    #     df_exists=load_df,
    #     solution_exists=load_no_unc_solution,
    #     sol_model_exists=load_sol_model,
    # )

    # df_debias = df_debias.reset_index()
    # process_gender_results(i, df_debias, result_dfs["debias"], df_bases_debias, params, specs, "debias")

save_results(result_dfs, path_dict, model_name)
