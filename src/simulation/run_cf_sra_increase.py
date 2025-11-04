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

from simulation.sim_tools.calc_aggregate_results import add_overall_results
from simulation.sim_tools.cv import calc_compensated_variation
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario


def process_gender_results(i, df, result_dfs, het_mask_dict, df_base_cv, params=None):
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

    het_names = list(het_mask_dict.keys()) + ["overall"]

    for het_name in het_names:
        if het_name != "overall":
            mask = het_mask_dict[het_name](df)
            df_scenario = df[mask].copy()
            print(f"  {het_name}: {len(df_scenario)} individuals", flush=True)
        else:
            df_scenario = df

        add_overall_results(
            result_df=result_dfs[het_name],
            index=i,
            pre_name="cf",
            df_scenario=df_scenario,
            specs=specs,
        )
        if i == 0:
            all_idxs = result_dfs[het_name].index
            for all_i in all_idxs:
                add_overall_results(
                    result_df=result_dfs[het_name],
                    index=all_i,
                    pre_name="base",
                    df_scenario=df_scenario,
                    specs=specs,
                )

        if df_base_cv is not None:
            if het_name != "overall":
                mask_base = het_mask_dict[het_name](df_base_cv)
                df_base_het = df_base_cv[mask_base].copy()
            else:
                df_base_het = df_base_cv.copy()

            cv = calc_compensated_variation(
                df_base=df_base_het,
                df_cf=df_scenario,
                specs=specs,
                params=params,
            )
            result_dfs[het_name].loc[i, "cv"] = cv
        else:
            result_dfs[het_name].loc[i, "cv"] = 0.0


# Save all results
def save_results(result_dfs, path_dict, model_name):
    """Save all result dataframes to CSV files"""
    save_folder = path_dict["sim_results"] + model_name + "/"
    os.makedirs(save_folder, exist_ok=True)
    for scenario in result_dfs:
        for het_name in result_dfs[scenario]:
            filename = f"sra_increase_aggregate_{scenario}_{het_name}.csv"
            result_dfs[scenario][het_name].to_csv(save_folder + filename)
            print(f"Saved: {filename}")


# Initialize result dataframes and baseline storage
def create_result_dfs(sra_at_63, scenarios, het_mask_dict):
    """Create result dataframes for all scenarios and gender categories"""
    result_dfs = {}
    for scenario in scenarios:
        result_dfs[scenario] = {}
        heterogeneities = list(het_mask_dict.keys()) + ["overall"]
        for het in heterogeneities:
            df = pd.DataFrame(dtype=float)
            df["sra_at_63"] = sra_at_63
            result_dfs[scenario][het] = df
    return result_dfs


# %%
# Set specifications
seeed = 123
model_name = specs["model_name"]
util_type = specs["util_type"]
load_unc_model = True
load_no_unc_model = True
load_solutions = None
load_df = None

het_mask_dict = {
    "men": lambda df: df["sex"] == 0,
    "women": lambda df: df["sex"] == 1,
    "low_edu": lambda df: df["education"] == 0,
    "high_edu": lambda df: df["education"] == 1,
    "low_men": lambda df: (df["sex"] == 0) & (df["education"] == 0),
    "high_men": lambda df: (df["sex"] == 0) & (df["education"] == 1),
    "low_women": lambda df: (df["sex"] == 1) & (df["education"] == 0),
    "high_women": lambda df: (df["sex"] == 1) & (df["education"] == 1),
    "initial_informed": lambda df: df["initial_informed"] == "Informed",
    "initial_uninformed": lambda df: df["initial_informed"] == "Uninformed",
}
# For welfare it is important that no_unc comes first
scenarios = ["no_unc", "unc"]

# Load params
params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)


# Initialize alpha values and replace 0.04 with subjective alpha
sra_at_63 = [67, 68, 69, 70]

result_dfs = create_result_dfs(
    sra_at_63=sra_at_63, scenarios=scenarios, het_mask_dict=het_mask_dict
)

# Initialize baseline cv overall
for scenario_label in scenarios:
    df_base_cv = None
    if scenario_label == "unc":
        subj_unc = True
        load_sol_model = load_unc_model
    else:
        subj_unc = False
        load_sol_model = load_no_unc_model

    model_sol = None
    df_base = None

    for i, sra in enumerate(sra_at_63):
        print("Start simulation for sra: ", sra, flush=True)

        # ========== UNCERTAINTY SCENARIO ==========
        df, model_sol = solve_and_simulate_scenario(
            path_dict=path_dict,
            params=params,
            subj_unc=subj_unc,
            custom_resolution_age=None,
            announcement_age=None,
            only_informed=False,
            SRA_at_retirement=sra,
            SRA_at_start=67,
            model_name=model_name,
            df_exists=load_df,
            solution_exists=load_solutions,
            sol_model_exists=load_sol_model,
            model_solution=model_sol,
            util_type=util_type,
        )

        df = df.reset_index()
        process_gender_results(
            i=i,
            df=df,
            result_dfs=result_dfs[scenario_label],
            het_mask_dict=het_mask_dict,
            df_base_cv=df_base_cv,
            params=params,
        )
        if i == 0:
            df_base_cv = df.copy()
        else:
            del df

save_results(result_dfs, path_dict, model_name)
