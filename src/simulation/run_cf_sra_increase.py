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

from simulation.sim_tools.calc_aggregate_results import add_overall_results
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario


def process_gender_results(i, df, result_dfs, het_spec_vars, het_var_name):
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

    het_names = list(het_spec_vars.keys()) + ["overall"]

    for het_name in het_names:
        if het_name != "overall":
            mask = df[het_var_name].isin(het_spec_vars[het_name])
            df_scenario = df[mask]
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


# Save all results
def save_results(result_dfs, path_dict, model_name):
    """Save all result dataframes to CSV files"""
    for scenario in result_dfs:
        for het_name in result_dfs[scenario]:
            filename = f"sra_increase_aggregate_{scenario}_{het_name}_{model_name}.csv"
            result_dfs[scenario][het_name].to_csv(path_dict["sim_results"] + filename)
            print(f"Saved: {filename}")


# Initialize result dataframes and baseline storage
def create_result_dfs(sra_at_63, scenarios, het_spec_vars):
    """Create result dataframes for all scenarios and gender categories"""
    result_dfs = {}
    for scenario in scenarios:
        result_dfs[scenario] = {}
        heterogeneities = list(het_spec_vars.keys()) + ["overall"]
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
load_sol_model = True
load_solutions = None
load_df = None

het_var_name = "sex"
het_spec_vars = {"men": [0], "women": [1]}
scenarios = ["unc", "no_unc"]

# Load params
params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)


# Initialize alpha values and replace 0.04 with subjective alpha
sra_at_63 = np.arange(67, 70 + 1, 1)

result_dfs = create_result_dfs(
    sra_at_63=sra_at_63, scenarios=scenarios, het_spec_vars=het_spec_vars
)

# Initialize baseline storage

for scenario_label in scenarios:
    if scenario_label == "unc":
        subj_unc = True
    else:
        subj_unc = False

    model_sol = None

    for i, sra in enumerate(sra_at_63):
        print("Start simulation for sra: ", sra, flush=True)
        if subj_unc:
            SRA_at_start = 67
        else:
            SRA_at_start = sra

        # ========== UNCERTAINTY SCENARIO ==========
        df, model_sol = solve_and_simulate_scenario(
            path_dict=path_dict,
            params=params,
            subj_unc=subj_unc,
            custom_resolution_age=None,
            announcement_age=None,
            only_informed=False,
            SRA_at_retirement=sra,
            SRA_at_start=SRA_at_start,
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
            het_spec_vars=het_spec_vars,
            het_var_name=het_var_name,
        )
        del df
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
