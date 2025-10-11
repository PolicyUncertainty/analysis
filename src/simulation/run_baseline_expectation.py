# %%
# Set paths of project
import os

import pandas as pd
from pandas.io.sql import table_exists

from set_paths import create_path_dict
from simulation.sim_tools.calc_aggregate_results import add_overall_results
from simulation.sim_tools.cv import calc_compensated_variation
from simulation.sim_tools.simulate_exp import simulate_exp
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario
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

n_multiply = 1_000
# Be aware: Starting at 67 is hardcoded everywhere

# Load params
params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)
edu_append = ["low", "high"]

sim_results_folder = path_dict["sim_results"] + model_name + "/"
os.makedirs(sim_results_folder, exist_ok=True)


def specify_state(sex, education):
    initial_obs_table = investigate_start_obs(
        path_dict=path_dict, model_class=None, load=True
    )

    states = {
        "period": 0,
        "sex": sex_var,
        "education": edu_var,
        "lagged_choice": 1,
        "health": 0,
        "assets_begin_of_period": initial_obs_table.loc[
            (sex_var, edu_var), "assets_begin_of_period_median"
        ],
        "experience": initial_obs_table.loc[(sex_var, edu_var), "experience_median"],
        "partner_state": 1,
        "job_offer": 1,
        "policy_state": 8,  # SRA = 67
    }
    return states


expected_SRA = 67 + 33 * specs["sra_belief_alpha"]

for sex_var, sex_label in enumerate(specs["sex_labels"]):
    for edu_var, edu_label in enumerate(specs["education_labels"]):
        file_append = sex_label + edu_append[edu_var]

        fixed_states = specify_state(sex=sex_var, education=edu_var)

        # Initialize empty DataFrame
        res_df_ex_ante = pd.DataFrame()

        df_base = None
        # The order of this loop matters for the cv calculation, but also
        # for re-using the solution in the ex post part below
        for subj_unc in [False, True]:
            model_solution = None
            for informed_label in ["Informed", "Uninformed"]:
                informed = int(informed_label == "Informed")
                table_col_prefix = f"{informed_label}_unc_{subj_unc}"
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
                    n_multiply=n_multiply,
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
                    res_df_ex_ante.loc[table_col_prefix, "_cv"] = 0.0
                else:
                    res_df_ex_ante.loc[table_col_prefix, "_cv"] = (
                        calc_compensated_variation(
                            df_base=df_base,
                            df_cf=df,
                            params=params,
                            specs=specs,
                        )
                    )

                # Use col_prefix as index, empty string as pre_name
                res_df_ex_ante = add_overall_results(
                    result_df=res_df_ex_ante,
                    df_scenario=df,
                    index=table_col_prefix,
                    pre_name="",
                    specs=specs,
                )
                if subj_unc:
                    res_df_ex_ante.loc[table_col_prefix, "_sra_at_63"] = expected_SRA

        res_df_ex_ante.to_csv(
            sim_results_folder + f"ex_ante_expected_margins_{file_append}.csv"
        )
        print("Wrote ex ante results for: ", file_append, flush=True)

        res_df_ex_post = pd.DataFrame()
        for reform_scenario in ["no_reform", "reform"]:
            for initial_informed, initial_informed_label in enumerate(
                [
                    "initial_uninf",
                    "initial_inf",
                ]
            ):
                if reform_scenario == "no_reform":
                    SRA_at_63 = 67.0
                else:
                    SRA_at_63 = expected_SRA

                table_column_prefix = f"{initial_informed_label}_{reform_scenario}"
                print("Eval ex post: ", table_column_prefix, flush=True)
                state = {
                    **fixed_states,
                    "informed": initial_informed,
                }
                initial_states = {
                    key: np.ones(n_multiply) * value for key, value in state.items()
                }

                df, model_solution = solve_and_simulate_scenario(
                    path_dict=path_dict,
                    params=params,
                    subj_unc=True,
                    custom_resolution_age=None,
                    SRA_at_start=67,
                    SRA_at_retirement=SRA_at_63,
                    announcement_age=None,
                    model_name=model_name,
                    initial_states=initial_states,
                    df_exists=False,
                    only_informed=False,
                    solution_exists=True,
                    sol_model_exists=True,
                    model_solution=model_solution,
                    util_type="add",
                )

                res_df_ex_post.loc[table_column_prefix, "_cv"] = (
                    calc_compensated_variation(
                        df_base=df_base,
                        df_cf=df,
                        params=params,
                        specs=specs,
                    )
                )

                # Use col_prefix as index, empty string as pre_name
                res_df_ex_post = add_overall_results(
                    result_df=res_df_ex_post,
                    df_scenario=df,
                    index=table_column_prefix,
                    pre_name="",
                    specs=specs,
                )

        res_df_ex_post.to_csv(
            sim_results_folder + f"ex_post_realized_margins_{file_append}.csv"
        )
