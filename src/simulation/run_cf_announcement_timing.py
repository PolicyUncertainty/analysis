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
from simulation.sim_tools.calc_life_time_results import add_new_life_cycle_results
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario

# %%
# Set specifications
seeed = 123
model_name = "disability"
load_solution = True  # baseline solution conntainer
load_sol_model = True  # informed state as types
load_sim_model = True  # informed state stochastic
load_df = (
    None  # True = load existing df, False = create new df, None = create but not save
)
load_df_base = load_df

# Load params
params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

# Create life cylce df object which is None.
# The result function will then initialize in the first iteration.
res_df_life_cycle = None

# initiaize announcement ages
annoucement_ages = [35, 45, 55]
for announcement_age in annoucement_ages:
    if load_df:
        print("Load existing dfs for announcement age: ", announcement_age)
    else:
        print("Start simulation for announcement age: ", announcement_age)

    # Simulate baseline with subjective belief
    df_base = solve_and_simulate_scenario(
        path_dict=path_dict,
        params=params,
        subj_unc=True,
        custom_resolution_age=None,
        annoucement_age=63,  # earliest age is 31
        SRA_at_retirement=69,
        SRA_at_start=67,
        model_name=model_name,
        df_exists=load_df_base,
        solution_exists=load_solution,
        sol_model_exists=load_sol_model,
        sim_model_exists=load_sim_model,
    ).reset_index()

    if load_df_base is not None:
        load_df_base = True

    load_sol_model = True
    load_sim_model = True
    load_solution = True

    # Simulate counterfactual
    df_cf = solve_and_simulate_scenario(
        path_dict=path_dict,
        params=params,
        subj_unc=True,
        custom_resolution_age=None,
        annoucement_age=announcement_age,  # Let's earliest age is 31
        SRA_at_retirement=69,
        SRA_at_start=67,
        model_name=model_name,
        df_exists=load_df,
        solution_exists=load_solution,
        sol_model_exists=load_sol_model,
        sim_model_exists=load_sim_model,
    ).reset_index()

    res_df_life_cycle = add_new_life_cycle_results(
        df_base=df_base,
        df_cf=df_cf,
        scenatio_indicator=announcement_age,
        res_df_life_cycle=res_df_life_cycle,
    )


res_df_life_cycle.to_csv(path_dict["sim_results"] + f"announcement_lc_{model_name}.csv")
