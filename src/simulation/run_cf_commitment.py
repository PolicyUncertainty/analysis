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
from simulation.sim_tools.calc_aggregate_results import (
    below_sixty_savings,
    calc_average_retirement_age,
    sra_at_retirement,
)
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario

# %%
# Set specifications
n_agents = 10000
seeed = 123
model_name = "partner_est"
load_base_solution = True  # baseline solution conntainer
load_second_solution = True  # counterfactual solution conntainer
load_sol_model = True  # informed state as type
load_sim_model = True  # informed state stochastic
load_df = (
    True  # True = load existing df, False = create new df, None = create but not save
)


# Load params
params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

# Initialize alpha values and replace 0.04 with subjective alpha(0.041...)
# sra_at_63 = np.arange(67, 70 + specs["SRA_grid_size"], specs["SRA_grid_size"])
# sra_at_63 = np.arange(67, 70 + specs["SRA_grid_size"], 1)
sra_at_63 = [67.0, 68.0, 69.0, 70.0]

# Create result dfs: one for aggregates and one for life-cycle profiles
res_df = pd.DataFrame(
    columns=[
        "alpha",
        "below_sixty_savings",
        "sra_at_ret",
        "ret_age",
        "working_hours",
        "cv",
    ],
    dtype=float,
)


res_df_life_cycle = pd.DataFrame(dtype=float)

# Assign alphas in dataframe
res_df["sra_at_63"] = sra_at_63
for i, sra in enumerate(sra_at_63):
    if load_df:
        print("Load existing dfs for sra: ", sra)
    else:
        print("Start simulation for sra: ", sra)

    # Simulate baseline with subjective belief
    df_base = solve_and_simulate_scenario(
        path_dict=path_dict,
        params=params,
        subj_unc=True,
        custom_resolution_age=None,
        annoucement_age=None,
        SRA_at_retirement=sra,
        SRA_at_start=67,
        model_name=model_name,
        df_exists=load_df,
        solution_exists=load_base_solution,
        sol_model_exists=load_sol_model,
        sim_model_exists=load_sim_model,
    ).reset_index()

    load_sim_model = True
    load_base_solution = True
    load_sol_model = True

    # Simulate counterfactual with no uncertainty and expected increase
    # same as simulated alpha_sim
    df_cf = solve_and_simulate_scenario(
        path_dict=path_dict,
        params=params,
        subj_unc=False,
        custom_resolution_age=None,
        annoucement_age=None,
        SRA_at_retirement=sra,
        SRA_at_start=sra,
        model_name=model_name,
        df_exists=load_df,
        solution_exists=load_second_solution,
        sol_model_exists=load_sol_model,
        sim_model_exists=load_sim_model,
    ).reset_index()

    # load_second_solution = True
    # Calculate results
    for k, df_scen in enumerate([df_cf, df_base]):
        if k == 0:
            pre = ""
        else:
            pre = "base_"

        res_df.loc[i, pre + "below_sixty_savings"] = below_sixty_savings(df_scen)
        res_df.loc[i, pre + "ret_age"] = calc_average_retirement_age(df_scen)
        res_df.loc[i, pre + "sra_at_ret"] = sra_at_retirement(df_scen)
        res_df.loc[i, pre + "working_hours"] = df_scen["working_hours"].mean()

    ages = np.arange(30, 101, 1)
    for k, df in enumerate([df_base, df_cf]):

        if k == 0:
            string = "base"
        else:
            string = "cf"

        res_df_life_cycle[f"working_hours_{sra}_{string}"] = df.groupby("age")[
            "working_hours"
        ].aggregate("mean")
        res_df_life_cycle[f"savings_{sra}_{string}"] = df.groupby("age")[
            "savings_dec"
        ].aggregate("mean")
        res_df_life_cycle[f"consumption_{sra}_{string}"] = df.groupby("age")[
            "consumption"
        ].aggregate("mean")
        res_df_life_cycle[f"income_{sra}_{string}"] = df.groupby("age")[
            "total_income"
        ].aggregate("mean")
        res_df_life_cycle[f"assets_{sra}_{string}"] = df.groupby("age")[
            "savings"
        ].aggregate("mean")
        res_df_life_cycle[f"savings_rate_{sra}_{string}"] = (
            res_df_life_cycle[f"savings_{sra}_{string}"]
            / res_df_life_cycle[f"income_{sra}_{string}"]
        )
        res_df_life_cycle[f"employment_rate_{sra}_{string}"] = df.groupby("age")[
            "choice"
        ].apply(lambda x: x.isin([2, 3]).sum() / len(x))
        res_df_life_cycle[f"retirement_rate_{sra}_{string}"] = df.groupby("age")[
            "choice"
        ].apply(lambda x: x.isin([0]).sum() / len(x))
    res_df_life_cycle[f"savings_rate_diff_{sra}"] = (
        res_df_life_cycle[f"savings_rate_{sra}_cf"]
        - res_df_life_cycle[f"savings_rate_{sra}_base"]
    )
    res_df_life_cycle[f"employment_rate_diff_{sra}"] = (
        res_df_life_cycle[f"employment_rate_{sra}_cf"]
        - res_df_life_cycle[f"employment_rate_{sra}_base"]
    )
    res_df_life_cycle[f"retirement_rate_diff_{sra}"] = (
        res_df_life_cycle[f"retirement_rate_{sra}_cf"]
        - res_df_life_cycle[f"retirement_rate_{sra}_base"]
    )
