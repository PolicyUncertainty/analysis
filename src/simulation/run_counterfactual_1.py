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
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario
from simulation.sim_tools.calc_margin_results import (
    calc_average_retirement_age,
    sra_at_retirement,
    below_sixty_savings,
)

from export_results.tables.cv import calc_compensated_variation

# %%
# Set specifications
n_agents = 10000
seeed = 123
model_name = "new"
load_base_solution = True  # baseline solution conntainer
load_second_solution = True  # counterfactual solution conntainer
load_sol_model = True  # informed state as type
load_sim_model = True  # informed state stochastic
load_df = (
    False  # True = load existing df, False = create new df, None = create but not save
)


# Load params
params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

# Initialize alpha values and replace 0.04 with subjective alpha(0.041...)
sra_at_63 = np.arange(67, 70 + specs["SRA_grid_size"], specs["SRA_grid_size"])

# Create result df
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
# Assign alphas in dataframe
res_df["sra_at_63"] = sra_at_63
for i, sra in enumerate(sra_at_63):
    print("Start simulation for sra: ", sra)
    # Calculate how much it has to increase starting from 67 in beaseline
    alpha_sim = (sra - 67) / (63 - specs["start_age"])

    # Simulate baseline with subjective belief
    df_base = solve_and_simulate_scenario(
        path_dict=path_dict,
        params=params,
        sim_alpha=alpha_sim,
        expected_alpha=False,  # if False, then exp_alpha = alpha_hat and sigma = sigma_hat, else sigma = 0
        resolution=True,
        initial_SRA=67,
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
        sim_alpha=0.0,
        expected_alpha=0.0,
        resolution=True,
        initial_SRA=sra,
        model_name=model_name,
        df_exists=load_df,
        solution_exists=load_second_solution,
        sol_model_exists=load_sol_model,
        sim_model_exists=load_sim_model,
    ).reset_index()

    load_second_solution = True
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

    res_df.loc[i, "cv"] = calc_compensated_variation(
        df_base=df_base,
        df_cf=df_cf.reset_index(),
        params=params,
        specs=specs,
    )

# Save results
res_df.to_csv(path_dict["sim_results"] + f"counterfactual_1_{model_name}.csv")
