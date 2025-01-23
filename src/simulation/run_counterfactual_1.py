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
model_name = "both"
load_base_solution = False
load_sol_model = False
load_sim_model = False
load_solution = None
load_df = None


# Load params
params = pkl.load(open(path_dict["est_results"] + f"est_params_{model_name}.pkl", "rb"))

# Initialize alpha values and replace 0.04 with subjective alpha(0.041...)
alphas_realized = np.arange(0, 0.11, 0.01)
alphas_realized[alphas_realized == 0.04] = np.loadtxt(
    path_dict["est_results"] + "exp_val_params.txt"
)

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
res_df["alpha"] = alphas_realized
for i, alpha_sim in enumerate(alphas_realized):
    print("Start simulation for alpha: ", alpha_sim)

    # Simulate baseline with subjective belief
    df_base = solve_and_simulate_scenario(
        path_dict=path_dict,
        params=params,
        sim_alpha=alpha_sim,
        expected_alpha=False,
        model_name=model_name,
        df_exists=load_df,
        solution_exists=load_base_solution,
        sol_model_exists=load_sol_model,
        sim_model_exists=load_sim_model,
    ).reset_index()

    load_sim_model = True
    load_sol_model = True
    load_base_solution = True

    # Simulate counterfactual with no uncertainty and expected increase
    # same as simulated alpha_sim
    df_cf = solve_and_simulate_scenario(
        path_dict=path_dict,
        params=params,
        sim_alpha=alpha_sim,
        expected_alpha=alpha_sim,
        model_name=model_name,
        df_exists=load_df,
        solution_exists=load_solution,
        sol_model_exists=load_sol_model,
        sim_model_exists=load_sim_model,
    ).reset_index()

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
