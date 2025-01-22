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
params = pkl.load(open(path_dict["est_params"], "rb"))

# Initialize alpha values and replace 0.04 with subjective alpha
alphas_realized = np.arange(0, 0.11, 0.01)
alphas_realized[alphas_realized == 0.04] = np.loadtxt(
    path_dict["est_results"] + "exp_val_params.txt"
)

# Create result df
result_df = pd.DataFrame(
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
result_df["alpha"] = alphas_realized
for i, alpha_sim in enumerate(alphas_realized):
    print("Start simulation for alpha: ", alpha_sim)
    # Create estimated model
    df = solve_and_simulate_scenario(
        path_dict=path_dict,
        params=params,
        sim_alpha=alpha_sim,
        expected_alpha=False,
        model_name="pete",
        df_exists=False,
        solution_exists=True,
        sol_model_exists=True,
        sim_model_exists=True,
    )

    result_df.loc[i, "below_sixty_savings"] = below_sixty_savings(df)
    result_df.loc[i, "ret_age"] = calc_average_retirement_age(df)
    result_df.loc[i, "sra_at_ret"] = sra_at_retirement(df)
    result_df.loc[i, "working_hours"] = df["working_hours"].mean()

    if i == 0:
        df_base = df.reset_index().copy()
    else:
        result_df.loc[i, "cv"] = calc_compensated_variation(
            df_base=df_base,
            df_cf=df.reset_index(),
            params=params,
            specs=specs,
        )

# Save results
result_df.to_csv("counterfactual_2.csv")
