# %%
import pandas as pd
import pickle as pkl
import jax
import os

from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario
from simulation.sim_tools.calc_life_cycle_detailed import calc_life_cycle_detailed

jax.config.update("jax_enable_x64", True)

# %%
path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)
model_name = specs["model_name"]

params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

# %%
# baseline: sra 67, with uncertainty and misinformation
df_baseline, _ = solve_and_simulate_scenario(
    path_dict=path_dict,
    params=params,
    subj_unc=True,
    custom_resolution_age=None,
    announcement_age=None,
    SRA_at_retirement=67,
    SRA_at_start=67,
    model_name=model_name,
    df_exists=False,
    only_informed=False,
    solution_exists=True,
    sol_model_exists=True,
)

df_baseline = df_baseline.reset_index()

# %%
# Generate detailed life cycle results
df_lc_detailed = calc_life_cycle_detailed(df_baseline)

# Save detailed results
output_path = path_dict["simulation_data"] + "/baseline/"
df_lc_detailed.to_csv(output_path + f"baseline_lc_{model_name}.csv")


# baseline: sra 67, without uncertainty but with misinformation
df_baseline_no_uncertainty, _ = solve_and_simulate_scenario(
    path_dict=path_dict,
    params=params,
    subj_unc=False,
    custom_resolution_age=None,
    announcement_age=None,
    SRA_at_retirement=67,
    SRA_at_start=67,
    model_name=model_name,
    df_exists=False,
    only_informed=False,
    solution_exists=True,
    sol_model_exists=True,
)

df_baseline_no_uncertainty = df_baseline_no_uncertainty.reset_index()

# Generate detailed life cycle results
df_lc_detailed_no_uncertainty = calc_life_cycle_detailed(df_baseline_no_uncertainty)

# Save detailed results
output_path = path_dict["simulation_data"] + "/baseline/"
df_lc_detailed_no_uncertainty.to_csv(output_path + f"baseline_lc_{model_name}_no_uncertainty.csv")


