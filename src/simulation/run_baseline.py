# %%
import pandas as pd
import pickle as pkl
import jax

from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario
from simulation.sim_tools.calc_life_time_results import add_new_life_cycle_results

jax.config.update("jax_enable_x64", True)

# %%
path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)
model_name = specs["model_name"]

params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

# %%
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
res_df_life_cycle = add_new_life_cycle_results(
    df_base=df_baseline,
    df_cf=df_baseline,
    scenatio_indicator="baseline",
    res_df_life_cycle=None,
)

res_df_life_cycle.to_csv(path_dict["intermediate_data"] + f"baseline_lc_{model_name}.csv")