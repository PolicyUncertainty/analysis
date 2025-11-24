# %%
import os
import pickle as pkl

from set_paths import create_path_dict
from simulation.internal_runs.internal_sim_tools.calc_life_cycle_detailed import (
    calc_life_cycle_detailed,
)
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario
from specs.derive_specs import generate_derived_and_data_derived_specs

# %%
path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)
model_name = specs["model_name"]
load_sol_model = True
load_unc_sol = None  # baseline solution conntainer
load_no_unc_solution = None  # no uncertainty solution container
load_baseline_df = None  # baseline dataframe
load_no_unc_df = None

params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

# %%
# baseline: sra 67, with uncertainty and misinformation
df_baseline, model = solve_and_simulate_scenario(
    path_dict=path_dict,
    params=params,
    subj_unc=False,
    custom_resolution_age=None,
    announcement_age=None,
    SRA_at_retirement=67,
    SRA_at_start=67,
    model_name=model_name,
    df_exists=load_baseline_df,
    only_informed=False,
    solution_exists=load_unc_sol,
    sol_model_exists=load_sol_model,
    util_type=specs["util_type"],
)

df_baseline = df_baseline.reset_index()
# filter by sex
df_baseline = df_baseline[df_baseline["sex"] == 1]

# %%
# Generate detailed life cycle results
df_lc_detailed = calc_life_cycle_detailed(df_baseline)
del df_baseline
del model


# Save detailed results
output_path = path_dict["simulation_data"] + "baseline/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

df_lc_detailed.to_csv(output_path + f"67_no_unc_lc_{model_name}.csv")


# baseline: sra 67, without uncertainty but with misinformation
df_cf, _ = solve_and_simulate_scenario(
    path_dict=path_dict,
    params=params,
    subj_unc=False,
    custom_resolution_age=None,
    announcement_age=None,
    SRA_at_retirement=69,
    SRA_at_start=67,
    model_name=model_name,
    df_exists=load_no_unc_df,
    only_informed=False,
    solution_exists=load_no_unc_solution,
    sol_model_exists=load_sol_model,
    util_type=specs["util_type"],
)

df_cf = df_cf.reset_index()
df_cf = df_cf[df_cf["sex"] == 1]


# Generate detailed life cycle results
df_cf_lc_detailed = calc_life_cycle_detailed(df_cf)

df_cf_lc_detailed.to_csv(output_path + f"69_no_unc_lc_{model_name}.csv")
