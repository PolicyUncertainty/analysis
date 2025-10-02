# %%
import pandas as pd
import pickle as pkl
import numpy as np
# paths and specs
from set_paths import create_path_dict
path_dict = create_path_dict()
from specs.derive_specs import generate_derived_and_data_derived_specs
specs = generate_derived_and_data_derived_specs(path_dict)
import jax
jax.config.update("jax_enable_x64", True)
# sim tools and plotting
from simulation.figures.retirement_plot import (
    plot_retirement_difference,
    plot_retirement_share,
)
from simulation.sim_tools.calc_life_time_results import add_new_life_cycle_results
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario
from simulation.sim_tools.simulation_print import start_simulation_print
from simulation.tables.cv import calc_compensated_variation

# %%
# Set specifications
seeed = 123
model_name = specs["model_name"]
util_type = specs["util_type"]
sra_at_63 = 69.0

load_model = False  # informed state as type
load_unc_solution = False  # baseline solution conntainer
load_df_baseline = False # True = load existing df, False = create new df, None = create but do not save
load_df_unbiased = False # same as above


# Load params
params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)


# Simulate baseline 
start_simulation_print(model_name=model_name, sra_63=sra_at_63, uncertainty=True, misinformation=True, load_model=load_model, load_solution=load_unc_solution, load_df=load_df_baseline)

df_base, model_solved_unc = solve_and_simulate_scenario(
    path_dict=path_dict,
    params=params,
    subj_unc=True,
    custom_resolution_age=None,
    announcement_age=None,
    SRA_at_retirement=sra_at_63,
    SRA_at_start=67,
    model_name=model_name,
    df_exists=load_df_baseline,
    only_informed=False,
    solution_exists=load_unc_solution,
    sol_model_exists=load_model,
    model_solution=load_unc_solution,
    util_type=util_type
)
df_base = df_base.reset_index()


# Simulate counterfactual without misinformation
start_simulation_print(model_name=model_name, sra_63=sra_at_63, uncertainty=True, misinformation=False, load_model=load_model, load_solution=load_unc_solution, load_df=load_df_unbiased)

df_cf, _ = solve_and_simulate_scenario(
    path_dict=path_dict,
    params=params,
    subj_unc=True,
    custom_resolution_age=None,
    announcement_age=None,
    SRA_at_retirement=sra_at_63,
    SRA_at_start=67,
    model_name=model_name,
    df_exists=load_df_unbiased,
    only_informed=True,
    solution_exists=load_unc_solution,
    sol_model_exists=load_model,
    model_solution=model_solved_unc,
    util_type=util_type
)
df_cf = df_cf.reset_index()

plot_retirement_difference(
    path_dict=path_dict,
    specs=specs,
    df_base=df_base,
    df_cf=df_cf,
    final_SRA=sra_at_63,
    model_name=model_name,
    left_difference=-4,
    right_difference=2,
    base_label="With Uninformed",
    cf_label="Only Informed",
)

plot_retirement_share(
    path_dict=path_dict,
    specs=specs,
    df_base=df_base,
    df_cf=df_cf,
    final_SRA=sra_at_63,
    model_name=model_name,
    base_label="With Uninformed",
    cf_label="Only Informed",
)
