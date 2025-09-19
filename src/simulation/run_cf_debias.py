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

from simulation.figures.retirement_plot import plot_retirement_difference
from simulation.sim_tools.calc_life_time_results import add_new_life_cycle_results
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario
from simulation.tables.cv import calc_compensated_variation

# %%
# Set specifications
seeed = 123
model_name = "all_men_3"
men_only = True
load_model = True  # informed state as type
load_unc_solution = False  # baseline solution conntainer
load_df_biased = False
load_df_unbiased = (
    False  # True = load existing df, False = create new df, None = create but not save
)


# Load params
params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

# Create life cylce df object which is None.
# The result function will then initialize in the first iteration.
res_df_life_cycle = None

# Initialize retirement ages
# sra_at_63 = np.arange(67, 70 + specs["SRA_grid_size"], specs["SRA_grid_size"])
# sra_at_63 = np.arange(67, 70 + specs["SRA_grid_size"], 1)
sra_at_63 = [67.0, 68.0, 69.0, 70.0]
for i, sra in enumerate(sra_at_63):
    print(sra)

    # Simulate baseline with subjective belief
    df_base, _ = solve_and_simulate_scenario(
        path_dict=path_dict,
        params=params,
        subj_unc=True,
        custom_resolution_age=None,
        announcement_age=None,
        SRA_at_retirement=sra,
        SRA_at_start=67,
        model_name=model_name,
        df_exists=load_df_biased,
        only_informed=False,
        solution_exists=load_unc_solution,
        sol_model_exists=load_model,
        men_only=men_only,
    )

    df_base = df_base.reset_index()

    load_unc_solution = True if load_unc_solution is not None else load_unc_solution
    load_model = True

    # Simulate counterfactual with no uncertainty and expected increase
    # same as simulated alpha_sim
    df_cf, _ = solve_and_simulate_scenario(
        path_dict=path_dict,
        params=params,
        subj_unc=True,
        custom_resolution_age=None,
        announcement_age=None,
        SRA_at_retirement=sra,
        SRA_at_start=67,
        model_name=model_name,
        df_exists=load_df_unbiased,
        only_informed=True,
        solution_exists=load_unc_solution,
        sol_model_exists=load_model,
    )

    fig = plot_retirement_difference(
        df_base=df_base,
        df_cf=df_cf,
        final_SRA=sra,
        left_difference=-4,
        right_difference=2,
        base_label="With Uninformed",
        cf_label="Only Informed",
    )
    fig.savefig(
        path_dict["plots"] + f"retirement_inflow_comparison_sra_{sra}_{model_name}.png"
    )

    df_cf = df_cf.reset_index()

    res_df_life_cycle = add_new_life_cycle_results(
        df_base=df_base,
        df_cf=df_cf,
        scenatio_indicator=sra,
        res_df_life_cycle=res_df_life_cycle,
    )

res_df_life_cycle.to_csv(path_dict["sim_results"] + f"debias_lc_{model_name}.csv")
