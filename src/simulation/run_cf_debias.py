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
from simulation.sim_tools.calc_life_time_results import add_new_life_cycle_results

from export_results.tables.cv import calc_compensated_variation

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

# Create life cylce df object which is None. The result function will then initialize in the first iteration
res_df_life_cycle = None

# Initialize retirement ages
# sra_at_63 = np.arange(67, 70 + specs["SRA_grid_size"], specs["SRA_grid_size"])
# sra_at_63 = np.arange(67, 70 + specs["SRA_grid_size"], 1)
sra_at_63 = [67.0, 68.0, 69.0, 70.0]
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
        subj_unc=True,
        custom_resolution_age=None,
        annoucement_age=None,
        SRA_at_retirement=sra,
        SRA_at_start=sra,
        model_name=model_name,
        df_exists=load_df,
        only_informed=True,
        solution_exists=load_second_solution,
        sol_model_exists=load_sol_model,
        sim_model_exists=load_sim_model,
    ).reset_index()

    res_df_life_cycle = add_new_life_cycle_results(
        df_base=df_base,
        df_cf=df_cf,
        sra=sra,
        res_df_life_cycle=res_df_life_cycle,
    )
    breakpoint()



import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
subset_ages = np.arange(30, 81, 2)
filtered_df = res_df_life_cycle.loc[subset_ages]

colors = {
    67.0: "darkblue",
    68.0: "lightblue",
    69.0: "lightcoral",  # light red
    70.0: "darkred",
}

for i, sra in enumerate(sra_at_63):
    savings_diff = filtered_df[f"savings_rate_diff_{sra}"]
    labor_supply_diff = filtered_df[f"employment_rate_diff_{sra}"]
    retirement_diff = filtered_df[f"retirement_rate_diff_{sra}"]
    ax[0].plot(
        filtered_df.index, savings_diff, label=f"SRA at {sra}", color=colors[sra]
    )
    ax[1].plot(
        filtered_df.index, labor_supply_diff, label=f"SRA at {sra}", color=colors[sra]
    )
    ax[2].plot(
        filtered_df.index, retirement_diff, label=f"SRA at {sra}", color=colors[sra]
    )
ax[0].set_title("Difference in savings rate")
ax[1].set_title("Difference in employment rate")
ax[2].set_title("Difference in retirement rate")
plt.legend()
for axis in ax:
    axis.axhline(y=0, color='black')
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.3f}'))
plt.tight_layout()
plt.legend()
plt.show()