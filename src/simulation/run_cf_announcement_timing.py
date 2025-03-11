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
model_name = "partner_est"
load_base_solution = True  # baseline solution conntainer
load_second_solution = True  # counterfactual solution conntainer
load_sol_model = True  # informed state as types
load_sim_model = True  # informed state stochastic
load_df = (
    True  # True = load existing df, False = create new df, None = create but not save
)

# Load params
params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

# initiaize announcement ages
annoucement_ages = [35, 45, 55]

# initialize result dfs
res_df_life_cycle = pd.DataFrame(dtype=float)


for announcement_age in annoucement_ages:
    if load_df:
        print("Load existing dfs for announcement age: ", announcement_age)
    else:
        print("Start simulation for announcement age: ", announcement_age)

    # Simulate baseline with subjective belief
    df_base = solve_and_simulate_scenario(
        path_dict=path_dict,
        params=params,
        subj_unc=True,
        custom_resolution_age=None,
        annoucement_age=63,  # earliest age is 31
        SRA_at_retirement=69,
        SRA_at_start=67,
        model_name=model_name,
        df_exists=load_df,
        solution_exists=load_base_solution,
        sol_model_exists=load_sol_model,
        sim_model_exists=load_sim_model,
    ).reset_index()


    # Simulate counterfactual 
    df_cf = solve_and_simulate_scenario(
        path_dict=path_dict,
        params=params,
        subj_unc=True,
        custom_resolution_age=None,
        annoucement_age= announcement_age,  # Let's earliest age is 31
        SRA_at_retirement=69,
        SRA_at_start=67,
        model_name=model_name,
        df_exists=load_df,
        solution_exists=load_base_solution,
        sol_model_exists=load_sol_model,
        sim_model_exists=load_sim_model,
    ).reset_index()

    # Calculate results
    ages = np.arange(30, 101, 1)
    for df in [df_base, df_cf]:
        if df is df_base:
            string = "base"
        else:
            string = "cf"

        res_df_life_cycle[f"working_hours_{announcement_age}_{string}"] = df.groupby("age")[
            "working_hours"
        ].aggregate("mean")
        res_df_life_cycle[f"savings_{announcement_age}_{string}"] = df.groupby("age")[
            "savings_dec"
        ].aggregate("mean")
        res_df_life_cycle[f"consumption_{announcement_age}_{string}"] = df.groupby("age")[
            "consumption"
        ].aggregate("mean")
        res_df_life_cycle[f"income_{announcement_age}_{string}"] = df.groupby("age")[
            "total_income"
        ].aggregate("mean")
        res_df_life_cycle[f"assets_{announcement_age}_{string}"] = df.groupby("age")[
            "savings"
        ].aggregate("mean")
        res_df_life_cycle[f"savings_rate_{announcement_age}_{string}"] = (
            res_df_life_cycle[f"savings_{announcement_age}_{string}"]
            / res_df_life_cycle[f"income_{announcement_age}_{string}"]
        )
        res_df_life_cycle[f"employment_rate_{announcement_age}_{string}"] = df.groupby("age")[
            "choice"
        ].apply(lambda x: x.isin([2, 3]).sum() / len(x))
        res_df_life_cycle[f"retirement_rate_{announcement_age}_{string}"] = df.groupby("age")[
            "choice"
        ].apply(lambda x: x.isin([0]).sum() / len(x))
    res_df_life_cycle[f"savings_rate_diff_{announcement_age}"] = (
        res_df_life_cycle[f"savings_rate_{announcement_age}_cf"]
        - res_df_life_cycle[f"savings_rate_{announcement_age}_base"]
    )
    res_df_life_cycle[f"employment_rate_diff_{announcement_age}"] = (
        res_df_life_cycle[f"employment_rate_{announcement_age}_cf"]
        - res_df_life_cycle[f"employment_rate_{announcement_age}_base"]
    )
    res_df_life_cycle[f"retirement_rate_diff_{announcement_age}"] = (
        res_df_life_cycle[f"retirement_rate_{announcement_age}_cf"]
        - res_df_life_cycle[f"retirement_rate_{announcement_age}_base"]
    ) 

import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 1, figsize=(10, 10))
subset_ages = np.arange(30, 81, 2)
filtered_df = res_df_life_cycle.loc[subset_ages]

for announcement_age in annoucement_ages:
    savings_rate_diff = filtered_df[f"savings_rate_diff_{announcement_age}"]
    employment_rate_diff = filtered_df[f"employment_rate_diff_{announcement_age}"]
    retirement_rate_diff = filtered_df[f"retirement_rate_diff_{announcement_age}"]
    ax[0].plot(filtered_df.index, savings_rate_diff, label=f"SRA announcement at age {announcement_age}")
    ax[1].plot(filtered_df.index, employment_rate_diff, label=f"SRA announcement age {announcement_age}")
    ax[2].plot(filtered_df.index, retirement_rate_diff, label=f"SRA announcement age {announcement_age}")

ax[0].set_title("Savings rate difference")
ax[1].set_title("Employment rate difference")
ax[2].set_title("Retirement rate difference")
for axis in ax:
    axis.axhline(y=0, color='black')
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.3f}'))
plt.tight_layout()
plt.legend()
plt.show()
plt.show()
