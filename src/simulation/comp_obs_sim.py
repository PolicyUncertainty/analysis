# %%
import pickle
import sys
from pathlib import Path

import jax
import numpy as np
import pandas as pd

file_dir_path = str(Path(__file__).resolve().parents[0]) + "/"
analysis_path = str(Path(__file__).resolve().parents[2]) + "/"

import pandas as pd


data_obs = pd.read_pickle(analysis_path + "output/decision_data.pkl")
data_sim = pd.read_pickle(file_dir_path + "results_and_data/simulated_data.pkl")
# Generate age
data_sim.reset_index(inplace=True)
data_sim["age"] = data_sim["period"] + 30
data_obs["age"] = data_obs["period"] + 30
# # %%
# # Generate plots for each dataframe and save them
# ax = (
#     data_obs.groupby("age")["choice"]
#     .value_counts(normalize=True)
#     .unstack()
#     .plot(kind="bar", stacked=True)
# )
# ax.legend(loc="upper left")
# ax.set_title("Observed choice probabilities by age")
# fig = ax.get_figure()
# fig.savefig(file_dir_path + "model_fits/observed_choice_probabilities_by_age.png")
# # %%
# ax1 = (
#     data_sim.groupby("age")["choice"]
#     .value_counts(normalize=True)
#     .unstack()
#     .plot(kind="bar", stacked=True)
# )
# ax1.legend(loc="upper left")
# ax1.set_title("Simulated choice probabilities by age")
# fig1 = ax1.get_figure()
# fig1.savefig(file_dir_path + "model_fits/simulated_choice_probabilities_by_age.png")


# Plot average wealth by age for both datasets
ax2 = data_obs.groupby("age")["wealth"].median().plot()
ax2.set_title("Average wealth by age in observed data")
ax2.legend(loc="upper left")
fig2 = ax2.get_figure()
fig2.savefig(file_dir_path + "model_fits/average_wealth_by_age_observed.png")

ax3 = data_sim.groupby("age")["resources_at_beginning"].median().plot()
ax3.set_title("Average wealth by age in simulated data")
ax3.legend(loc="upper left")
fig3 = ax3.get_figure()
fig3.savefig(file_dir_path + "model_fits/average_wealth_by_age_simulated.png")

# %%
