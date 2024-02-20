# %%
import pickle
import sys
from pathlib import Path

import jax
import numpy as np
import pandas as pd

file_dir_path = str(Path(__file__).resolve().parents[0]) + "/"
analysis_path = str(Path(__file__).resolve().parents[2]) + "/"

project_paths = {
    "project_path": analysis_path,
    "model_path": file_dir_path + "results_and_data/",
}
sys.path.insert(0, analysis_path + "submodules/dcegm/src/")
sys.path.insert(0, analysis_path + "src/")

jax.config.update("jax_enable_x64", True)

from estimation.tools import process_data_and_model

data_decision = pd.read_pickle(analysis_path + "output/decision_data.pkl")
data_decision = data_decision[data_decision["lagged_choice"] != 2]
# data_decision["policy_state"] = 8

res = pickle.load(open(file_dir_path + "results_and_data/res.pkl", "rb"))
start_params_all = {
    # Utility parameters
    "mu": 0.5,
    "dis_util_work": res["x"][2],
    "dis_util_unemployed": res["x"][1],
    "bequest_scale": res["x"][0],
    # Taste and income shock scale
    "lambda": 1.0,
    "sigma": 1.0,
    # Interest rate and discount factor
    "interest_rate": 0.03,
    "beta": 0.95,
}

# %%
# solved_model = prep_data_and_model(
#     data_decision=data_decision,
#     project_paths=project_paths,
#     start_params_all=start_params_all,
#     load_model=True,
#     output="solved_model",
# )
# pickle.dump(solved_model, open(file_dir_path + "results_and_data/solved_model_67.pkl", "wb"))
# %%
solved_model = pickle.load(
    open(file_dir_path + "results_and_data/solved_model_67.pkl", "rb")
)
choice_probs_observations, value, policy_left, policy_right, endog_grid = solved_model
choice_probs_observations = np.nan_to_num(choice_probs_observations, nan=0.0)
# %%
data_decision = pd.read_pickle(analysis_path + "output/decision_data.pkl")

data_decision["choice_0"] = 0
data_decision["choice_1"] = 0
data_decision["choice_2"] = 1
data_decision.loc[
    data_decision["lagged_choice"] != 2, "choice_0"
] = choice_probs_observations[:, 0]
data_decision.loc[
    data_decision["lagged_choice"] != 2, "choice_1"
] = choice_probs_observations[:, 1]
data_decision.loc[
    data_decision["lagged_choice"] != 2, "choice_2"
] = choice_probs_observations[:, 2]
# generate age
data_decision["age"] = data_decision["period"] + 30
data_decision = data_decision[data_decision["age"] < 75]
# %%
import os

os.makedirs(file_dir_path + "model_fits", exist_ok=True)
model_fit_dir = file_dir_path + "model_fits/"


age_range = np.arange(31, 70, 1)
ax = (
    data_decision.groupby("age")["choice"]
    .value_counts(normalize=True)
    .loc[age_range]
    .unstack()
    .plot(kind="bar", stacked=True)
)
ax.legend(loc="upper left")
ax.set_title("Observed choice probabilities by age")
fig = ax.get_figure()
fig.savefig(model_fit_dir + "observed_choice_probabilities_by_age.png")
# Create same plot as above but instead of value counting choice 0, 1, 2, we use the predicted choice probabilities
# choice_0, choice_1, choice_2.
ax1 = (
    data_decision.groupby("age")[["choice_0", "choice_1", "choice_2"]]
    .mean()
    .loc[age_range]
    .plot(kind="bar", stacked=True)
)
ax1.legend(loc="upper left")
ax1.set_title("Predicted choice probabilities by age")
fig1 = ax1.get_figure()
fig1.savefig(model_fit_dir + "predicted_choice_probabilities_by_age.png")
# %%
# Create wealth terciles as the terciles in each group. Individuals can therefore change tercile over time.
data_decision["wealth_tercile"] = data_decision.groupby("age")["wealth"].transform(
    lambda x: pd.qcut(x, 3, labels=False)
)


# %%
for id_tercile in range(3):
    ax = (
        data_decision[data_decision["wealth_tercile"] == id_tercile]
        .groupby("age")["choice"]
        .value_counts(normalize=True)
        .unstack()
        .plot(kind="bar", stacked=True)
    )
    ax.legend(loc="upper left")
    ax.set_title(
        f"Observed choice probabilities by age for wealth tercile {id_tercile}"
    )
    fig = ax.get_figure()
    fig.savefig(
        model_fit_dir
        + f"observed_choice_probabilities_by_age_wealth_tercile_{id_tercile}.png"
    )

    ax1 = (
        data_decision[data_decision["wealth_tercile"] == id_tercile]
        .groupby("age")[["choice_0", "choice_1", "choice_2"]]
        .mean()
        .plot(kind="bar", stacked=True)
    )
    ax1.legend(loc="upper left")
    ax1.set_title(
        f"Predicted choice probabilities by age for wealth tercile {id_tercile}"
    )
    fig1 = ax1.get_figure()
    fig1.savefig(
        model_fit_dir
        + f"predicted_choice_probabilities_by_age_wealth_tercile_{id_tercile}.png"
    )

# Now plot wealth tercile values by age, i.e. the 0.33 and 0.66 values for wealth. Make three distinct lines
ax = data_decision.groupby("age")["wealth"].quantile([0.33, 0.66]).unstack().plot()
ax.legend(loc="upper left")
ax.set_title("Wealth terciles by age")
ax.get_figure().savefig(model_fit_dir + "wealth_terciles_by_age.png")
# %%
# Do the same analysis as before with experience terciles. Also plot experience terciles by age.
data_decision["experience_tercile"] = data_decision.groupby("age")[
    "experience"
].transform(lambda x: pd.qcut(x, 3, labels=False, duplicates="drop"))
for id_tercile in range(3):
    ax = (
        data_decision[data_decision["experience_tercile"] == id_tercile]
        .groupby("age")["choice"]
        .value_counts(normalize=True)
        .unstack()
        .plot(kind="bar", stacked=True)
    )
    ax.legend(loc="upper left")
    ax.set_title(
        f"Observed choice probabilities by age for experience tercile {id_tercile}"
    )
    fig = ax.get_figure()
    fig.savefig(
        model_fit_dir
        + f"observed_choice_probabilities_by_age_experience_tercile_{id_tercile}.png"
    )

    ax1 = (
        data_decision[data_decision["experience_tercile"] == id_tercile]
        .groupby("age")[["choice_0", "choice_1", "choice_2"]]
        .mean()
        .plot(kind="bar", stacked=True)
    )
    ax1.legend(loc="upper left")
    ax1.set_title(
        f"Predicted choice probabilities by age for experience tercile {id_tercile}"
    )
    fig1 = ax1.get_figure()
    fig1.savefig(
        model_fit_dir
        + f"predicted_choice_probabilities_by_age_experience_tercile_{id_tercile}.png"
    )

# Now plot experience tercile values by age, i.e. the 0.33 and 0.66 values for experience.
ax = data_decision.groupby("age")["experience"].quantile([0.33, 0.66]).unstack().plot()
ax.legend(loc="upper left")
ax.set_title("Experience terciles by age")
ax.get_figure().savefig(model_fit_dir + "experience_terciles_by_age.png")


# %%
