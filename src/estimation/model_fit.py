# %%
# Set paths of project
import sys
from pathlib import Path

analysis_path = str(Path(__file__).resolve().parents[2]) + "/"
sys.path.insert(0, analysis_path + "submodules/dcegm/src/")
sys.path.insert(0, analysis_path + "src/")

from set_paths import create_path_dict

paths_dict = create_path_dict(analysis_path)

import pickle
import jax
import numpy as np
import pandas as pd


jax.config.update("jax_enable_x64", True)

# %%
import os

model_fit_dir = analysis_path + "output/plots/model_fits/"
os.makedirs(model_fit_dir, exist_ok=True)


est_params = pickle.load(open(paths_dict["est_results"] + "est_params.pkl", "rb"))

from model_code.model_solver import solve_model
from model_code.policy_states_belief import expected_SRA_probs_estimation
from model_code.policy_states_belief import update_specs_exp_ret_age_trans_mat

est_model = solve_model(
    path_dict=paths_dict,
    params=est_params,
    update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
    policy_state_trans_func=expected_SRA_probs_estimation,
    file_append="est",
    load_model=True,
    load_solution=True,
)

from model_code.specify_model import specify_model

model, options, params = specify_model(
    path_dict=paths_dict,
    params=est_params,
    update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
    policy_state_trans_func=expected_SRA_probs_estimation,
    load_model=True,
)

data_decision = pd.read_pickle(paths_dict["intermediate_data"] + "decision_data.pkl")
data_decision["age"] = data_decision["period"] + 30
data_decision = data_decision[data_decision["age"] < 75]

from dcegm.likelihood import create_observed_choice_indexes
from dcegm.likelihood import calc_choice_probs_for_observed_states

states_dict = {name: data_decision[name].values for name in model["state_space_names"]}
observed_state_choice_indexes = create_observed_choice_indexes(states_dict, model)
choice_probs_observations = calc_choice_probs_for_observed_states(
    value_solved=est_model["value"],
    endog_grid_solved=est_model["endog_grid"],
    params=est_params,
    observed_states=states_dict,
    state_choice_indexes=observed_state_choice_indexes,
    oberseved_wealth=data_decision["wealth"].values,
    choice_range=np.arange(options["model_params"]["n_choices"], dtype=int),
    compute_utility=model["model_funcs"]["compute_utility"],
)
choice_probs_observations = np.nan_to_num(choice_probs_observations, nan=0.0)
data_decision["choice_0"] = choice_probs_observations[:, 0]
data_decision["choice_1"] = choice_probs_observations[:, 1]
data_decision["choice_2"] = choice_probs_observations[:, 2]


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
