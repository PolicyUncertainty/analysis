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
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# %%
# specify parameters
import os

model_fit_dir = analysis_path + "output/plots/model_fits/"
os.makedirs(model_fit_dir, exist_ok=True)


params = pickle.load(open(paths_dict["est_results"] + "est_params.pkl", "rb"))
# params = {
#     # Utility parameters
#     "mu": 0.5,
#     "dis_util_work": 4.0,
#     "dis_util_unemployed": 1.0,
#     "bequest_scale": 1.3,
#     # Taste and income shock scale
#     "lambda": 0.5,
#     # Interest rate and discount factor
#     "interest_rate": 0.03,
#     "beta": 0.95,
# }

# specify and solve model

from model_code.model_solver import specify_and_solve_model
from model_code.policy_states_belief import expected_SRA_probs_estimation
from model_code.policy_states_belief import update_specs_exp_ret_age_trans_mat

est_model, model, options, params = specify_and_solve_model(
    path_dict=paths_dict,
    params=params,
    update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
    policy_state_trans_func=expected_SRA_probs_estimation,
    # note: file_append is used to load the model and solution from the file specified by the string
    file_append="subj",
    load_model=True,
    load_solution=False,
)
value = est_model["value"]
policy = est_model["policy"]
endog_grid = est_model["endog_grid"]


# %%
# load and modify data
def load_and_modify_data(paths_dict, options):
    start_age = options["model_params"]["start_age"]

    data_decision = pd.read_pickle(
        paths_dict["intermediate_data"] + "decision_data.pkl"
    )
    data_decision["wealth"] = data_decision["wealth"].clip(lower=1e-16)
    data_decision["age"] = data_decision["period"] + start_age
    data_decision = data_decision[data_decision["age"] < 75]
    data_decision["wealth_tercile"] = data_decision.groupby("age")["wealth"].transform(
        lambda x: pd.qcut(x, 3, labels=False)
    )
    data_decision["experience_tercile"] = data_decision.groupby("age")[
        "experience"
    ].transform(lambda x: pd.qcut(x, 3, labels=False, duplicates="drop"))
    return data_decision


data_decision = load_and_modify_data(paths_dict, options)

# %%
# create choice probs for each observation
from dcegm.interface import get_state_choice_index_per_state
from dcegm.likelihood import calc_choice_probs_for_states


def create_choice_probs_for_each_observation(
    value_solved, endog_grid_solved, params, data_decision, model, options
):
    model_structure = model["model_structure"]
    states_dict = {
        name: data_decision[name].values
        for name in model_structure["state_space_names"]
    }
    observed_state_choice_indexes = get_state_choice_index_per_state(
        states=states_dict,
        map_state_choice_to_index=model_structure["map_state_choice_to_index"],
        state_space_names=model_structure["state_space_names"],
    )
    choice_probs_observations = calc_choice_probs_for_states(
        value_solved=value_solved,
        endog_grid_solved=endog_grid_solved,
        params=params,
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
    return data_decision, choice_probs_observations, observed_state_choice_indexes


(
    data_decision,
    choice_probs_observations,
    observed_state_choice_indexes,
) = create_choice_probs_for_each_observation(
    value_solved=est_model["value"],
    endog_grid_solved=est_model["endog_grid"],
    params=params,
    data_decision=data_decision,
    model=model,
    options=options,
)
# %%
# test if all predicted probabilities of choices conditional on state are not nan
choice_probs_each_obs = jnp.take_along_axis(
    choice_probs_observations, data_decision["choice"].values[:, None], axis=1
)[:, 0]
explained_0 = data_decision[data_decision["choice"] == 0]["choice_0"].mean()
explained_1 = data_decision[data_decision["choice"] == 1]["choice_1"].mean()
explained_2 = data_decision[data_decision["choice"] == 2]["choice_2"].mean()
# %%
# Plot observed choice shares by age
age_range = np.arange(30, 70, 1)
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
fig.savefig(model_fit_dir + "observed_choice_shares_by_age.png")
# %%
# Plot predicted choice probabilities by age
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


# %%
