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

est_model, model_collection, params = specify_and_solve_model(
    path_dict=paths_dict,
    params=params,
    update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
    policy_state_trans_func=expected_SRA_probs_estimation,
    # note: file_append is used to load the model and solution from the file specified by the string
    file_append="old_age",
    load_model=True,
    load_solution=False,
)
est_model_batches = pickle.load(
    open(
        paths_dict["intermediate_data"] + "solved_models/model_solution_subj.pkl",
        "rb",
    )
)
model_batches = pickle.load(open(paths_dict["intermediate_data"] + "model.pkl", "rb"))

options_main = model_collection["model_main"]["options"]

state_space_names = model_batches["model_structure"]["state_space_names"]
state_choice_space = model_batches["model_structure"]["state_choice_space"]
state_choice_space_dict = model_batches["model_structure"]["state_choice_space_dict"]
map_to_deduction_state = options_main["model_params"]["old_age_state_index_mapping"]

id_state_batches = -35_000

education = state_choice_space_dict["education"][id_state_batches]
policy_state = state_choice_space_dict["policy_state"][id_state_batches]
experience = state_choice_space_dict["experience"][id_state_batches]
retirement_age_id = state_choice_space_dict["retirement_age_id"][id_state_batches]
choice = state_choice_space_dict["choice"][id_state_batches]
lagged_choice = state_choice_space_dict["lagged_choice"][id_state_batches]
period = state_choice_space_dict["period"][id_state_batches]

deduction_state = map_to_deduction_state[policy_state, retirement_age_id]
state_old_age = {
    "period": period - options_main["state_space"]["n_periods"] + 1,
    "lagged_choice": 0,
    "education": education,
    "experience": experience,
    "deduction_state": deduction_state,
    "choice": 0,
    "dummy_exog": 0,
}
from dcegm.interface import get_state_choice_index_per_state

model_structure_old_age = model_collection["model_old_age"]["model_structure"]
observed_state_choice_indexes = get_state_choice_index_per_state(
    states=state_old_age,
    map_state_choice_to_index=model_structure_old_age["map_state_choice_to_index"],
    state_space_names=model_structure_old_age["state_space_names"],
)
value_old_age = est_model["value_old_age"][observed_state_choice_indexes[0]]
endog_grid_old_age = est_model["endog_grid_old_age"][observed_state_choice_indexes[0]]
policy_old_age = est_model["policy_old_age"][observed_state_choice_indexes[0]]
exog_grid = endog_grid_old_age - policy_old_age
value_batches = est_model_batches["value"][id_state_batches]
endog_grid_batches = est_model_batches["endog_grid"][id_state_batches]
policy_batches = est_model_batches["policy"][id_state_batches]
exog_grid_batches = endog_grid_batches - policy_batches
from model_code.wealth_and_budget.old_age_budget_equation import (
    old_age_budget_constraint,
)

next_period_wealth = np.empty_like(exog_grid)
for i, saving in enumerate(exog_grid):
    next_period_wealth[i] = old_age_budget_constraint(
        experience=experience,
        deduction_state=deduction_state,
        savings_end_of_previous_period=saving,
        income_shock_previous_period=None,
        params=params,
        options=model_collection["model_old_age"]["options"]["model_params"],
    )

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(endog_grid_old_age, value_old_age, label="old age")
# ax.plot(endog_grid_batches, value_batches, label="batches")
# ax.plot(exog_grid, policy_old_age, label="old age")
# ax.plot(exog_grid_batches, policy_batches, label="batches")
ax.set_title("value")
ax.legend()
plt.show()


breakpoint()
# value = est_model["value"]
# policy = est_model["policy"]
# endog_grid = est_model["endog_grid"]


# %%
# load and modify data
def load_and_modify_data(paths_dict, options):
    start_age = options["model_params"]["start_age"]

    data_decision = pd.read_pickle(
        paths_dict["intermediate_data"] + "structural_estimation_sample.pkl"
    )
    data_decision["wealth"] = data_decision["wealth"].clip(lower=1e-16)
    data_decision["age"] = data_decision["period"] + start_age
    data_decision = data_decision[data_decision["lagged_choice"] < 2]
    data_decision["wealth_tercile"] = data_decision.groupby("age")["wealth"].transform(
        lambda x: pd.qcut(x, 3, labels=False)
    )
    data_decision["experience_tercile"] = data_decision.groupby("age")[
        "experience"
    ].transform(lambda x: pd.qcut(x, 3, labels=False, duplicates="drop"))
    return data_decision


data_decision = load_and_modify_data(paths_dict, option_collection["options_main"])

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
    model=model_collection["model_main"],
    options=option_collection["options_main"],
)
# %%
# test if all predicted probabilities of choices conditional on state are not nan
choice_probs_each_obs = jnp.take_along_axis(
    choice_probs_observations, data_decision["choice"].values[:, None], axis=1
)[:, 0]
breakpoint()
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
