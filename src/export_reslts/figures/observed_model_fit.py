import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dcegm.likelihood import calc_choice_probs_for_observed_states
from dcegm.likelihood import create_observed_choice_indexes
from model_code.derive_specs import generate_derived_and_data_derived_specs
from model_code.model_solver import specify_and_solve_model
from model_code.policy_states_belief import expected_SRA_probs_estimation
from model_code.policy_states_belief import update_specs_exp_ret_age_trans_mat


def observed_model_fit(paths_dict):
    params = pickle.load(open(paths_dict["est_results"] + "est_params.pkl", "rb"))
    specs = generate_derived_and_data_derived_specs(paths_dict)

    est_model, model, options, params = specify_and_solve_model(
        path_dict=paths_dict,
        params=params,
        update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
        policy_state_trans_func=expected_SRA_probs_estimation,
        file_append="subj",
        load_model=True,
        load_solution=True,
    )
    data_decision = pd.read_pickle(
        paths_dict["intermediate_data"] + "decision_data.pkl"
    )
    data_decision["wealth"] = data_decision["wealth"].clip(lower=1e-16)
    data_decision["age"] = data_decision["period"] + specs["start_age"]
    data_decision = data_decision[data_decision["age"] < 75]
    model_structure = model["model_structure"]
    states_dict = {
        name: data_decision[name].values
        for name in model_structure["state_space_names"]
    }
    observed_state_choice_indexes = create_observed_choice_indexes(states_dict, model)
    choice_probs_observations = calc_choice_probs_for_observed_states(
        value_solved=est_model["value"],
        endog_grid_solved=est_model["endog_grid"],
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

    choice_shares_obs = (
        data_decision.groupby(["age"])["choice"].value_counts(normalize=True).unstack()
    )

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    labels = ["Unemployment", "Employment", "Retirement"]
    for choice, ax in enumerate(axes):
        choice_shares_predicted = data_decision.groupby(["age"])[
            f"choice_{choice}"
        ].mean()
        choice_shares_predicted.plot(ax=ax, label="Simulated")
        choice_shares_obs[choice].plot(ax=ax, label="Observed", ls="--")
        ax.set_xlabel("Age")
        ax.set_ylabel("Choice share")
        ax.set_title(f"{labels[choice]}")
        ax.set_ylim([-0.05, 1.05])
        if choice == 0:
            ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(
        paths_dict["plots"] + "observed_model_fit.png", transparent=True, dpi=300
    )
