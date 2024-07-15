import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dcegm.interface import get_state_choice_index_per_state
from dcegm.likelihood import calc_choice_probs_for_states
from model_code.derive_specs import generate_derived_and_data_derived_specs
from model_code.model_solver import specify_and_solve_model
from model_code.policy_states_belief import expected_SRA_probs_estimation
from model_code.policy_states_belief import update_specs_exp_ret_age_trans_mat


def observed_model_fit(paths_dict):
    params = pickle.load(open(paths_dict["est_results"] + "est_params.pkl", "rb"))
    specs = generate_derived_and_data_derived_specs(paths_dict)

    est_model, model_collection, params = specify_and_solve_model(
        path_dict=paths_dict,
        params=params,
        update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
        policy_state_trans_func=expected_SRA_probs_estimation,
        file_append="subj",
        load_model=True,
        load_solution=False,
    )

    model_main = model_collection["model_main"]

    data_decision = pd.read_pickle(
        paths_dict["intermediate_data"] + "structural_estimation_sample.pkl"
    )
    data_decision["wealth"] = data_decision["wealth"].clip(lower=1e-16)
    data_decision["age"] = data_decision["period"] + specs["start_age"]
    data_decision = data_decision[data_decision["age"] < 75]
    model_structure = model_main["model_structure"]
    states_dict = {
        name: data_decision[name].values
        for name in model_structure["state_space_names"]
    }

    observed_state_choice_indexes = get_state_choice_index_per_state(
        states=states_dict,
        map_state_choice_to_index=model_main["model_structure"][
            "map_state_choice_to_index"
        ],
        state_space_names=model_main["model_structure"]["state_space_names"],
    )
    choice_probs_observations = calc_choice_probs_for_states(
        value_solved=est_model["value"],
        endog_grid_solved=est_model["endog_grid"],
        params=params,
        observed_states=states_dict,
        state_choice_indexes=observed_state_choice_indexes,
        oberseved_wealth=data_decision["wealth"].values,
        choice_range=np.arange(
            model_main["options"]["model_params"]["n_choices"], dtype=int
        ),
        compute_utility=model_main["model_funcs"]["compute_utility"],
    )
    choice_probs_observations = np.nan_to_num(choice_probs_observations, nan=0.0)
    data_decision["choice_0"] = choice_probs_observations[:, 0]
    data_decision["choice_1"] = choice_probs_observations[:, 1]
    data_decision["choice_2"] = choice_probs_observations[:, 2]

    file_append = ["low", "high"]

    for edu in range(2):
        data_edu = data_decision[data_decision["education"] == edu]
        choice_shares_obs = (
            data_edu.groupby(["age"])["choice"].value_counts(normalize=True).unstack()
        )

        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        labels = ["Unemployment", "Employment", "Retirement"]
        for choice, ax in enumerate(axes):
            choice_shares_predicted = data_edu.groupby(["age"])[
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
            paths_dict["plots"] + f"observed_model_fit_{file_append[edu]}.png",
            transparent=True,
            dpi=300,
        )
