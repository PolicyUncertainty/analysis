import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dcegm.likelihood import create_choice_prob_func_unobserved_states
from model_code.derive_specs import generate_derived_and_data_derived_specs
from model_code.specify_model import specify_and_solve_model
from model_code.stochastic_processes.policy_states_belief import (
    expected_SRA_probs_estimation,
)
from model_code.stochastic_processes.policy_states_belief import (
    update_specs_exp_ret_age_trans_mat,
)


def observed_model_fit(paths_dict):
    params = pickle.load(open(paths_dict["est_results"] + "est_params.pkl", "rb"))
    specs = generate_derived_and_data_derived_specs(paths_dict)

    est_model, model, params = specify_and_solve_model(
        path_dict=paths_dict,
        params=params,
        update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
        policy_state_trans_func=expected_SRA_probs_estimation,
        file_append="subj",
        load_model=True,
        load_solution=True,
    )

    data_decision = pd.read_pickle(
        paths_dict["intermediate_data"] + "structural_estimation_sample.pkl"
    )
    data_decision["wealth"] = data_decision["wealth"].clip(lower=1e-16)
    data_decision["age"] = data_decision["period"] + specs["start_age"]
    data_decision = data_decision[data_decision["age"] < 75]
    model_structure = model["model_structure"]
    states_dict = {
        name: data_decision[name].values
        for name in model_structure["state_space_names"]
    }

    def weight_func(**kwargs):
        # We need to weight the unobserved job offer state for each of its possible values
        # The weight function is called with job offer new beeing the unobserved state
        job_offer = kwargs["job_offer_new"]
        return model["model_funcs"]["processed_exog_funcs"]["job_offer"](**kwargs)[
            job_offer
        ]

    relevant_prev_period_state_choices_dict = {
        "period": data_decision["period"].values - 1,
        "education": data_decision["education"].values,
    }
    unobserved_state_specs = {
        "observed_bool": data_decision["full_observed_state"].values,
        "weight_func": weight_func,
        "states": ["job_offer"],
        "pre_period_states": relevant_prev_period_state_choices_dict,
        "pre_period_choices": data_decision["lagged_choice"].values,
    }

    for choice in range(3):
        choice_vals = np.ones_like(data_decision["choice"].values) * choice
        choice_prob_func = create_choice_prob_func_unobserved_states(
            model=model,
            observed_states=states_dict,
            observed_wealth=data_decision["wealth"].values,
            observed_choices=choice_vals,
            unobserved_state_specs=unobserved_state_specs,
            weight_full_states=False,
        )

        choice_probs_observations = choice_prob_func(
            value_in=est_model["value"],
            endog_grid_in=est_model["endog_grid"],
            params_in=params,
        )

        choice_probs_observations = np.nan_to_num(choice_probs_observations, nan=0.0)
        data_decision[f"choice_{choice}"] = choice_probs_observations

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
