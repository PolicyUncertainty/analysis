import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dcegm.likelihood import create_choice_prob_func_unobserved_states
from estimation.estimate_setup import load_and_prep_data
from model_code.specify_model import specify_and_solve_model
from model_code.stochastic_processes.policy_states_belief import (
    expected_SRA_probs_estimation,
)
from model_code.stochastic_processes.policy_states_belief import (
    update_specs_exp_ret_age_trans_mat,
)
from specs.derive_specs import generate_derived_and_data_derived_specs


def observed_model_fit(paths_dict):
    params = pickle.load(open(paths_dict["est_results"] + "est_params_new.pkl", "rb"))
    last_end = {
        "mu": 1.5047157067888446,
        "dis_util_work_high": 0.8506518010882294,
        "dis_util_work_low": 0.9099473575263033,
        "dis_util_unemployed_high": 6,
        "dis_util_unemployed_low": 6,
        "job_finding_logit_const": 0.15094919042166413,
        "job_finding_logit_age": -0.03065231823063151,
        "job_finding_logit_high_educ": 0.7610909327643549,
    }
    params.update(last_end)

    specs = generate_derived_and_data_derived_specs(paths_dict)

    est_model, model, params = specify_and_solve_model(
        path_dict=paths_dict,
        params=params,
        update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
        policy_state_trans_func=expected_SRA_probs_estimation,
        file_append="subj",
        load_model=True,
        load_solution=False,
    )

    data_decision, _ = load_and_prep_data(
        paths_dict, params, model, drop_retirees=False
    )
    data_decision["age"] = data_decision["period"] + specs["start_age"]
    data_decision = data_decision[data_decision["age"] < 75]
    states_dict = {
        name: data_decision[name].values
        for name in model["model_structure"]["discrete_states_names"]
    }
    states_dict["experience"] = data_decision["experience"].values
    states_dict["wealth"] = data_decision["adjusted_wealth"].values

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
        choice_probs_observations = choice_probs_for_choice_vals(
            choice_vals,
            states_dict,
            model,
            unobserved_state_specs,
            params,
            est_model,
            weight_full_states=False,
        )

        choice_probs_observations = np.nan_to_num(choice_probs_observations, nan=0.0)
        data_decision[f"choice_{choice}"] = choice_probs_observations

    # prob_choice_observed = choice_probs_for_choice_vals(
    #     data_decision["choice"].values,
    #     states_dict,
    #     model,
    #     unobserved_state_specs,
    #     params,
    #     est_model,
    #     weight_full_states=False,
    # )

    file_append = ["low", "high"]
    data_decision["married"] = (data_decision["partner_state"] > 0).astype(int)

    # for partner_val, partner_label in enumerate(partner_labels):
    for edu in range(2):
        data_subset = data_decision[
            (data_decision["education"] == edu)
            # & (data_decision["married"] == partner_val)
        ]
        choice_shares_obs = (
            data_subset.groupby(["age"])["choice"]
            .value_counts(normalize=True)
            .unstack()
        )

        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        labels = ["Unemployment", "Employment", "Retirement"]
        for choice, ax in enumerate(axes):
            choice_shares_predicted = data_subset.groupby(["age"])[
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
        # Fig title
        fig.tight_layout()
        fig.savefig(
            paths_dict["plots"] + f"observed_model_fit_{file_append[edu]}.png",
            transparent=True,
            dpi=300,
        )
        fig.suptitle(f"Choice shares {specs['education_labels'][edu]}")


def choice_probs_for_choice_vals(
    choice_vals,
    states_dict,
    model,
    unobserved_state_specs,
    params,
    est_model,
    weight_full_states,
):
    choice_prob_func = create_choice_prob_func_unobserved_states(
        model=model,
        observed_states=states_dict,
        observed_choices=choice_vals,
        unobserved_state_specs=unobserved_state_specs,
        weight_full_states=weight_full_states,
    )

    choice_probs_observations = choice_prob_func(
        value_in=est_model["value"],
        endog_grid_in=est_model["endog_grid"],
        params_in=params,
    )
    return choice_probs_observations
