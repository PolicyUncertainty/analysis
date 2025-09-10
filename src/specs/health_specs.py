import numpy as np
import pandas as pd
from jax import numpy as jnp


def read_in_health_transition_specs(paths_dict, specs):
    trans_probs_df = pd.read_csv(
        paths_dict["first_step_results"] + "health_transition_matrix.csv",
    )

    death_prob_df = pd.read_csv(
        paths_dict["first_step_results"] + "mortality_transition_matrix.csv",
    )

    observed_health_vars = specs["observed_health_vars"]
    good_health_var = specs["good_health_var"]
    bad_health_var = specs["bad_health_var"]
    disabled_health_var = specs["disabled_health_var"]
    death_health_var = specs["death_health_var"]

    # Transition probalities for health
    health_trans_mat = np.zeros(
        (
            specs["n_sexes"],
            specs["n_education_types"],
            specs["n_periods"],
            specs["n_all_health_states"],
            specs["n_all_health_states"],
        ),
        dtype=float,
    )
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            for period in range(specs["n_periods"]):
                for current_health_var in observed_health_vars:
                    for lead_health_var in observed_health_vars:
                        current_health_label = specs["health_labels"][
                            current_health_var
                        ]
                        next_health_label = specs["health_labels"][lead_health_var]
                        trans_prob = trans_probs_df.loc[
                            (trans_probs_df["sex"] == sex_label)
                            & (trans_probs_df["education"] == edu_label)
                            & (trans_probs_df["period"] == period)
                            & (trans_probs_df["health"] == current_health_label)
                            & (trans_probs_df["lead_health"] == next_health_label),
                            "transition_prob",
                        ].values[0]
                        health_trans_mat[
                            sex_var,
                            edu_var,
                            period,
                            current_health_var,
                            lead_health_var,
                        ] = trans_prob

                    current_age = period + specs["start_age"]
                    # This needs to become label based
                    death_prob = death_prob_df.loc[
                        (death_prob_df["age"] == current_age)
                        & (death_prob_df["sex"] == sex_var)
                        & (death_prob_df["health"] == current_health_var)
                        & (death_prob_df["education"] == edu_var),
                        "death_prob",
                    ].values[0]
                    # Death state. Condition health transitions on surviving and then assign death
                    # probability to death state
                    health_trans_mat[
                        sex_var, edu_var, period, current_health_var, :
                    ] *= (1 - death_prob)
                    health_trans_mat[
                        sex_var, edu_var, period, current_health_var, death_health_var
                    ] = death_prob

    # Death as absorbing state. There are only zeros in the last row of the
    # transition matrix and a 1 on the diagonal element
    health_trans_mat[:, :, :, death_health_var, death_health_var] = 1

    # Now assign probabilities to disabled state. It will be scaled with the
    # real probabilities in params later. Here we assign the same probability row
    # as the bad health state
    health_trans_mat[:, :, :, disabled_health_var, :] = health_trans_mat[
        :, :, :, bad_health_var, :
    ]

    return jnp.asarray(health_trans_mat)


def process_health_labels(specs):
    # For health states, get number and var values for alive states
    specs["n_all_health_states"] = len(specs["health_labels"])
    specs["n_observed_health_states"] = len(specs["observed_health_labels"])
    # Read out vars, as we need those also inside the model
    specs["observed_health_vars"] = np.where(
        np.isin(specs["health_labels"], specs["observed_health_labels"])
    )[0]

    specs["good_health_var"] = np.where(
        np.array(specs["health_labels"]) == "Good Health"
    )[0][0]

    specs["bad_health_var"] = np.where(
        np.array(specs["health_labels"]) == "Bad Health"
    )[0][0]
    specs["disabled_health_var"] = np.where(
        np.array(specs["health_labels"]) == "Disabled"
    )[0][0]
    specs["death_health_var"] = np.where(np.array(specs["health_labels"]) == "Death")[
        0
    ][0]
    return specs
