import numpy as np
import pandas as pd
from jax import numpy as jnp


def read_in_health_transition_specs(paths_dict, specs):
    trans_probs_df = pd.read_csv(
        paths_dict["est_results"] + "health_transition_matrix.csv",
        index_col=[0, 1, 2, 3],
    )["transition_prob"]

    death_probs = pd.read_csv(
        paths_dict["est_results"] + "mortality_transition_matrix.csv",
        index_col=[0, 1, 2, 3],
    )["death_prob"]
    n_periods = specs["n_periods"]
    n_health_states = specs["n_health_states"]
    n_edu_types = len(specs["education_labels"])

    alive_health_states = np.where(np.array(specs["health_labels"]) != "Death")[0]
    death_health_state = np.where(np.array(specs["health_labels"]) == "Death")[0][0]

    # Transition probalities for health
    trans_probs = np.zeros(
        (n_edu_types, n_periods, n_health_states, n_health_states), dtype=float
    )

    for edu in range(n_edu_types):
        for period in range(n_periods):
            for current_state in alive_health_states:
                for next_state in alive_health_states:
                    trans_probs[
                        edu, period, current_state, next_state
                    ] = trans_probs_df.loc[(edu, period, current_state, next_state)]

                current_age = period + specs["start_age"]
                period_mortality_est = current_age - specs["start_age_mortality"]
                death_prob = death_probs.loc[
                    (period_mortality_est, 0, current_state, edu)
                ]
                # Death state. Condition health transitions on surviving and then assign death
                # probability to death state
                trans_probs[edu, period, current_state, :] *= 1 - death_prob
                trans_probs[edu, period, current_state, death_health_state] = death_prob

    # Death as absorbing state. There are only zeros in the last row of the
    # transition matrix and a 1 on the diagonal element
    trans_probs[:, :, death_health_state, death_health_state] = 1

    return jnp.asarray(trans_probs)
