import numpy as np
import pandas as pd
from jax import numpy as jnp


def read_in_health_transition_specs(paths_dict, specs):
    trans_probs_df = pd.read_csv(
        paths_dict["est_results"] + "health_transition_matrix.csv",
        index_col=[0, 1, 2, 3],
    )["transition_prob"]
    n_periods = specs["n_periods"]
    n_health_states = trans_probs_df.index.get_level_values("health_state").nunique()
    n_edu_types = len(specs["education_labels"])

    # Transition probalities for health
    trans_probs = np.zeros(
        (n_edu_types, n_periods, n_health_states, n_health_states), dtype=float
    )

    for edu in range(n_edu_types):
        for period in range(n_periods):
            for current_state in range(n_health_states):
                for next_state in range(n_health_states):
                    trans_probs[
                        edu, period, current_state, next_state
                    ] = trans_probs_df.loc[(edu, period, current_state, next_state)]

    return jnp.asarray(trans_probs), n_health_states
