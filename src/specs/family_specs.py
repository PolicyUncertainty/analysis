import numpy as np
import pandas as pd
from jax import numpy as jnp


def predict_children_by_state(path_dict, specs):
    """Predicts the number of children in the household for each individual conditional
    on state.

    Produces children array of shape (n_sexes, n_education_types, n_has_partner_states,
    n_periods)

    """
    n_periods = specs["end_age"] - specs["start_age"] + 1
    params = pd.read_csv(
        path_dict["est_results"] + "nb_children_estimates.csv", index_col=[0, 1, 2]
    )
    # populate numpy ndarray which maps state to average number of children
    children = np.zeros((2, specs["n_education_types"], 2, n_periods))
    for sex in [0, 1]:
        for edu in range(specs["n_education_types"]):
            for has_partner in [0, 1]:
                for period in range(n_periods):
                    predicted_nb_children = (
                        params.loc[(sex, edu, has_partner), "const"]
                        + params.loc[(sex, edu, has_partner), "period"] * period
                        + params.loc[(sex, edu, has_partner), "period_sq"] * period**2
                    )
                    children[sex, edu, has_partner, period] = np.maximum(
                        0, predicted_nb_children
                    )
    return jnp.asarray(children)


def read_in_partner_transition_specs(paths_dict, specs):
    trans_probs = pd.read_csv(
        paths_dict["est_results"] + "partner_transition_matrix.csv",
        index_col=[0, 1, 2, 3, 4],
    )["proportion"]

    n_periods = specs["n_periods"]
    n_partner_states = trans_probs.index.get_level_values("partner_state").nunique()
    n_edu_types = len(specs["education_labels"])
    n_sexes = 2
    age_bins = trans_probs.index.get_level_values("age_bin").unique()

    full_series_index = pd.MultiIndex.from_product(
        [
            range(n_edu_types),
            range(n_sexes),
            age_bins,
            range(n_partner_states),
            range(n_partner_states),
        ],
        names=trans_probs.index.names,
    )
    full_series = pd.Series(index=full_series_index, data=0.0, name=trans_probs.name)
    full_series.update(trans_probs)

    # Transition probalities for partner
    trans_probs = np.zeros(
        (n_edu_types, n_sexes, n_periods, n_partner_states, n_partner_states), dtype=float
    )

    for edu in range(n_edu_types):
        for period in range(n_periods):
            for current_state in range(n_partner_states):
                for next_state in range(n_partner_states):
                    for sex in range(n_sexes):
                        age = period + specs["start_age"]
                        # Check if age is in between 30 and 40, 40 and 50, 50 and 60, 60 and 70
                        age_bin = np.floor(age / 10) * 10
                        trans_probs[
                            edu, sex, period, current_state, next_state
                        ] = full_series.loc[(sex, edu, age_bin, current_state, next_state)]

    return jnp.asarray(trans_probs), n_partner_states
