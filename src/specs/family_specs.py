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
        path_dict["first_step_results"] + "nb_children_estimates.csv",
        index_col=[0, 1, 2],
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
    return children


def read_in_partner_transition_specs(paths_dict, specs):
    est_probs_df = pd.read_csv(
        paths_dict["first_step_results"] + "partner_transition_matrix.csv",
    )

    n_periods_transition = specs["n_periods"] - 1
    n_partner_states = specs["n_partner_states"]
    n_edu_types = specs["n_education_types"]
    n_sexes = specs["n_sexes"]

    # Transition probalities for partner
    trans_probs = np.zeros(
        (
            n_sexes,
            n_edu_types,
            n_periods_transition,
            n_partner_states,
            n_partner_states,
        ),
        dtype=float,
    )

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            for period in range(n_periods_transition):
                age = period + specs["start_age"]
                for partner_state_var, partner_state_label in enumerate(
                    specs["partner_labels"]
                ):
                    for lead_partner_state_var, lead_partner_state_label in enumerate(
                        specs["partner_labels"]
                    ):
                        mask = (
                            (est_probs_df["sex"] == sex_label)
                            & (est_probs_df["education"] == edu_label)
                            & (est_probs_df["age"] == age)
                            & (est_probs_df["partner_state"] == partner_state_label)
                            & (
                                est_probs_df["lead_partner_state"]
                                == lead_partner_state_label
                            )
                        )
                        trans_probs[
                            sex_var,
                            edu_var,
                            period,
                            partner_state_var,
                            lead_partner_state_var,
                        ] = est_probs_df.loc[mask, "proportion"].values[0]
                    # # Assign absorbing 1 if no one in the data
                    # if not np.allclose(
                    #     trans_probs[sex_var, edu_var, period, partner_state_var].sum(),
                    #     1,
                    # ):
                    #     raise ValueError("This should happen in a parametric world")

    return trans_probs, n_partner_states
