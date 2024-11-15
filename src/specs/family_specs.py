import numpy as np
import pandas as pd
from jax import numpy as jnp


def calculate_partner_incomes(path_dict, specs):
    """Calculate income of working aged partner."""
    periods = np.arange(0, specs["n_periods"], dtype=int)
    n_edu_types = len(specs["education_labels"])

    # Only do this for men now
    partner_wage_params_men = pd.read_csv(
        path_dict["est_results"] + "partner_wage_eq_params_men.csv", index_col=0
    )
    # partner_wage_params_women = pd.read_csv(
    #     path_dict["est_results"] + "partner_wage_eq_params_women.csv"
    # )
    partner_wages = np.zeros((n_edu_types, specs["n_periods"]))
    for edu_val, edu_label in enumerate(specs["education_labels"]):
        for period in periods:
            partner_wages[edu_val, period] = (
                partner_wage_params_men.loc[edu_label, "constant"]
                + partner_wage_params_men.loc[edu_label, "period"] * period
                + partner_wage_params_men.loc[edu_label, "period_sq"] * period**2
            ) / specs["wealth_unit"]

    # Wealth hack
    partner_pension = partner_wages.mean(axis=1) * 0.48
    return jnp.asarray(partner_wages), jnp.asarray(partner_pension)


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
    sexes = [0, 1]
    age_bins = trans_probs.index.get_level_values("age_bin").unique()

    full_series_index = pd.MultiIndex.from_product(
        [
            sexes,
            range(n_edu_types),
            age_bins,
            range(n_partner_states),
            range(n_partner_states),
        ],
        names=trans_probs.index.names,
    )
    full_series = pd.Series(index=full_series_index, data=0.0, name=trans_probs.name)
    full_series.update(trans_probs)

    # Transition probalities for partner
    male_trans_probs = np.zeros(
        (n_edu_types, n_periods, n_partner_states, n_partner_states), dtype=float
    )

    for edu in range(n_edu_types):
        for period in range(n_periods):
            for current_state in range(n_partner_states):
                for next_state in range(n_partner_states):
                    age = period + specs["start_age"]
                    # Check if age is in between 30 and 40, 40 and 50, 50 and 60, 60 and 70
                    age_bin = np.floor(age / 10) * 10
                    male_trans_probs[
                        edu, period, current_state, next_state
                    ] = full_series.loc[(0, edu, age_bin, current_state, next_state)]

    return jnp.asarray(male_trans_probs), n_partner_states


def informed_state_transition_specs(paths, specs):
    informed_params = pd.read_pickle(
        paths["est_results"] + "uninformed_hazard_rate.pkl"
    )
    return informed_params
