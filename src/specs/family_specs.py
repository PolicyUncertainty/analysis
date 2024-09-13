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
    return partner_wages, partner_pension


# def  calculate_partner_hrly_wage(path_dict, specs):
#     """Calculates average hourly wage of working partners (i.e. conditional on working
#     hours > 0).
#
#     Produces partner_hrly_wage array of shape (n_sexes, n_education_types,
#     n_working_periods)
#
#     """
#     start_age = specs["start_age"]
#     end_age = specs["max_ret_age"]
#     ages = np.arange(start_age, end_age + 1)
#     n_edu_types = specs["n_education_types"]
#
#     # wage equation: ln(partner_wage(sex, edu)) = constant(sex, edu) + beta(sex, edu) * ln(age)
#     partner_wage_params_men = pd.read_csv(
#         path_dict["est_results"] + "partner_wage_eq_params_men.csv"
#     )
#     partner_wage_params_women = pd.read_csv(
#         path_dict["est_results"] + "partner_wage_eq_params_women.csv"
#     )
#     partner_wage_params_men["sex"] = 0
#     partner_wage_params_women["sex"] = 1
#     partner_wage_params = pd.concat(
#         [partner_wage_params_men, partner_wage_params_women]
#     )
#     partner_wage_params = partner_wage_params.rename(
#         columns={partner_wage_params.columns[0]: "edu"}
#     )
#
#     partner_wages = np.zeros((2, n_edu_types, len(ages)))
#
#     for edu in np.arange(0, n_edu_types):
#         for sex in [0, 1]:
#             beta_0 = partner_wage_params.loc[
#                 (partner_wage_params["edu"] == edu)
#                 & (partner_wage_params["sex"] == sex),
#                 "constant",
#             ].values[0]
#             beta_1 = partner_wage_params.loc[
#                 (partner_wage_params["edu"] == edu)
#                 & (partner_wage_params["sex"] == sex),
#                 "ln_age",
#             ].values[0]
#             partner_wages[sex, edu] = np.exp(beta_0 + beta_1 * np.log(ages))
#     import matplotlib.pyplot as plt
#     breakpoint()
#
#     return jnp.asarray(partner_wages)
#
#
# def calculate_partner_hours(path_dict, specs):
#     """Calculates average hours worked by working partners (i.e. conditional on working
#     hours > 0) Produces partner_hours array of shape (n_sexes, n_education_types,
#     n_working_periods)"""
#     start_age = specs["start_age"]
#     end_age = specs["max_ret_age"]
#     # load data
#     partner_hours = pd.read_csv(
#         path_dict["est_results"] + "partner_hours.csv",
#         index_col=[0, 1, 2],
#         dtype={"sex": int, "education": int, "age_bin": int},
#     )
#     # populate numpy ndarray which maps state to average hours worked by partner
#     partner_hours_np = np.zeros(
#         (2, specs["n_education_types"], end_age - start_age + 1)
#     )
#     for sex in [0, 1]:
#         for edu in range(specs["n_education_types"]):
#             for t in range(end_age - start_age + 1):
#                 if t + start_age >= 60:
#                     age_bin = 60
#                 else:
#                     age_bin = int(np.floor((t + start_age) / 10) * 10)
#                 partner_hours_np[sex, edu, t] = partner_hours.loc[
#                     (sex, edu, age_bin), "working_hours_p"
#                 ]
#     return jnp.asarray(partner_hours_np)


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
                for t in range(n_periods):
                    predicted_nb_children = (
                        params.loc[(sex, edu, has_partner), "const"]
                        + params.loc[(sex, edu, has_partner), "period"] * t
                        + params.loc[(sex, edu, has_partner), "period_sq"] * t**2
                    )
                    children[sex, edu, has_partner, t] = np.maximum(
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

    # Save in specs
    specs["partner_trans_mat"] = jnp.asarray(male_trans_probs)
    specs["n_partner_states"] = n_partner_states
    return specs
