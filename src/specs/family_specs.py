import numpy as np
import pandas as pd
from jax import numpy as jnp
from specs.derive_specs import read_and_derive_specs


def calculate_partner_hrly_wage(path_dict):
    """Calculates average hourly wage of working partners (i.e. conditional on working
    hours > 0).

    Produces partner_hrly_wage array of shape (n_sexes, n_education_types,
    n_working_periods)

    """
    specs = read_and_derive_specs(path_dict["specs"])
    start_age = specs["start_age"]
    end_age = specs["max_ret_age"]
    ages = np.arange(start_age, end_age + 1)
    n_edu_types = specs["n_education_types"]

    # wage equation: ln(partner_wage(sex, edu)) = constant(sex, edu) + beta(sex, edu) * ln(age)
    partner_wage_params_men = pd.read_csv(
        path_dict["est_results"] + "partner_wage_eq_params_men.csv"
    )
    partner_wage_params_women = pd.read_csv(
        path_dict["est_results"] + "partner_wage_eq_params_women.csv"
    )
    partner_wage_params_men["sex"] = 0
    partner_wage_params_women["sex"] = 1
    partner_wage_params = pd.concat(
        [partner_wage_params_men, partner_wage_params_women]
    )
    partner_wage_params = partner_wage_params.rename(
        columns={partner_wage_params.columns[0]: "edu"}
    )

    partner_wages = np.zeros((2, n_edu_types, len(ages)))

    for edu in np.arange(0, n_edu_types):
        for sex in [0, 1]:
            beta_0 = partner_wage_params.loc[
                (partner_wage_params["edu"] == edu)
                & (partner_wage_params["sex"] == sex),
                "constant",
            ].values[0]
            beta_1 = partner_wage_params.loc[
                (partner_wage_params["edu"] == edu)
                & (partner_wage_params["sex"] == sex),
                "ln_age",
            ].values[0]
            partner_wages[sex, edu] = np.exp(beta_0 + beta_1 * np.log(ages))

    return jnp.asarray(partner_wages)


def calculate_partner_hours(path_dict):
    """Calculates average hours worked by working partners (i.e. conditional on working
    hours > 0) Produces partner_hours array of shape (n_sexes, n_education_types,
    n_working_periods)"""
    specs = read_and_derive_specs(path_dict["specs"])
    start_age = specs["start_age"]
    end_age = specs["max_ret_age"]
    # load data
    partner_hours = pd.read_csv(
        path_dict["est_results"] + "partner_hours.csv",
        index_col=[0, 1, 2],
        dtype={"sex": int, "education": int, "age_bin": int},
    )
    # populate numpy ndarray which maps state to average hours worked by partner
    partner_hours_np = np.zeros(
        (2, specs["n_education_types"], end_age - start_age + 1)
    )
    for sex in [0, 1]:
        for edu in range(specs["n_education_types"]):
            for t in range(end_age - start_age + 1):
                if t + start_age >= 60:
                    age_bin = 60
                else:
                    age_bin = int(np.floor((t + start_age) / 10) * 10)
                partner_hours_np[sex, edu, t] = partner_hours.loc[
                    (sex, edu, age_bin), "working_hours_p"
                ]
    return jnp.asarray(partner_hours_np)


def predict_children_by_state(path_dict):
    """Predicts the number of children in the household for each individual conditional
    on state.

    Produces children array of shape (n_sexes, n_education_types, n_has_partner_states,
    n_periods)

    """
    specs = read_and_derive_specs(path_dict["specs"])
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
