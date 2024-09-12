import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml


def generate_specs_and_update_params(path_dict, start_params):
    specs = generate_derived_and_data_derived_specs(path_dict)
    # Assign income shock scale to start_params_all
    start_params["sigma"] = specs["income_shock_scale"]
    start_params["interest_rate"] = specs["interest_rate"]
    start_params["beta"] = specs["discount_factor"]
    return specs, start_params


def generate_derived_and_data_derived_specs(path_dict, load_precomputed=False):
    specs = read_and_derive_specs(path_dict["specs"])

    # wages
    wage_params = pd.read_csv(
        path_dict["est_results"] + "wage_eq_params.csv", index_col=0
    )

    specs["gamma_0"] = jnp.asarray(wage_params["constant"].values)
    specs["gamma_1"] = jnp.asarray(wage_params["ln_exp"].values)
    specs["income_shock_scale"] = wage_params["income_shock_std"].values.mean()

    # pensions
    specs["pension_point_value_by_edu_exp"] = calculate_pension_values(specs, path_dict)

    # partner income
    specs["partner_hrly_wage"] = calculate_partner_hrly_wage(path_dict)
    specs["partner_hours"] = calculate_partner_hours(path_dict)
    # specs["partner_pension"] = calculate_partner_pension(path_dict)

    # family transitions
    specs["children_by_state"] = predict_children_by_state(path_dict)

    # Set initial experience
    specs["max_init_experience"] = create_initial_exp(path_dict, load_precomputed)

    specs["job_sep_probs"] = jnp.asarray(
        np.loadtxt(path_dict["est_results"] + "job_sep_probs.csv", delimiter=",")
    )
    return specs


def create_initial_exp(path_dict, load_precomputed):
    # Initial experience
    if load_precomputed:
        max_init_experience = int(
            np.loadtxt(path_dict["intermediate_data"] + "max_init_exp.txt")
        )
    else:
        # max initial experience
        data_decision = pd.read_pickle(
            path_dict["intermediate_data"] + "structural_estimation_sample.pkl"
        )
        max_init_experience = (
            data_decision["experience"] - data_decision["period"]
        ).max()
        np.savetxt(
            path_dict["intermediate_data"] + "max_init_exp.txt", [max_init_experience]
        )
    return max_init_experience


def read_and_derive_specs(spec_path):
    specs = yaml.safe_load(open(spec_path))

    # Number of periods in model
    specs["n_periods"] = specs["end_age"] - specs["start_age"] + 1
    # you can retire from min retirement age until max retirement age
    specs["n_possible_ret_ages"] = specs["max_ret_age"] - specs["min_ret_age"] + 1
    specs["n_policy_states"] = int(
        ((specs["max_SRA"] - specs["min_SRA"]) / specs["SRA_grid_size"]) + 1
    )
    specs["SRA_values_policy_states"] = np.arange(
        specs["min_SRA"],
        specs["max_SRA"] + specs["SRA_grid_size"],
        specs["SRA_grid_size"],
    )

    return specs


def calculate_pension_values(specs, path_dict):
    wage_params = pd.read_csv(
        path_dict["est_results"] + "wage_eq_params.csv", index_col=0
    )
    wage_params_full_sample = pd.read_csv(
        path_dict["est_results"] + "wage_eq_params_full_sample.csv", index_col=0
    )

    experience = np.arange(0, specs["exp_cap"] + 1)
    wage_by_experience_average = np.exp(
        wage_params_full_sample.loc["constant"].values
        + wage_params_full_sample.loc["ln_exp"].values * np.log(experience + 1)
    )
    # if number of education types changes, this needs to be adjusted
    wage_by_experience = np.ndarray(shape=(2, len(experience)))
    adjustment_factor_by_exp = np.ndarray(shape=(2, len(experience)))
    for education in [0, 1]:
        wage_by_experience[education] = np.exp(
            wage_params.loc[education, "constant"]
            + wage_params.loc[education, "ln_exp"] * np.log(experience + 1)
        )
        adjustment_factor_by_exp[education] = (
            wage_by_experience[education] / wage_by_experience_average
        )
        for i in range(1, len(experience)):
            adjustment_factor_by_exp[education, i] = adjustment_factor_by_exp[
                education, 1 : i + 1
            ].mean()

    # Generate average pension point value weighted by east and west
    # pensions
    pension_point_value = (
        0.75 * specs["pension_point_value_west_2010"]
        + 0.25 * specs["pension_point_value_east_2010"]
    ) / specs["wealth_unit"]
    return jnp.asarray(adjustment_factor_by_exp) * pension_point_value


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
