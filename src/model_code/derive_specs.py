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
    specs["partner_hrly_wage"] = calculate_partner_wage(path_dict)
    specs["partner_hours"] = calculate_partner_hours(path_dict)
    specs["partner_pension"] = calculate_partner_pension(path_dict)

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
    
    specs = read_and_derive_specs(path_dict["specs"])
    start_age = specs["start_age"]
    end_age = specs["end_age"]

    # wage equation: ln(partner_wage(sex, edu)) = constant(sex, edu) + beta(sex, edu) * ln(age)
    partner_wage_params_men = pd.read_pickle(
        path_dict["intermediate_data"] + "partner_wage_estimation_sample_men.pkl"
    )
    partner_wage_params_women = pd.read_pickle(
        path_dict["intermediate_data"] + "partner_wage_estimation_sample_women.pkl"
    )
    ages = np.arange(start_age, end_age + 1)
    
    partner_wages_men = np.zeros((2, len(ages)))
    partner_wages_women = np.zeros(2, len(ages))
    for edu in [0, 1]:
        partner_wages_men[edu] = np.exp(
            partner_wage_params_men["constant"].values[edu] + partner_wage_params_men["ln_age"].values[edu] * np.log(ages)
        )
        partner_wages_women[edu] = np.exp(
            partner_wages_women["constant"].values[edu] + partner_wages_women["ln_age"].values[edu] * np.log(ages)
        )
    
    partner_wages = np.concatenate(partner_wages_men, partner_wages_women)
    return jnp.asarray(partner_wages)

def calculate_partner_hours(path_dict):
    """Calculate average hours worked by *working* partners (i.e. conditional on working hours > 0)"""
    # load data
    df = pd.read_pickle(path_dict["intermediate_data"] + "partner_wage_estimation_sample.pkl")
    
    # calculate average hours worked by partner by age, sex and education
    cov_list = ["sex", "education", "age"]
    partner_hours = df.groupby(cov_list)["working_hours_p"].mean()
    
    return jnp.asarray(partner_hours)    




