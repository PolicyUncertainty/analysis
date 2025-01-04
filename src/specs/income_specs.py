import numpy as np
import pandas as pd
from jax import numpy as jnp


def add_income_specs(specs, path_dict):
    # wages
    (
        specs["gamma_0"],
        specs["gamma_1"],
        specs["income_shock_scale"],
    ) = process_wage_params(path_dict, specs)

    # unemployment benefits
    (
        specs["annual_unemployment_benefits"],
        specs["annual_unemployment_benefits_housing"],
        specs["annual_child_unemployment_benefits"],
    ) = calc_annual_unemployment_benefits(specs)

    specs["annual_child_benefits"] = specs["monthly_child_benefits"] * 12

    # pensions
    specs["annual_pension_point_value"] = calc_annual_pension_point_value(specs)

    # partner income
    (
        specs["annual_partner_wage"],
        specs["annual_partner_pension"],
    ) = calculate_partner_incomes(path_dict, specs)

    specs = add_population_averages(specs, path_dict)

    # Add minimum wage
    specs["annual_min_wage_pt"], specs["annual_min_wage_ft"] = add_pt_and_ft_min_wage(
        specs
    )
    return specs


def calc_annual_unemployment_benefits(specs):
    annual_unemployment_benefits = specs["monthly_unemployment_benefits"] * 12
    annual_unemployment_benefits_housing = (
        specs["monthly_unemployment_benefits_housing"] * 12
    )
    annual_child_unemployment_benefits = (
        specs["monthly_child_unemployment_benefits"] * 12
    )
    return (
        annual_unemployment_benefits,
        annual_unemployment_benefits_housing,
        annual_child_unemployment_benefits,
    )


def add_population_averages(specs, path_dict):
    # Assign population averages
    pop_averages = pd.read_csv(path_dict["est_results"] + "population_averages_working_hours.csv")
    labor_choices = [2,3]
    edu_labels = specs["education_labels"]
    sex_labels = specs["sex_labels"]
    avg_hours_by_type_choice = np.zeros((len(edu_labels), len(sex_labels), len(labor_choices)))
    for edu, edu_label in enumerate(edu_labels):
        for sex, sex_label in enumerate(sex_labels):
            for choice, choice_nb in enumerate(labor_choices):
                edu_mask = pop_averages["education"] == edu
                sex_mask = pop_averages["sex"] == sex
                choice_mask = pop_averages["choice"] == choice_nb
                pop_averages_mask = edu_mask & sex_mask & choice_mask
                avg_hours_by_type_choice[edu][sex][choice] = pop_averages[pop_averages_mask]["annual_hours"].values[0]
    specs["av_annual_hours_pt"] = avg_hours_by_type_choice[:][:][0]
    specs["av_annual_hours_ft"] = avg_hours_by_type_choice[:][:][1]

    mean_annual_wage = np.load(path_dict["est_results"] + "pop_avg_annual_wage.npy")
    specs["mean_hourly_ft_wage"] = mean_annual_wage / avg_hours_by_type_choice[:][:][1]
    return specs


def add_pt_and_ft_min_wage(specs):
    """ Computes the annual minimum wage for part-time and full-time workers. Type-specific as hours are different between types. """
    annual_min_wage_pt = np.zeros((specs["n_education_types"], 2))

    for edu in range(specs["n_education_types"]):
        for sex in [0, 1]:
            hours_ratio = (
                specs["av_annual_hours_pt"][edu][sex] / specs["av_annual_hours_ft"][edu][sex]
            )
            annual_min_wage_pt[edu][sex] = specs["monthly_min_wage"] * hours_ratio * 12

    return jnp.asarray(annual_min_wage_pt), specs["monthly_min_wage"] * 12


def calc_annual_pension_point_value(specs):
    # Generate average pension point value weighted by east and west
    # pensions
    pension_point_value = (
        0.75 * specs["monthly_pension_point_value_west_2010"]
        + 0.25 * specs["monthly_pension_point_value_east_2010"]
    )
    return pension_point_value * 12


def process_wage_params(path_dict, specs):
    # wages
    wage_params = pd.read_csv(
        path_dict["est_results"] + "wage_eq_params.csv", index_col=0
    )
    edu_labels = specs["education_labels"]
    sex_labels = specs["sex_labels"]
    wage_params.reset_index(inplace=True)
    
    gamma_0 = np.array([len(edu_labels), len(sex_labels)])
    gamma_1 = np.array([len(edu_labels), len(sex_labels)])

    for edu in edu_labels:
        for sex in sex_labels:
            mask = (wage_params["education"] == edu) & (wage_params["sex"] == sex)
            gamma_0 = wage_params[mask & (wage_params["parameter"] == "constant")]
            gamma_1 = wage_params[mask & (wage_params["parameter"] == "ln_exp")]
    mask = (wage_params["education"] == "all") & (wage_params["sex"] == "all") & (wage_params["parameter"] == "income_shock_std")
    income_shock_scale = wage_params[mask]["value"].values[0]
    return gamma_0, gamma_1, income_shock_scale


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
            )
    # annual partner wage
    annual_partner_wages = partner_wages * 12

    # Wealth hack
    annual_partner_pension = annual_partner_wages.mean(axis=1) * 0.48
    return jnp.asarray(annual_partner_wages), jnp.asarray(annual_partner_pension)
