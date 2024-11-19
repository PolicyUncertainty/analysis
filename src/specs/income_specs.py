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
        specs["yearly_unemployment_benefits"],
        specs["yearly_unemployment_benefits_housing"],
        specs["yearly_child_unemployment_benefits"],
    ) = calc_yearly_unemployment_benefits(specs)

    specs["yearly_child_benefits"] = specs["monthly_child_benefits"] * 12

    # pensions
    specs["yearly_pension_point_value"] = calc_yearly_pension_point_value(specs)

    # partner income
    (
        specs["yearly_partner_wage"],
        specs["yearly_partner_pension"],
    ) = calculate_partner_incomes(path_dict, specs)

    specs = add_population_averages(specs, path_dict)

    # Add minimum wage
    specs["yearly_min_wage_pt"], specs["yearly_min_wage_ft"] = add_pt_and_ft_min_wage(
        specs
    )
    return specs


def calc_yearly_unemployment_benefits(specs):
    yearly_unemployment_benefits = specs["monthly_unemployment_benefits"] * 12
    yearly_unemployment_benefits_housing = (
        specs["monthly_unemployment_benefits_housing"] * 12
    )
    yearly_child_unemployment_benefits = (
        specs["monthly_child_unemployment_benefits"] * 12
    )
    return (
        yearly_unemployment_benefits,
        yearly_unemployment_benefits_housing,
        yearly_child_unemployment_benefits,
    )


def add_population_averages(specs, path_dict):
    # Assign population averages
    pop_averages = pd.read_csv(
        path_dict["est_results"] + "population_averages_working.csv"
    ).iloc[0]
    specs["av_annual_hours_ft"] = jnp.array(
        [
            pop_averages["annual_hours_low_ft_work"],
            pop_averages["annual_hours_high_ft_work"],
        ]
    )
    specs["av_annual_hours_pt"] = jnp.array(
        [
            pop_averages["annual_hours_low_pt_work"],
            pop_averages["annual_hours_high_pt_work"],
        ]
    )

    mean_wage_edu = pop_averages["annual_mean_wage"] / specs["av_annual_hours_ft"]
    specs["mean_hourly_ft_wage"] = mean_wage_edu
    return specs


def add_pt_and_ft_min_wage(specs):
    yearly_min_wage_pt = np.zeros(specs["n_education_types"])

    for edu in range(specs["n_education_types"]):
        hours_ratio = (
            specs["av_annual_hours_pt"][edu] / specs["av_annual_hours_ft"][edu]
        )
        yearly_min_wage_pt[edu] = specs["monthly_min_wage"] * hours_ratio * 12

    return jnp.asarray(yearly_min_wage_pt), specs["monthly_min_wage"] * 12


def calc_yearly_pension_point_value(specs):
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

    gamma_0 = jnp.asarray(wage_params.loc[edu_labels, "constant"].values)
    gamma_1 = jnp.asarray(wage_params.loc[edu_labels, "ln_exp"].values)
    income_shock_scale = wage_params.loc["all", "income_shock_std"]
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
    # Yearly partner wage
    partner_wages *= 12

    # Wealth hack
    yearly_partner_pension = partner_wages.mean(axis=1) * 0.48 * 12
    return jnp.asarray(partner_wages), jnp.asarray(yearly_partner_pension)
