import numpy as np
import pandas as pd


def add_income_specs(specs, path_dict):
    # wages
    specs = add_wage_specs(path_dict, specs)

    specs = add_population_averages(specs, path_dict)

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
    pop_averages = pd.read_csv(
        path_dict["first_step_incomes"] + "population_averages_working_hours.csv"
    )
    av_annual_hours_pt = np.zeros(
        (specs["n_sexes"], specs["n_education_types"]), dtype=float
    )
    av_annual_hours_ft = np.zeros(
        (specs["n_sexes"], specs["n_education_types"]), dtype=float
    )

    for edu_var, edu_label in enumerate(specs["education_labels"]):
        for sex_var, sex_label in enumerate(specs["sex_labels"]):
            mask = (pop_averages["education"] == edu_var) & (
                pop_averages["sex"] == sex_var
            )
            # We assign for men and women, but men-part-time is never used.
            av_annual_hours_pt[sex_var, edu_var] = pop_averages.loc[
                mask & (pop_averages["choice"] == 2), "annual_hours"
            ].values[0]

            av_annual_hours_ft[sex_var, edu_var] = pop_averages.loc[
                mask & (pop_averages["choice"] == 3), "annual_hours"
            ].values[0]

    specs["av_annual_hours_pt"] = av_annual_hours_pt
    specs["av_annual_hours_ft"] = av_annual_hours_ft

    # Create auxiliary mean hourly full time wage for pension calculation (see appendix)
    mean_annual_wage = np.loadtxt(
        path_dict["first_step_incomes"] + "pop_avg_annual_wage.txt"
    )
    specs["mean_hourly_ft_wage"] = mean_annual_wage / av_annual_hours_ft
    return specs


def add_pt_and_ft_min_wage(specs):
    """Computes the annual minimum wage for part-time and full-time workers.

    Type-specific as hours are different between types.

    """
    annual_min_wage_pt = np.zeros((specs["n_sexes"], specs["n_education_types"]))

    for edu in range(specs["n_education_types"]):
        for sex in [0, 1]:
            hours_ratio = (
                specs["av_annual_hours_pt"][sex, edu]
                / specs["av_annual_hours_ft"][sex, edu]
            )
            annual_min_wage_pt[sex, edu] = specs["monthly_min_wage"] * hours_ratio * 12

    return annual_min_wage_pt, specs["monthly_min_wage"] * 12


def calc_annual_pension_point_value(specs):
    # Generate average pension point value weighted by east and west
    # pensions
    pension_point_value = (
        specs["pop_share_west"] * specs["monthly_pension_point_value_west_2020"]
        + specs["pop_share_east"] * specs["monthly_pension_point_value_east_2020"]
    )
    return pension_point_value * 12


def add_wage_specs(path_dict, specs):
    # wages
    wage_params = pd.read_csv(
        path_dict["first_step_incomes"] + "wage_eq_params.csv", index_col=0
    )

    wage_params.reset_index(inplace=True)

    gamma_0 = np.zeros((specs["n_sexes"], specs["n_education_types"]), dtype=float)
    gamma_ln_exp = np.zeros((specs["n_sexes"], specs["n_education_types"]), dtype=float)
    gamma_above_50 = np.zeros(
        (specs["n_sexes"], specs["n_education_types"]), dtype=float
    )

    for edu_id, edu_label in enumerate(specs["education_labels"]):
        for sex_var, sex in enumerate(specs["sex_labels"]):
            mask = (wage_params["education"] == edu_label) & (wage_params["sex"] == sex)
            gamma_0[sex_var, edu_id] = wage_params.loc[
                mask & (wage_params["parameter"] == "constant"), "value"
            ].values[0]
            gamma_ln_exp[sex_var, edu_id] = wage_params.loc[
                mask & (wage_params["parameter"] == "ln_exp"), "value"
            ].values[0]
            gamma_above_50[sex_var, edu_id] = wage_params.loc[
                mask & (wage_params["parameter"] == "above_50_age"), "value"
            ].values[0]

    # Assign to specs
    specs["gamma_0"] = gamma_0
    specs["gamma_ln_exp"] = gamma_ln_exp
    specs["gamma_above_50"] = gamma_above_50

    # Now read out params for all for pension calculation
    all_mask = (wage_params["education"] == "all") & (wage_params["sex"] == "all")
    income_shock_scale_all = wage_params.loc[
        all_mask & (wage_params["parameter"] == "income_shock_std"), "value"
    ].values[0]

    specs["income_shock_std"] = income_shock_scale_all
    return specs


def calculate_partner_incomes(path_dict, specs):
    """Calculate income of working aged partner."""
    periods = np.arange(0, specs["n_periods"], dtype=float)
    # Limit periods to the one of max retirement age as we restricted our estimation sample
    # until then. For the predictions after max retirement age we use the last max retirement period
    not_predicted_periods = np.where(
        periods > (specs["max_age_partner_working"] - specs["start_age"])
    )[0]
    periods[not_predicted_periods] = (
        specs["max_age_partner_working"] - specs["start_age"]
    )

    partner_wages = np.zeros(
        (specs["n_sexes"], specs["n_education_types"], specs["n_periods"]), dtype=float
    )
    sex_param_labels = ["men", "women"]
    for edu_var, edu_label in enumerate(specs["education_labels"]):
        for sex_var, sex_label in enumerate(sex_param_labels):
            params = pd.read_csv(
                path_dict["first_step_incomes"]
                + f"partner_wage_eq_params_{sex_label}.csv"
            )
            mask = params["education"] == edu_label
            partner_wages[sex_var, edu_var, :] = (
                params.loc[mask, "constant"].values[0]
                + params.loc[mask, "period"].values[0] * periods
                + params.loc[mask, "period_sq"].values[0] * periods**2
            )
            # Additional cubic term
            if sex_label == "men":
                partner_wages[sex_var, edu_var, :] += (
                    params.loc[mask, "period_cub"].values[0] * periods**3
                )

    # annual partner wage
    annual_partner_wages = partner_wages * 12

    # Average pension 2020 of men and women
    # Source: https://statistik-rente.de/drv/extern/publikationen
    # /statistikbaende/documents/Rente_2020.pdf
    # annual_partner_pension = np.array([[800, 800], [1227, 1227]]) * 12
    # Below is soep estimate
    annual_partner_pension = np.zeros((2, 2))
    partner_pension = pd.read_csv(
        path_dict["first_step_incomes"] + "partner_pension.csv"
    )
    for sex in range(2):
        for edu in range(2):
            mask = (partner_pension["sex"] == sex) & (
                partner_pension["education"] == edu
            )
            annual_partner_pension[sex, edu] = partner_pension.loc[
                mask, "public_pension_p"
            ].values[0]

    return annual_partner_wages, annual_partner_pension
