# Description: This file estimates the parameters of the HOURLY wage equation using the SOEP panel data.
# We estimate the following equation for each education level:
# ln_wage = beta_0 + beta_1 * ln_(exp+1) + individual_FE + time_FE + epsilon
import numpy as np
import pandas as pd
from linearmodels.panel.model import PanelOLS
from scipy.stats import norm
from statsmodels.api import OLS
from statsmodels.discrete.discrete_model import Probit
from statsmodels.tools import add_constant


def estimate_wage_parameters(paths_dict, specs):
    """Estimate the wage parameters for each education group in the sample.

    Also estimate for all individuals.

    """
    # specs, data, and parameter containers
    edu_labels = specs["education_labels"]
    sex_labels = specs["sex_labels"]
    regressors = ["constant", "ln_exp", "above_50_age", "IMR"]
    wage_data_raw = load_and_prepare_wage_data(paths_dict)

    wage_data_raw = estimate_selection_correction_grouped(wage_data_raw, specs)

    invalid_mask = wage_data_raw[regressors + ["education", "sex"]].isnull().any(axis=1)
    wage_data = wage_data_raw.loc[~invalid_mask].copy()
    # Calculate average working hours by type and choice
    # # Map annual hours to data, for hours smaller and equal than 0
    # wage_data = add_missing_hours(wage_data, average_working_hours)
    wage_data = wage_data[wage_data["weekly_hours"] > 5].copy()

    # Restrict to maximum experience
    wage_data = wage_data[wage_data["experience"] <= specs["max_est_age_labor"] - 14]

    # Now everyone has correct monthly hours. We can define hourly wage
    wage_data["hourly_wage"] = wage_data["monthly_wage"] / wage_data["monthly_hours"]
    # We are 2013 onwards, so everybody must at least earn 8.5. This is to filter out false reporting.
    wage_data = wage_data[wage_data["hourly_wage"] > 8.5]
    wage_data["ln_wage"] = np.log(wage_data["hourly_wage"])

    # Restrict to relevant ages
    wage_data = wage_data[wage_data["age"] < specs["max_est_age_labor"]]

    # Initialize containers for wage parameters and year fixed effects
    wage_parameters, year_fixed_effects = initialize_coeficient_containers(
        regressors, specs
    )

    # Estimate wage equation for each type (sex x education)
    fit_panel_reg_model(
        wage_data_type=wage_data,
        regressors=regressors,
        wage_parameters=wage_parameters,
        year_fixed_effects=year_fixed_effects,
        edu_label="all",
        sex_label="all",
        specs=specs,
    )

    # Store all wage data with predictions for plotting
    all_wage_data_with_predictions = []

    for sex_val, sex_label in enumerate(sex_labels):
        for edu_val, edu_label in enumerate(edu_labels):
            wage_data_type = wage_data[
                (wage_data["education"] == edu_val) & (wage_data["sex"] == sex_val)
            ].copy()
            year_fixed_effects[edu_label, sex_label] = {}

            wage_parameters, year_fixed_effects, wage_data_type = fit_panel_reg_model(
                wage_data_type=wage_data_type,
                regressors=regressors,
                wage_parameters=wage_parameters,
                year_fixed_effects=year_fixed_effects,
                edu_label=edu_label,
                sex_label=sex_label,
                specs=specs,
            )

            # Store this data for plotting
            all_wage_data_with_predictions.append(wage_data_type)

    # Combine all wage data with predictions for plotting
    if all_wage_data_with_predictions:
        combined_wage_data = pd.concat(
            all_wage_data_with_predictions, ignore_index=True
        )
        # Save wage data with predictions for plotting
        combined_wage_data.to_csv(
            paths_dict["first_step_data"]
            + "wage_estimation_sample_with_predictions.csv"
        )

    # Save results
    wage_parameters.to_csv(paths_dict["first_step_incomes"] + "wage_eq_params.csv")
    pd.DataFrame(year_fixed_effects).T.to_csv(
        paths_dict["first_step_incomes"] + "wage_eq_year_FE.csv"
    )

    wage_parameters.T.to_latex(
        paths_dict["tables"] + "wage_eq_params.tex", float_format="%.4f"
    )
    # After estimation print some summary statistics
    # print_wage_equation(wage_parameters, edu_labels, sex_labels)
    print_lc_wage_and_exp(wage_data, edu_labels, sex_labels)
    calc_wage_population_averages(wage_data_raw, year_fixed_effects, specs, paths_dict)
    calc_population_working_hours(wage_data, paths_dict)

    return wage_parameters


def estimate_selection_correction_grouped(wage_data, specs):
    """Estimate Heckman selection correction separately for each sex x education group."""
    wage_data = wage_data.copy()
    wage_data["IMR"] = np.nan  # prepare column

    pred_vars = ["age", "health", "education", "married", "age_squared", "num_children"]

    invalid_mask = wage_data[pred_vars].isnull().any(axis=1)

    for sex_val, _ in enumerate(specs["sex_labels"]):
        mask = (wage_data["sex"] == sex_val) & ~invalid_mask
        group_data = wage_data.loc[mask].copy()

        y = (group_data["weekly_hours"] > 5).astype(int)
        X = add_constant(group_data[pred_vars]).astype(float)

        probit_model = Probit(y, X)
        probit_res = probit_model.fit(disp=False)

        xb = X @ probit_res.params  # Direct calculation instead of predict
        pdf_xb = norm.pdf(xb)
        cdf_xb = norm.cdf(xb)

        # Clip to avoid numerical issues
        cdf_xb = np.clip(cdf_xb, 1e-8, 1 - 1e-8)

        # Calculate IMR only for selected observations (monthly_hours > 0)
        imr = np.where(y == 1, pdf_xb / cdf_xb, np.nan)

        wage_data.loc[mask, "IMR"] = imr

    return wage_data


def load_and_prepare_wage_data(paths_dict):
    # Load wage data
    wage_data = pd.read_csv(
        paths_dict["first_step_data"] + "wage_estimation_sample.csv", index_col=0
    )
    wage_data["ln_exp"] = np.log(wage_data["experience"] + 1)
    wage_data["ln_exp_55"] = np.log(wage_data["experience"] + 1) * (
        wage_data["age"] >= 55
    )
    wage_data["above_50_age"] = (wage_data["age"] >= 50) * (wage_data["age"] - 50)
    wage_data["exp"] = wage_data["experience"]

    wage_data["age_squared"] = wage_data["age"] ** 2

    wage_data["exp_squared"] = wage_data["experience"] ** 2
    wage_data["constant"] = np.ones(len(wage_data))
    # Format & Index
    wage_data["year"] = wage_data["syear"].astype(int).astype("category")
    wage_data = wage_data.set_index(["pid", "syear"])
    return wage_data


def add_missing_hours(wage_data, average_working_hours):
    # Identify missing annual_hours
    missing_mask = wage_data["annual_hours"] < 0
    # Set index for mapping
    index_cols = ["education", "sex", "choice"]
    wage_data.loc[missing_mask, "annual_hours"] = wage_data.loc[
        missing_mask, index_cols
    ].apply(lambda row: average_working_hours.loc[tuple(row)], axis=1)
    wage_data.loc[missing_mask, "monthly_hours"] = (
        wage_data.loc[missing_mask, "annual_hours"] / 12
    )
    wage_data.loc[missing_mask, "weekly_hours"] = (
        wage_data.loc[missing_mask, "annual_hours"] / 52
    )
    return wage_data


def initialize_coeficient_containers(regressors, specs):
    coefficents = regressors + [param + "_ser" for param in regressors]
    index = pd.MultiIndex.from_product(
        [specs["education_labels"], specs["sex_labels"], coefficents],
        names=["education", "sex", "parameter"],
    )
    index_all_types = pd.MultiIndex.from_product(
        [["all"], ["all"], coefficents], names=["education", "sex", "parameter"]
    )
    index = index.append(index_all_types)
    wage_parameters = pd.DataFrame(index=index, columns=["value"])
    year_fixed_effects = {}
    year_fixed_effects["all", "all"] = {}
    return wage_parameters, year_fixed_effects


def fit_panel_reg_model(
    wage_data_type,
    regressors,
    wage_parameters,
    year_fixed_effects,
    edu_label,
    sex_label,
    specs,
):
    # year FE: for every year except reference year, we add a dummy
    reference_year = specs["reference_year"]
    years = list(range(specs["start_year"], specs["end_year"] + 1))
    years.remove(reference_year)

    year_dummies = pd.get_dummies(
        wage_data_type["year"], prefix="year", drop_first=False
    )
    year_dummies = year_dummies.drop(columns=[f"year_{reference_year}"])
    wage_data_type = pd.concat([wage_data_type, year_dummies], axis=1)

    rhs_vars = wage_data_type[regressors + list(year_dummies.columns)]

    # estimate parametric regression, save parameters
    model = PanelOLS(
        dependent=wage_data_type["ln_wage"],
        exog=rhs_vars,
        entity_effects=True,
    )
    fitted_model = model.fit(
        cov_type="clustered", cluster_entity=True, cluster_time=True
    )
    print(fitted_model.summary)
    # Add prediction to data
    wage_data_type["predicted_ln_wage"] = fitted_model.predict()
    wage_data_type["predicted_wage"] = np.exp(wage_data_type["predicted_ln_wage"])

    # Assign estimated parameters (column list corresponds to model params, so only these are assigned)
    for param in regressors:
        wage_parameters.loc[edu_label, sex_label, param] = fitted_model.params[param]
        wage_parameters.loc[edu_label, sex_label, param + "_ser"] = (
            fitted_model.std_errors[param]
        )
    for year in years:
        year_fixed_effects[(edu_label, sex_label)][year] = fitted_model.params[
            f"year_{year}"
        ]
    # Get estimate for income shock std
    (
        wage_parameters.loc[edu_label, sex_label, "income_shock_std"],
        wage_parameters.loc[edu_label, sex_label, "income_shock_std_ser"],
    ) = est_shock_std(
        residuals=fitted_model.resids,
        n_obs=wage_data_type.shape[0],
        n_params=fitted_model.params.shape[0],
    )
    return wage_parameters, year_fixed_effects, wage_data_type


def calc_wage_population_averages(df, year_fixed_effects, specs, paths_dict):
    """Save population average of annual wage (for pension calculation) and working
    hours by education (to compute annual wages).

    We do this here (as opposed to model specs) to avoid loading the data twice.

    """
    reference_year = specs["reference_year"]
    years = list(range(specs["start_year"], specs["end_year"] + 1))
    years.remove(reference_year)
    edu_labels = specs["education_labels"]
    sex_labels = specs["sex_labels"]

    df = df[df["weekly_hours"] > 5].copy()

    # Now everyone has correct monthly hours. We can define hourly wage
    df["hourly_wage"] = df["monthly_wage"] / df["monthly_hours"]
    # We are 2013 onwards, so everybody must at least earn 8.5. This is to filter out false reporting.
    df = df[df["hourly_wage"] > 8.5]
    df["ln_wage"] = np.log(df["hourly_wage"])

    # annual average wage (deflated or inflated by type-specific year fixed effects)

    df["ln_wage_deflated"] = df["ln_wage"].copy()
    for edu_val, edu_label in enumerate(edu_labels):
        for sex_val, sex_label in enumerate(sex_labels):
            for year in years:
                edu_mask = df["education"] == edu_val
                sex_mask = df["sex"] == sex_val
                year_mask = df["year"] == year
                # ref year is always the omitted category, so we add the year FE
                df.loc[
                    edu_mask & sex_mask & year_mask, "ln_wage_deflated"
                ] -= year_fixed_effects[(edu_label, sex_label)][year]

    df["wage_deflated"] = np.exp(df["ln_wage_deflated"])
    df["annual_wage_deflated"] = df["wage_deflated"] * df["annual_hours"]
    pop_avg_annual_wage = df["annual_wage_deflated"].mean()

    np.savetxt(
        paths_dict["first_step_incomes"] + "pop_avg_annual_wage.txt",
        np.array([pop_avg_annual_wage]),
    )

    print(
        f"Population average for annual wage (inflated/deflated to {specs['reference_year']}) : "
        + str(pop_avg_annual_wage)
    )


def calc_population_working_hours(df, paths_dict):
    df_hours = df[df["annual_hours"] > 0].copy()
    # average working hours by type
    avg_hours_by_type_choice = df_hours.groupby(["education", "sex", "choice"])[
        "annual_hours"
    ].mean()
    avg_hours_by_type_choice.to_csv(
        paths_dict["first_step_incomes"] + "population_averages_working_hours.csv",
        index=True,
    )
    print("Population averages for working hours: \n")
    print(avg_hours_by_type_choice)

    return avg_hours_by_type_choice


def est_shock_std(residuals, n_obs, n_params):
    """Estimate income shock std and its standard error."""
    rss = residuals @ residuals
    n_minus_k = n_obs - n_params
    income_shock_var = rss / n_minus_k
    income_shock_std = np.sqrt(income_shock_var)
    income_shock_std_ser = np.sqrt((2 * income_shock_var**2) / n_minus_k)
    return income_shock_std, income_shock_std_ser


def print_wage_equation(wage_parameters, edu_labels, sex_labels):
    """Print wage equation parameters for each education-sex combination."""
    print("=" * 60)
    print("WAGE EQUATION PARAMETERS")
    print("=" * 60)

    for edu_val, edu_label in enumerate(edu_labels):
        for sex_val, sex_label in enumerate(sex_labels):
            # Check if this combination exists in the data
            if (edu_label, sex_label, "constant") not in wage_parameters.index:
                continue

            print(f"\nHourly wage equation: {edu_label} {sex_label}")

            # Get coefficients and standard errors
            constant = wage_parameters.loc[(edu_label, sex_label, "constant"), "value"]
            constant_se = wage_parameters.loc[
                (edu_label, sex_label, "constant_ser"), "value"
            ]
            exp_coef = wage_parameters.loc[(edu_label, sex_label, "ln_exp"), "value"]
            exp_coef_se = wage_parameters.loc[
                (edu_label, sex_label, "ln_exp_ser"), "value"
            ]
            above_50_coeff = wage_parameters.loc[
                (edu_label, sex_label, "above_50_age"), "value"
            ]
            above_50_coeff_ser = wage_parameters.loc[
                (edu_label, sex_label, "above_50_age_ser"), "value"
            ]

            # Format the equation with proper signs
            exp_sign = "+" if exp_coef >= 0 else ""
            exp_sq_sign = "+" if above_50_coeff >= 0 else ""

            print(
                f"ln(hrly_wage) = {constant:.2f} ({constant_se:.2f}) {exp_sign}{exp_coef:.2f} ({exp_coef_se:.2f}) * exp {exp_sq_sign}{above_50_coeff:.4f} ({above_50_coeff_ser:.4f}) * 1( + epsilon"
            )

            # Calculate example wage
            exp = 21
            hrly_wage_with_20_exp = np.exp(constant + exp_coef * exp)
            print(
                f"Example: hourly wage with 20 years of experience: {hrly_wage_with_20_exp:.2f}"
            )

            # Get income shock std
            shock_std = wage_parameters.loc[
                (edu_label, sex_label, "income_shock_std"), "value"
            ]
            print(f"Income shock std: {shock_std:.2f}")
            print("-" * 40)


def print_lc_wage_and_exp(wage_data, edu_labels, sex_labels):
    """Print diagnostic table showing average experience and hourly wage by age groups and type."""
    print("\n" + "=" * 80)
    print("LIFE-CYCLE WAGE AND EXPERIENCE DIAGNOSTICS")
    print("=" * 80)

    # Define age groups
    age_groups = {
        "20s": (20, 29),
        "30s": (30, 39),
        "40s": (40, 49),
        "50s": (50, 59),
        "60s": (60, 69),
    }

    # Create age group column
    wage_data_copy = wage_data.copy()
    wage_data_copy["age_group"] = None
    for group_name, (min_age, max_age) in age_groups.items():
        mask = (wage_data_copy["age"] >= min_age) & (wage_data_copy["age"] <= max_age)
        wage_data_copy.loc[mask, "age_group"] = group_name

    for edu_val, edu_label in enumerate(edu_labels):
        for sex_val, sex_label in enumerate(sex_labels):
            # Filter data for this type
            type_data = wage_data_copy[
                (wage_data_copy["education"] == edu_val)
                & (wage_data_copy["sex"] == sex_val)
                & (wage_data_copy["age_group"].notna())
            ]

            if type_data.empty:
                continue

            print(f"\n{edu_label} {sex_label}")
            print("-" * 60)

            # Calculate correlations
            age_exp_corr = type_data["age"].corr(type_data["experience"])
            age_wage_corr = type_data["age"].corr(type_data["hourly_wage"])

            print(
                f"Correlations: Age-Experience: {age_exp_corr:.3f}, Age-Wage: {age_wage_corr:.3f}"
            )
            print("-" * 60)
            print(
                f"{'Age Group':<10} {'Experience':<20} {'Hourly Wage':<20} {'N Obs':<10}"
            )
            print(f"{'':10} {'Mean (Std)':<20} {'Mean (Std)':<20} {'':10}")
            print("-" * 60)

            for group_name in ["20s", "30s", "40s", "50s", "60s"]:
                group_data = type_data[type_data["age_group"] == group_name]

                if group_data.empty:
                    print(f"{group_name:<10} {'No data':<20} {'No data':<20} {'0':<10}")
                else:
                    exp_mean = group_data["experience"].mean()
                    exp_std = group_data["experience"].std()
                    wage_mean = group_data["hourly_wage"].mean()
                    wage_std = group_data["hourly_wage"].std()
                    n_obs = len(group_data)

                    print(
                        f"{group_name:<10} {exp_mean:.1f} ({exp_std:.1f}){'':<6} {wage_mean:.1f} ({wage_std:.1f}){'':<6} {n_obs:<10}"
                    )

            print()  # Add spacing between types
