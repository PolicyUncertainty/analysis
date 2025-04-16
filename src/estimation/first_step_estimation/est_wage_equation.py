# Description: This file estimates the parameters of the HOURLY wage equation using the SOEP panel data.
# We estimate the following equation for each education level:
# ln_wage = beta_0 + beta_1 * ln_(exp+1) + individual_FE + time_FE + epsilon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from export_results.figures.color_map import JET_COLOR_MAP
from linearmodels.panel.model import PanelOLS


def estimate_wage_parameters(paths_dict, specs):
    """Estimate the wage parameters for each education group in the sample.

    Also estimate for all individuals.

    """
    # specs, data, and parameter containers
    edu_labels = specs["education_labels"]
    sex_labels = specs["sex_labels"]
    regressors = ["constant", "ln_exp"]
    wage_data = load_and_prepare_wage_data(paths_dict)
    wage_parameters, year_fixed_effects = initialize_coeficient_containers(regressors, specs)

    # Estimate wage equation for each type (sex x education)
    fit_panel_reg_model(
        wage_data, regressors, wage_parameters, year_fixed_effects, "all", "all", specs
    )
    for sex_val, sex_label in enumerate(sex_labels):
        fig, ax = plt.subplots(figsize=(12, 8))
        for edu_val, edu_label in enumerate(edu_labels):
            wage_data_type = wage_data[
                (wage_data["education"] == edu_val) & (wage_data["sex"] == sex_val)
            ].copy()
            year_fixed_effects[edu_label, sex_label] = {}
            wage_parameters, year_fixed_effects, wage_data_type = fit_panel_reg_model(
                wage_data_type,
                regressors,
                wage_parameters,
                year_fixed_effects,
                edu_label,
                sex_label,
                specs
            )

            wage_data_type = wage_data_type[
                wage_data_type["age"] < specs["max_est_age_labor"]
            ]
            # Plot
            ax.plot(
                wage_data_type.groupby("age")["ln_wage"].mean(),
                color=JET_COLOR_MAP[edu_val],
                ls="--",
                label=f"Obs. {edu_label}",
            )
            ax.plot(
                wage_data_type.groupby("age")["predicted_ln_wage"].mean(),
                color=JET_COLOR_MAP[edu_val],
                label=f"Est. {edu_label}",
            )
        ax.set_title(f"{sex_label}")
        ax.set_xlabel("Age")
        ax.set_ylabel("Log hourly wage")
        ax.legend(loc="upper left")
        file_appends = ["men", "women"]
        fig.savefig(paths_dict["plots"] + f"wages_{file_appends[sex_val]}.png")
        #plt.show()

    # Save results
    wage_parameters.to_csv(paths_dict["est_results"] + "wage_eq_params.csv")
    wage_parameters.T.to_latex(
        paths_dict["tables"] + "wage_eq_params.tex", float_format="%.4f"
    )
    # After estimation print some summary statistics
    print_wage_equation(wage_parameters, edu_labels, sex_labels)
    calc_population_averages(wage_data, year_fixed_effects, specs, paths_dict)

    return wage_parameters


def load_and_prepare_wage_data(paths_dict):
    # Load wage data
    wage_data = pd.read_pickle(
        paths_dict["intermediate_data"] + "wage_estimation_sample.pkl"
    )
    # Modify
    wage_data["ln_wage"] = np.log(wage_data["hourly_wage"])
    wage_data["ln_exp"] = np.log(wage_data["experience"] + 1)
    wage_data["constant"] = np.ones(len(wage_data))
    # Format & Index
    wage_data["year"] = wage_data["syear"].astype("category")
    wage_data = wage_data.set_index(["pid", "syear"])
    return wage_data

def initialize_coeficient_containers(regressors, specs):
    edu_labels = specs["education_labels"]
    sex_labels = specs["sex_labels"]
    coefficents = regressors + [param + "_ser" for param in regressors]
    index = pd.MultiIndex.from_product(
        [edu_labels, sex_labels, coefficents], names=["education", "sex", "parameter"]
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
    specs
):
    # year FE: for every year except reference year, we add a dummy
    reference_year = specs["reference_year"]
    years = list(range(specs["start_year"], specs["end_year"] + 1))
    years.remove(reference_year)
    year_dummies = pd.get_dummies(wage_data_type["year"], prefix="year", drop_first=False)
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
    # Add prediction to data
    wage_data_type["predicted_ln_wage"] = fitted_model.predict()

    # Assign estimated parameters (column list corresponds to model params, so only these are assigned)
    for param in regressors:
        wage_parameters.loc[edu_label, sex_label, param] = fitted_model.params[param]
        wage_parameters.loc[
            edu_label, sex_label, param + "_ser"
        ] = fitted_model.std_errors[param]
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


def calc_population_averages(df, year_fixed_effects, specs, paths_dict):
    """Save population average of annual wage (for pension calculation) and working
    hours by education (to compute annual wages).

    We do this here (as opposed to model specs) to avoid loading the data twice.

    """
    reference_year = specs["reference_year"]
    years = list(range(specs["start_year"], specs["end_year"] + 1))
    years.remove(reference_year)
    edu_labels = specs["education_labels"]
    sex_labels = specs["sex_labels"]

    # annual average wage (deflated or inflated by type-specific year fixed effects)
    
    df["ln_wage_deflated"] = df["ln_wage"].copy()
    for edu_val, edu_label in enumerate(edu_labels):
        for sex_val, sex_label in enumerate(sex_labels):
            for year in years:
                edu_mask = df["education"] == edu_val
                sex_mask = df["sex"] == sex_val
                year_mask = df["year"] = year
                # ref year is always the omitted category, so we add the year FE
                df.loc[
                    edu_mask & sex_mask & year_mask, "ln_wage_deflated"
                ] -= year_fixed_effects[(edu_label, sex_label)][year]

    df["annual_hours"] = df["monthly_hours"] * 12
    df["annual_wage_deflated"] = np.exp(df["ln_wage_deflated"]) * df["annual_hours"]
    pop_avg_annual_wage = df["annual_wage_deflated"].mean()
    np.save(paths_dict["est_results"] + "pop_avg_annual_wage", pop_avg_annual_wage)


    print(f"Population average for annual wage (inflated/deflated to {specs['reference_year']}) : " + str(pop_avg_annual_wage))

    # averageannual working hours by type
    avg_hours_by_type_choice = df.groupby(["education", "sex", "choice"])[
        "annual_hours"
    ].mean()
    avg_hours_by_type_choice.to_csv(
        paths_dict["est_results"] + "population_averages_working_hours.csv", index=True
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
    # print wage equation
    for edu_val, edu_label in enumerate(edu_labels):
        for sex_val, sex_label in enumerate(sex_labels):
            print("Hourly wage equation: " + edu_label + " " + sex_label)
            print(
                "ln(hrly_wage) = "
                + str(wage_parameters.loc[(edu_label, sex_label, "constant"), "value"])
                + " + "
                + str(wage_parameters.loc[(edu_label, sex_label, "ln_exp"), "value"])
                + " * ln(exp+1) + epsilon"
            )
            hrly_wage_with_20_exp = np.exp(
                wage_parameters.loc[(edu_label, sex_label, "constant"), "value"]
                + wage_parameters.loc[(edu_label, sex_label, "ln_exp"), "value"]
                * np.log(20)
            )
            print(
                "Example: hourly wage with 20 years of experience: "
                + str(hrly_wage_with_20_exp)
            )
            print(
                "Income shock std: "
                + str(
                    wage_parameters.loc[
                        (edu_label, sex_label, "income_shock_std"), "value"
                    ]
                )
            )
            print("--------------------")
