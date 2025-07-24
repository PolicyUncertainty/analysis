import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from export_results.figures.color_map import JET_COLOR_MAP
from model_code.pension_system.experience_stock import (
    calc_experience_years_for_pension_adjustment,
)
from model_code.wealth_and_budget.budget_equation import budget_constraint
from model_code.wealth_and_budget.pension_payments import (
    calc_gross_pension_income,
    calc_pensions_after_ssc,
)
from model_code.wealth_and_budget.transfers import calc_child_benefits
from model_code.wealth_and_budget.wages import (
    calc_labor_income_after_ssc,
    calculate_gross_labor_income,
)
from specs.derive_specs import generate_derived_and_data_derived_specs


def plot_incomes(path_dict):
    "Plots incomes of men. TODO: add women"
    specs = generate_derived_and_data_derived_specs(path_dict)

    exp_levels = np.arange(0, 50)

    annual_unemployment = specs["annual_unemployment_benefits"]
    unemployment_benefits = np.ones_like(exp_levels) * annual_unemployment

    fig, axes = plt.subplots(2, 2, figsize=(10, 5))

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        # Now loop over education to generate specific net and gross wages and pensions
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            ax = axes[sex_var, edu_var]

            ax.plot(
                exp_levels,
                unemployment_benefits,
                label="Unemployment benefits",
                color=JET_COLOR_MAP[0],
            )

            # Initialize empty containers, part and full time wages and pensions
            gross_pt_wages = np.zeros_like(exp_levels, dtype=float)
            after_ssc_pt_wages = np.zeros_like(exp_levels, dtype=float)

            gross_ft_wages = np.zeros_like(exp_levels, dtype=float)
            after_ssc_ft_wages = np.zeros_like(exp_levels, dtype=float)

            net_pensions = np.zeros_like(exp_levels, dtype=float)
            gross_pensions = np.zeros_like(exp_levels, dtype=float)
            for i, exp in enumerate(exp_levels):
                gross_pt_wages[i] = calculate_gross_labor_income(
                    lagged_choice=2,
                    experience_years=exp,
                    education=edu_var,
                    sex=sex_var,
                    income_shock=0,
                    model_specs=specs,
                )
                after_ssc_pt_wages[i], _ = calc_labor_income_after_ssc(
                    lagged_choice=2,
                    experience_years=exp,
                    education=edu_var,
                    sex=sex_var,
                    income_shock=0,
                    model_specs=specs,
                )

                gross_ft_wages[i] = calculate_gross_labor_income(
                    lagged_choice=3,
                    experience_years=exp,
                    education=edu_var,
                    sex=sex_var,
                    income_shock=0,
                    model_specs=specs,
                )
                after_ssc_ft_wages[i], _ = calc_labor_income_after_ssc(
                    lagged_choice=3,
                    experience_years=exp,
                    education=edu_var,
                    sex=sex_var,
                    income_shock=0,
                    model_specs=specs,
                )

                exp_stock_pension = calc_experience_years_for_pension_adjustment(
                    period=37,
                    sex=sex_var,
                    experience_years=exp,
                    education=edu_var,
                    policy_state=8,  # Make 67 as well
                    informed=1,
                    health=0,
                    model_specs=specs,
                )

                gross_pensions[i] = np.maximum(
                    calc_gross_pension_income(
                        experience_years=exp_stock_pension,
                        education=edu_var,
                        sex=sex_var,
                        model_specs=specs,
                    ),
                    annual_unemployment,
                )

                net_pensions[i] = np.maximum(
                    calc_pensions_after_ssc(
                        experience_years=exp_stock_pension,
                        education=edu_var,
                        sex=sex_var,
                        model_specs=specs,
                    ),
                    annual_unemployment,
                )

            ax.plot(
                exp_levels,
                after_ssc_pt_wages,
                label="Average after ssc pt wage",
                color=JET_COLOR_MAP[1],
            )
            ax.plot(
                exp_levels,
                gross_pt_wages,
                label="Average gross pt wage",
                ls="--",
                color=JET_COLOR_MAP[1],
            )

            ax.plot(
                exp_levels,
                after_ssc_ft_wages,
                label="Average after ssc ft wage",
                color=JET_COLOR_MAP[2],
            )
            ax.plot(
                exp_levels,
                gross_ft_wages,
                label="Average gross ft wage",
                ls="--",
                color=JET_COLOR_MAP[2],
            )

            ax.plot(
                exp_levels,
                net_pensions,
                label="Average after ssc pension",
                color=JET_COLOR_MAP[3],
            )
            ax.plot(
                exp_levels,
                gross_pensions,
                label="Average gross pension",
                ls="--",
                color=JET_COLOR_MAP[3],
            )

            ax.legend(loc="upper left")
            ax.set_xlabel("Experience")
            ax.set_ylabel("Average income")
            ax.set_title(f"{sex_label}; {edu_label}")

    fig.suptitle("After sssc income")
    fig.tight_layout()
    fig.savefig(path_dict["plots"] + "incomes.png", transparent=True, dpi=300)


def plot_total_income(specs):

    exp_levels = np.arange(0, 46)
    marriage_labels = ["Single", "Partnered"]
    edu_labels = specs["education_labels"]

    max_inc = 100_000

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
        for married_val, married_label in enumerate(marriage_labels):
            for edu_val, edu_label in enumerate(edu_labels):
                for choice, work_label in enumerate(specs["choice_labels"]):
                    total_income = np.zeros_like(exp_levels, dtype=float)
                    for i, exp in enumerate(exp_levels):
                        if work_label == "Retired":
                            period = 45
                        else:
                            period = exp
                        exp_share = exp / (
                            exp + specs["max_exp_diffs_per_period"][period]
                        )
                        total_income[i] = budget_constraint(
                            period=period,
                            education=edu_val,
                            lagged_choice=choice,
                            experience=exp_share,
                            health=jnp.array(0),
                            sex=sex_var,
                            partner_state=np.array(married_val),
                            asset_end_of_previous_period=0,
                            income_shock_previous_period=0,
                            model_specs=specs,
                        )
                    axs[edu_val, married_val].plot(
                        exp_levels,
                        total_income * specs["wealth_unit"],
                        label=f"{work_label}",
                    )
                axs[edu_val, married_val].set_title(f"{edu_label} and {married_label}")
                axs[edu_val, married_val].set_xlabel("Period equals experience")
                axs[edu_val, married_val].set_ylim([0, max_inc])
                axs[edu_val, married_val].legend()

        fig.suptitle(f"Total income; {sex_label}")


def plot_partner_wage(paths_dict, specs):
    """Plot the partner wage by age."""
    df = pd.read_pickle(
        paths_dict["intermediate_data"] + "partner_wage_estimation_sample.pkl"
    )

    start_age = specs["start_age"]

    wage_data = df.groupby(["sex", "education", "age"])["wage_p"].mean()
    partner_wage_est = specs["annual_partner_wage"]

    fig, axs = plt.subplots(ncols=2)
    # Only plot until 70
    max_period = 40
    max_age = start_age + max_period - 1
    periods = np.arange(40)
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        ax = axs[sex_var]
        for edu_val, edu_label in enumerate(specs["education_labels"]):
            ax.plot(
                periods,
                wage_data.loc[(sex_var, edu_val, slice(start_age, max_age))] * 12,
                label=f"edu {edu_label}",
            )
            ax.plot(
                periods,
                partner_wage_est[sex_var, edu_val, :max_period],
                linestyle="--",
                label=f"edu {edu_label} est",
            )

        ax.legend()
        ax.set_title(f"Partner wage of {sex_label} by period")


def plot_child_benefits(specs):
    max_period = 40
    periods = np.arange(40)
    education_labels = specs["education_labels"]
    marriage_labels = ["Single", "Partnered"]

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for partner_val, partner_label in enumerate(marriage_labels):
            ax = axs[sex_var, partner_val]
            for edu_val, edu_label in enumerate(education_labels):
                child_benefits = np.zeros_like(periods, dtype=float)
                for i, period in enumerate(periods):
                    child_benefits[i] = calc_child_benefits(
                        education=edu_val,
                        sex=sex_var,
                        has_partner_int=partner_val,
                        period=period,
                        model_specs=specs,
                    )
                ax.plot(periods, child_benefits, label=f"{edu_label}")
            ax.set_title(f"{sex_label}; {partner_label}")
            ax.set_xlabel("Periods")
            ax.legend()
    fig.suptitle("Child benefits")


# def plot_wages(path_dict):
#     specs = generate_derived_and_data_derived_specs(path_dict)
#
#     exp_levels = np.arange(46)
#     # Initialize emoty containers
#     gross_wages = np.zeros_like(exp_levels)
#     net_wages = np.zeros_like(exp_levels)
#
#     net_pensions = np.zeros_like(exp_levels)
#     gross_pensions = np.zeros_like(exp_levels)
#     for i, exp in enumerate(exp_levels):
#         gross_wages[i] = calculate_gross_labor_income(exp, edu, 0, specs)
#         net_wages[i] = calc_labor_income(exp, edu, 0, specs)
#
#         gross_pensions[i] = calc_gross_pension_income(exp, edu, 0, 2, specs)
#         net_pensions[i] = calc_pensions(exp, edu, 0, 2, specs)
#
#     unemployment_benefits = (
#         np.ones_like(exp_levels) * specs["monthly_unemployment_benefits"] * 12
#     )
#
#     fig, ax = plt.subplots()
#     ax.plot(exp_levels, net_wages, label="Average net wage")
#     ax.plot(exp_levels, gross_wages, label="Average gross wage")
#     ax.plot(exp_levels, unemployment_benefits, label="Unemployment benefits")
#     ax.legend(loc="upper left")
#     ax.set_xlabel("Experience")
#     ax.set_ylabel("Average wage")
