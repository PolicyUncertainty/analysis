import matplotlib.pyplot as plt
import numpy as np
from model_code.wealth_and_budget.pensions import calc_gross_pension_income
from model_code.wealth_and_budget.pensions import calc_pensions
from model_code.wealth_and_budget.wages import calc_labor_income
from model_code.wealth_and_budget.wages import calculate_gross_labor_income
from specs.derive_specs import generate_derived_and_data_derived_specs


def plot_incomes(path_dict):
    specs = generate_derived_and_data_derived_specs(path_dict)

    exp_levels = np.arange(0, specs["exp_cap"] + 1)

    yearly_unemployment = specs["unemployment_benefits"] * 12
    unemployment_benefits = np.ones_like(exp_levels) * yearly_unemployment

    labels = ["Low Education", "High Education"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Now loop over education to generate specific net and gross wages and pensions
    for edu in range(specs["n_education_types"]):
        ax = axes[edu]
        ax.set_ylim([0, 100])

        ax.plot(exp_levels, unemployment_benefits, label="Unemployment benefits")

        # Initialize emoty containers
        gross_wages = np.zeros_like(exp_levels, dtype=float)
        net_wages = np.zeros_like(exp_levels, dtype=float)

        net_pensions = np.zeros_like(exp_levels, dtype=float)
        gross_pensions = np.zeros_like(exp_levels, dtype=float)
        for i, exp in enumerate(exp_levels):
            gross_wages[i] = calculate_gross_labor_income(exp, edu, 0, specs)
            net_wages[i] = calc_labor_income(exp, edu, 0, specs)

            gross_pensions[i] = np.maximum(
                calc_gross_pension_income(exp, edu, 0, 2, specs), yearly_unemployment
            )

            net_pensions[i] = np.maximum(
                calc_pensions(exp, edu, 0, 2, specs), yearly_unemployment
            )

        ax.plot(exp_levels, net_wages, label="Average net wage")
        ax.plot(exp_levels, gross_wages, label="Average gross wage", ls="--")
        ax.plot(exp_levels, net_pensions, label="Average net pension")
        ax.plot(exp_levels, gross_pensions, label="Average gross pension", ls="--")

        ax.legend(loc="upper left")
        ax.set_xlabel("Experience")
        ax.set_ylabel("Average income")
        ax.set_title(f"{labels[edu]}")

    fig.tight_layout()
    fig.savefig(path_dict["plots"] + "incomes.png", transparent=True, dpi=300)


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
#         np.ones_like(exp_levels) * specs["unemployment_benefits"] * 12
#     )
#
#     fig, ax = plt.subplots()
#     ax.plot(exp_levels, net_wages, label="Average net wage")
#     ax.plot(exp_levels, gross_wages, label="Average gross wage")
#     ax.plot(exp_levels, unemployment_benefits, label="Unemployment benefits")
#     ax.legend(loc="upper left")
#     ax.set_xlabel("Experience")
#     ax.set_ylabel("Average wage")
