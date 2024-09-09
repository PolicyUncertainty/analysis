import matplotlib.pyplot as plt
import numpy as np
from model_code.derive_specs import generate_derived_and_data_derived_specs
from model_code.wealth_and_budget.pensions import calc_net_income_pensions
from model_code.wealth_and_budget.wages import calc_net_income_working


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

        gross_wages = (
            np.maximum(
                np.exp(
                    specs["gamma_0"][edu]
                    + specs["gamma_1"][edu] * np.log(exp_levels + 1)
                )
                / specs["wealth_unit"],
                specs["min_wage"],
            )
            * 12
        )

        net_wages = np.zeros_like(gross_wages)
        net_pensions = np.zeros_like(gross_wages)
        # Calculate pensions times experience
        gross_pensions = specs["pension_point_value_by_edu_exp"][edu, :] * exp_levels * 12
        gross_pensions = np.maximum(gross_pensions, unemployment_benefits)

        for i, exp in enumerate(exp_levels):
            net_wages[i] = calc_net_income_working(gross_wages[i], specs)

            net_pensions[i] = calc_net_income_pensions(gross_pensions[i], specs)
            net_pensions[i] = np.maximum(yearly_unemployment, net_pensions[i])

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


def plot_wages(path_dict):
    specs = generate_derived_and_data_derived_specs(path_dict)

    exp_levels = np.arange(46)
    gross_wages = (
        np.maximum(
            (
                specs["gamma_0"]
                + specs["gamma_1"] * exp_levels
                + specs["gamma_2"] * exp_levels**2
            ),
            specs["min_wage"],
        )
        * 12
    )

    net_wages = np.zeros_like(gross_wages)
    for i, exp in enumerate(exp_levels):
        net_wages[i] = calc_net_income_working(gross_wages[i], specs)

    unemployment_benefits = (
        np.ones_like(exp_levels) * specs["unemployment_benefits"] * 12
    )

    fig, ax = plt.subplots()
    ax.plot(exp_levels, net_wages, label="Average net wage")
    ax.plot(exp_levels, gross_wages, label="Average gross wage")
    ax.plot(exp_levels, unemployment_benefits, label="Unemployment benefits")
    ax.legend(loc="upper left")
    ax.set_xlabel("Experience")
    ax.set_ylabel("Average wage")
