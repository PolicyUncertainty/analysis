import matplotlib.pyplot as plt
import numpy as np
from model_code.budget_equation import calc_net_income_pensions
from model_code.budget_equation import calc_net_income_working
from model_code.derive_specs import generate_derived_and_data_derived_specs


def plot_incomes(path_dict):
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

    yearly_unemployment = specs["unemployment_benefits"] * 12
    unemployment_benefits = np.ones_like(exp_levels) * yearly_unemployment

    net_wages = np.zeros_like(gross_wages)
    net_pensions = np.zeros_like(gross_wages)
    for i, exp in enumerate(exp_levels):
        net_wages[i] = calc_net_income_working(gross_wages[i])
        gross_pension = specs["pension_point_value"] * exp * 12
        net_pensions[i] = calc_net_income_pensions(gross_pension)
        net_pensions[i] = np.maximum(yearly_unemployment, net_pensions[i])

    fig, ax = plt.subplots()
    ax.plot(exp_levels, net_wages, label="Average net wage")
    ax.plot(exp_levels, net_pensions, label="Average net pension")
    ax.plot(exp_levels, unemployment_benefits, label="Unemployment benefits")
    ax.legend(loc="upper left")
    ax.set_xlabel("Experience")
    ax.set_ylabel("Average wage")
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
        net_wages[i] = calc_net_income_working(gross_wages[i])

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
