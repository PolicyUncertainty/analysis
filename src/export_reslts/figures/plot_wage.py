import matplotlib.pyplot as plt
import numpy as np
from model_code.budget_equation import calc_net_income_working
from model_code.derive_specs import generate_derived_and_data_derived_specs


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
    ax.set_ylim(0, 80)
    ax.legend()
