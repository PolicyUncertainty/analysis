import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from model_code.wealth_and_budget.budget_equation import budget_constraint


def plot_budget_of_unemployed(specs):
    """Plot the budget constraint (=wealth) for different levels of end-of period
    savings of an unemployed person.

    Special emphasis on the area around eligibility for unemployment benefits.

    """
    params = {
        "interest_rate": specs["interest_rate"],
    }

    savings = np.linspace(0, 60)

    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        ax = axs[sex_var]
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            wealth = budget_constraint(
                period=72,
                education=edu_var,
                lagged_choice=1,
                experience=0.05,
                sex=sex_var,
                partner_state=np.array([0]),
                savings_end_of_previous_period=savings,
                income_shock_previous_period=0,
                params=params,
                options=specs,
            )
            ax.plot(savings, wealth * specs["wealth_unit"], label=edu_label)
            ax.set_xlabel("Savings")
            ax.set_ylabel("Wealth")
            ax.legend()
            ax.set_title(f"Unemployment benefits {sex_label}; {edu_label}")
