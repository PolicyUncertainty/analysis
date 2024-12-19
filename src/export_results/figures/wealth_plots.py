import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_code.wealth_and_budget.budget_equation import budget_constraint


def plot_budget_of_unemployed(specs):
    """Plot the budget constraint (=wealth) for different levels of end-of period savings of an unemployed person.
    Special emphasis on the area around eligibility for unemployment benefits."""
    params = {
        "interest_rate": specs["interest_rate"],
    }

    savings = np.linspace(0, 30)
    wealth = budget_constraint(
        period=0,
        education=0,
        lagged_choice=0,
        experience=0,
        partner_state=np.array([2]),
        savings_end_of_previous_period=savings,
        income_shock_previous_period=0,
        params=params,
        options=specs,
    )

    fig, ax = plt.subplots()
    ax.plot(savings, wealth * specs["wealth_unit"])
    ax.set_xlabel("Savings")
    ax.set_ylabel("Wealth")
    ax.set_title("Budget constraint of an unemployed person")

