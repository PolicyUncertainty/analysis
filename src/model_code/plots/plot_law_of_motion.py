import numpy as np
from matplotlib import pyplot as plt

from model_code.state_space.experience import (
    construct_experience_years,
    get_next_period_experience,
    scale_experience_years,
)


def plot_ret_experience(specs):
    periods = np.arange(30, 40)
    fig, ax = plt.subplots()
    for exp_years in np.arange(40, 60, 10):

        # We scale the experience for a not retired person last period (policy state 8 is below)
        exp = scale_experience_years(
            experience_years=exp_years,
            period=periods - 1,
            is_retired=np.zeros_like(periods, dtype=bool),
            model_specs=specs,
        )

        exp_next = get_next_period_experience(
            period=periods,
            lagged_choice=0,
            policy_state=8,
            sex=0,
            education=1,
            experience=exp,
            informed=1,
            health=0,
            model_specs=specs,
        )
        exp_years_next = construct_experience_years(
            exp_next, periods, specs["max_exp_diffs_per_period"]
        )
        ax.plot(periods + 30, exp_years_next, label=f"Exp {exp_years}")

    ax.legend()

    plt.show()
