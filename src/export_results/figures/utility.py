# %%
import matplotlib.pyplot as plt
import numpy as np
from model_code.utility_functions import utility_func


default_params = {"mu": 0.5, "dis_util_work": 0.1, "dis_util_unemployed": 0.2}

# %%


def plot_utility(params=default_params, options = None):
    consumption = np.linspace(0.5, 4, 100)
    partner_state = np.array([1])
    education = 1
    period = 35
    choice = [0, 1, 2]
    

    plt.plot(
        utility_func(consumption, partner_state, education, period, choice[0], params, options), consumption, label="Unemployed"
    )
    plt.plot(utility_func(consumption, partner_state, education, period, choice[1], params, options), consumption, label="Working")
    plt.plot(utility_func(consumption, partner_state, education, period, choice[2], params, options), consumption, label="Retired")
    plt.legend()
    plt.ylabel("Consumption")
    plt.xlabel("Utility")
    plt.title("Utility function (reversed axes)")