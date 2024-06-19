#%%

import matplotlib.pyplot as plt
import numpy as np

from model_code.utility_functions import utility_func


default_params = {"mu": 0.5, "dis_util_work": 0.1, "dis_util_unemployed": 0.2}

#%%

def plot_utility(params=default_params):
    consumption = np.linspace(0, 4, 100)
    choice = [0, 1, 2]
    params = {"mu": 0.5, "dis_util_work": 0.1, "dis_util_unemployed": 0.2}

    plt.plot(consumption, utility_func(consumption, choice[0], params), label="Unemployed")
    plt.plot(consumption, utility_func(consumption, choice[1], params), label="Working")
    plt.plot(consumption, utility_func(consumption, choice[2], params), label="Retired")
    plt.legend()
    plt.xlabel("Consumption")
    plt.ylabel("Utility")