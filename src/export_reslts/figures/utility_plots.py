#%%

import matplotlib.pyplot as plt
import numpy as np
from model_code.utility_functions import utility_func

default_params = {
    "mu": 0.5,
    "dis_util_work": 1,
    "dis_util_unemployed": 2,
}

def plot_utility_by_choice(path_dict, params=default_params):
    consumption = np.linspace(0, 4, 100)
    utility_retired = utility_func(consumption, 2, params)
    utility_work = utility_func(consumption, 1, params)
    utility_unemployed = utility_func(consumption, 0, params)

    fig, ax = plt.subplots()
    ax.plot(consumption, utility_retired, label="retired")
    ax.plot(consumption, utility_work, label="working")
    ax.plot(consumption, utility_unemployed, label="unemployed")
    ax.legend(loc="upper left")
    ax.set_xlabel("Consumption")
    ax.set_ylabel("Utility")
    fig.tight_layout()
    fig.savefig(path_dict["plots"] + "utility_by_choice.png", transparent=True, dpi=300)
# %%
