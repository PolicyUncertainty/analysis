import numpy as np
import pandas as pd


def create_max_experience(path_dict, specs, load_precomputed):
    # Initial experience
    if load_precomputed:
        max_init_experience = int(
            np.loadtxt(path_dict["intermediate_data"] + "max_init_exp.txt")
        )
        max_experience = int(np.loadtxt(path_dict["intermediate_data"] + "max_exp.txt"))
    else:
        # max initial experience
        data_decision = pd.read_pickle(
            path_dict["intermediate_data"] + "structural_estimation_sample.pkl"
        )
        max_init_experience = (
            data_decision["experience"] - data_decision["period"]
        ).max()
        np.savetxt(
            path_dict["intermediate_data"] + "max_init_exp.txt", [max_init_experience]
        )

        # Now max overall
        max_experience = data_decision["experience"].max()
        np.savetxt(path_dict["intermediate_data"] + "max_exp.txt", [max_experience])
    return max_init_experience, max_experience
