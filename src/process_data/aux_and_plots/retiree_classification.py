import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from set_paths import create_path_dict
path_dict = create_path_dict()
from specs.derive_specs import generate_derived_and_data_derived_specs
specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)

def classify_retirees(paths):
    struct_est_sample = pd.read_pickle(paths["struct_est_sample"])
    struct_est_sample["age"] = struct_est_sample["period"] + specs["start_age"]

    fresh_retirees = struct_est_sample[
        (struct_est_sample["choice"] == 0) & (struct_est_sample["lagged_choice"] != 0)
    ]
    fresh_retirees.loc[:, "ret_after_SRA"] = (fresh_retirees["age"] >= fresh_retirees["policy_state_value"]).astype(int)
    fresh_retirees.loc[:, "ret_before_SRA_over_45_years"] = ((fresh_retirees["age"] < fresh_retirees["policy_state_value"]) & (fresh_retirees["working_years"] >= 45)).astype(int)
    fresh_retirees.loc[:, "ret_before_SRA_under_45_years"] = ((fresh_retirees["age"] < fresh_retirees["policy_state_value"]) & (fresh_retirees["working_years"] < 45)).astype(int)

    return fresh_retirees

def plot_retiree_classification(paths):
    fresh_retirees = classify_retirees(paths)
    fresh_retirees.groupby("age")[["ret_after_SRA", "ret_before_SRA_over_45_years", "ret_before_SRA_under_45_years"]].sum().plot(kind="bar", stacked=True)
    plt.xlabel("Age")
    plt.ylabel("Number of Individuals")
    plt.title("Retiree Types")
    