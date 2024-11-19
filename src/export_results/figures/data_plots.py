import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from specs.derive_specs import generate_derived_and_data_derived_specs


def plot_data_choices(path_dict):
    struct_est_sample = pd.read_pickle(path_dict["struct_est_sample"])

    specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)

    struct_est_sample["age"] = struct_est_sample["period"] + specs["start_age"]
    struct_est_sample = struct_est_sample[struct_est_sample["age"] < 75]

    plot_ages = np.arange(specs["start_age"], 75)
    n_choices = struct_est_sample["choice"].nunique()

    multi_index = pd.MultiIndex.from_product(
        [plot_ages, np.arange(n_choices)], names=["age", "choice"]
    )

    fig, axs = plt.subplots(ncols=n_choices, nrows=specs["n_education_types"])

    choice_labels = ["Retired", "Unemployed", "Part-time", "Full-time"]
    edu_labels = specs["education_labels"]

    for edu in range(specs["n_education_types"]):
        edu_df = struct_est_sample[struct_est_sample["education"] == edu]
        choice_shares = edu_df.groupby("age")["choice"].value_counts(normalize=True)

        # Initialize zero shares for all ages and choices
        full_shares = pd.Series(index=multi_index, data=0)
        full_shares[choice_shares.index] = choice_shares.values

        for choice in range(n_choices):
            choice_shares = full_shares.loc[(plot_ages, choice)]
            axs[edu, choice].plot(plot_ages, choice_shares)
            axs[edu, choice].set_title(f"{edu_labels[edu]}: {choice_labels[choice]}")
            axs[edu, choice].set_ylim(-0.05, 1.05)
