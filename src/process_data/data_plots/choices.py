import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from export_results.figures.color_map import JET_COLOR_MAP
from specs.derive_specs import generate_derived_and_data_derived_specs


def plot_data_choices(path_dict, lagged=False):
    struct_est_sample = pd.read_csv(path_dict["struct_est_sample"])

    specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)
    struct_est_sample["age"] = struct_est_sample["period"] + specs["start_age"]
    struct_est_sample = struct_est_sample[struct_est_sample["age"] < 75]

    plot_ages = np.arange(specs["start_age"], 75)
    n_choices = struct_est_sample["choice"].nunique()

    multi_index = pd.MultiIndex.from_product(
        [plot_ages, np.arange(n_choices)], names=["age", "choice"]
    )

    choice_labels = ["Retired", "Unemployed", "Part-time", "Full-time"]
    edu_labels = specs["education_labels"]
    sex_labels = specs["sex_labels"]

    fig, axs = plt.subplots(ncols=n_choices, nrows=2)
    for sex_var, sex_label in enumerate(sex_labels):
        for edu_var, edu_label in enumerate(edu_labels):
            edu_df = struct_est_sample[
                (struct_est_sample["education"] == edu_var)
                & (struct_est_sample["sex"] == sex_var)
            ]
            if not lagged:
                choice_shares = edu_df.groupby("age")["choice"].value_counts(
                    normalize=True
                )
            else:
                choice_shares = edu_df.groupby("age")["lagged_choice"].value_counts(
                    normalize=True
                )
            # Initialize zero shares for all ages and choices
            full_shares = pd.Series(index=multi_index, data=0.0)
            full_shares[choice_shares.index] = choice_shares.values

            for choice in range(n_choices):
                if (sex_var == 0) & (choice == 2):
                    continue
                choice_shares = full_shares.loc[(plot_ages, choice)]
                axs[sex_var, choice].plot(
                    plot_ages,
                    choice_shares,
                    color=JET_COLOR_MAP[edu_var],
                    label=edu_label,
                )
                axs[sex_var, choice].set_ylim(-0.05, 1.05)

                axs[0, choice].set_title(f"{choice_labels[choice]}")

    axs[0, 1].legend()
