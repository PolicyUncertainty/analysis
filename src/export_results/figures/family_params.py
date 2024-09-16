import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_children(paths_dict, specs):
    """Plot the number of children by age."""

    """Calculate the number of children in the household for each individual conditional
    on sex, education and age bin."""
    df = pd.read_pickle(
        paths_dict["intermediate_data"] + "partner_transition_estimation_sample.pkl"
    )

    start_age = specs["start_age"]
    end_age = specs["end_age"]
    df = df[df["age"] <= end_age]

    df["has_partner"] = (df["partner_state"] > 0).astype(int)

    # calculate average hours worked by partner by age, sex and education
    cov_list = ["sex", "education", "has_partner", "age"]
    nb_children_data = df.groupby(cov_list)["children"].mean()

    nb_children_est = specs["children_by_state"]
    ages = np.arange(start_age, end_age + 1)

    fig, axs = plt.subplots(ncols=4, figsize=(12, 8))
    i = 0

    sex_labels = ["Men", "Women"]
    partner_labels = ["Single", "Partnered"]
    for sex, sex_label in enumerate(sex_labels):
        for has_partner, partner_label in enumerate(partner_labels):
            ax = axs[i]
            i += 1
            for edu, edu_label in enumerate(specs["education_labels"]):
                nb_children_data_edu = nb_children_data.loc[
                    (sex, edu, has_partner, slice(start_age, end_age))
                ].values
                nb_children_est_edu = nb_children_est[sex, edu, has_partner, :]
                ax.plot(ages, nb_children_data_edu, label=f"edu {edu}")
                ax.plot(
                    ages, nb_children_est_edu, linestyle="--", label=f"edu {edu} est"
                )

            ax.set_ylim([0, 2.5])
            ax.set_title(f"{sex_label}, {partner_label}")
            ax.legend()
