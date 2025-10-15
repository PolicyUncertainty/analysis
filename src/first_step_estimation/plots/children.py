import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from set_styles import get_figsize, set_colors


def plot_children(path_dict, specs, show=False, save=False):
    """Plot the number of children by age.

    Parameters
    ----------
    path_dict : dict
        Dictionary containing paths to data and output directories
    specs : dict
        Dictionary containing model specifications
    show : bool, default False
        Whether to display plots
    save : bool, default False
        Whether to save plots to disk
    """
    # Calculate the number of children in the household for each individual conditional
    # on sex, education and age bin.
    df = pd.read_pickle(
        path_dict["first_step_data"] + "partner_transition_estimation_sample.pkl"
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

    fig, axs = plt.subplots(ncols=4, figsize=get_figsize(ncols=4))
    i = 0

    colors, _ = set_colors()
    sex_labels = ["Men", "Women"]
    partner_labels = ["Single", "Partnered"]

    for sex, sex_label in enumerate(sex_labels):
        for has_partner, partner_label in enumerate(partner_labels):
            ax = axs[i]
            i += 1
            for edu, edu_label in enumerate(specs["education_labels"]):
                nb_children_data_edu = nb_children_data.loc[
                    (sex, edu, has_partner, slice(None))
                ]
                nb_children_container = pd.Series(data=0, index=ages, dtype=float)
                nb_children_container.update(nb_children_data_edu)

                nb_children_est_edu = nb_children_est[sex, edu, has_partner, :]
                ax.plot(
                    ages,
                    nb_children_container,
                    color=colors[edu],
                    linestyle="--",
                    label=f"Obs. {edu_label}",
                )
                ax.plot(
                    ages,
                    nb_children_est_edu,
                    color=colors[edu],
                    label=f"Est. {edu_label}",
                )

            ax.set_ylim([0, 2.5])
            ax.set_title(f"{sex_label}, {partner_label}")

    axs[0].legend()
    plt.tight_layout()

    if save:
        fig.savefig(path_dict["first_step_plots"] + "children.pdf", bbox_inches="tight")
        fig.savefig(
            path_dict["first_step_plots"] + "children.png", bbox_inches="tight", dpi=300
        )

    if show:
        plt.show()
    else:
        plt.close(fig)
