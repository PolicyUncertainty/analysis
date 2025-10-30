import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from set_styles import get_figsize, set_colors, set_plot_defaults


def plot_children(path_dict, specs, show=False, paper_plot=False):
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
    set_plot_defaults()
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

    if paper_plot:
        figs = []
        axs = []
        for _ in range(4):
            fig, ax = plt.subplots()
            figs.append(fig)
            axs.append(ax)
    else:
        fig, axs = plt.subplots(ncols=4, figsize=get_figsize(ncols=4))

    colors, _ = set_colors()
    sex_labels = ["Men", "Women"]
    partner_labels = ["Single", "Partnered"]
    i = 0
    titles = []
    for sex, sex_label in enumerate(sex_labels):
        for has_partner, partner_label in enumerate(partner_labels):
            ax = axs[i]
            if paper_plot:
                sex_label = ["men", "women"][sex]
                partner_label = ["single", "partnered"][has_partner]
                titles.append(f"{sex_label}_{partner_label}")
            else:
                ax.set_title(f"{sex_label}, {partner_label}")
            i += 1
            for edu, edu_label in enumerate(specs["education_labels"]):
                nb_children_data_edu = nb_children_data.loc[
                    (sex, edu, has_partner, slice(None))
                ]
                nb_children_container = pd.Series(data=0, index=ages, dtype=float)
                nb_children_container.update(nb_children_data_edu)

                nb_children_est_edu = nb_children_est[sex, edu, has_partner, :]

                low_edu_label = edu_label.lower()
                ax.plot(
                    ages,
                    nb_children_container,
                    color=colors[edu],
                    linestyle="--",
                    label=f"obs. {low_edu_label}",
                )
                ax.plot(
                    ages,
                    nb_children_est_edu,
                    color=colors[edu],
                    label=f"est. {low_edu_label}",
                )

            ax.set_ylim([0, 2.5])
            ax.legend(frameon=False)

    if paper_plot:
        for fig, title in zip(figs, titles):
            fig.tight_layout()
            fig.savefig(
                path_dict["first_step_plots"] + f"children_{title}.png",
                bbox_inches="tight",
                dpi=100,
            )

    else:
        fig.tight_layout()
        fig.savefig(path_dict["first_step_plots"] + "children.pdf", bbox_inches="tight")
        fig.savefig(
            path_dict["first_step_plots"] + "children.png", bbox_inches="tight", dpi=100
        )

    if show:
        plt.show()
    else:
        plt.close(fig)
