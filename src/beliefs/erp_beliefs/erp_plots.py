import matplotlib.pyplot as plt


def plot_predicted_vs_actual(path_dict, predicted_shares, observed_shares, specs):
    plt.rcParams.update(
        {
            "axes.titlesize": 30,
            "axes.labelsize": 30,
            "xtick.labelsize": 30,
            "ytick.labelsize": 30,
            "legend.fontsize": 30,
        }
    )
    # Make lines of plots thicker
    plt.rcParams["lines.linewidth"] = 3
    fig, ax = plt.subplots(figsize=(16, 9))
    for edu_val, edu_label in enumerate(specs["education_labels"]):
        ax.plot(
            observed_shares[edu_label].rolling(window=3).mean(),
            label=f"Obs. {edu_label}",
            marker="o",
            linestyle="None",
            markersize=4,
            color=JET_COLOR_MAP[edu_val],
        )
        ax.plot(
            predicted_shares[edu_label],
            color=JET_COLOR_MAP[edu_val],
            label=f"Est. {edu_label}",
        )
    # Set labels
    ax.set_xlabel("Age")
    ax.set_ylabel("Share Informed")
    ax.legend()
    fig.savefig(path_dict["paper_plots"] + "informed_shares.png")
    plt.show()