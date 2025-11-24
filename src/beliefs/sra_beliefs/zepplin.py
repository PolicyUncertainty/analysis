import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from beliefs.sra_beliefs.sra_plots import JET_COLOR_MAP


def plot_example_sra_evolution(
    alpha_star,
    SRA_30,
    resolution_age,
    alpha=None,
    sigma_sq=None,
    use_estimated_params=False,
    path_dict=None,
    show=False,
    save=False,
):
    """Plot the evolution of true policy and SRA expectations over age with confidence intervals."""

    if use_estimated_params:
        # Load estimated parameters from the beliefs data
        df = pd.read_csv(path_dict["beliefs_est_results"] + "beliefs_parameters.csv")
        alpha = df[df["parameter"] == "alpha"]["estimate"].values[0]
        sigma_sq = df[df["parameter"] == "sigma_sq"]["estimate"].values[0]

    if alpha is None or sigma_sq is None:
        raise ValueError("Alpha and sigma_sq must be provided or estimated from data.")

    ages = np.arange(30, resolution_age + 1, 1)
    SRA_t = np.ones(ages.shape) * SRA_30
    SRA_t = SRA_t + (ages - 30) * alpha_star
    exp_SRA_resolution = SRA_t + (resolution_age - ages) * alpha
    ci_upper = exp_SRA_resolution + 1.96 * np.sqrt(sigma_sq) * np.sqrt(
        resolution_age - ages
    )
    ci_lower = exp_SRA_resolution - 1.96 * np.sqrt(sigma_sq) * np.sqrt(
        resolution_age - ages
    )
    fig, ax = plt.subplots()
    ax.plot(ages, SRA_t, label="$SRA_t$", color=JET_COLOR_MAP[3])
    ax.plot(
        ages,
        exp_SRA_resolution,
        label=f"$E[SRA_{{{resolution_age}}}|SRA_t]$",
        color="C0",
    )
    ax.plot(ages, ci_upper, label="95% CI", linestyle="--", color=JET_COLOR_MAP[0])
    ax.plot(ages, ci_lower, label="", linestyle="--", color=JET_COLOR_MAP[0])
    ax.set_xlabel("Age")
    ax.set_ylabel("SRA")
    ax.legend()

    # ax.set_title(
    #    f"Evolution of SRA Expectations with $\\alpha^*={alpha_star}$, $SRA_{{30}}={SRA_30}$, $\\alpha={alpha}$, $\\sigma^2={sigma_sq}$"
    # )
    # ax.grid()
    plt.tight_layout()

    if save:
        if path_dict is None:
            raise ValueError("path_dict must be provided when save=True")
        plt.savefig(
            path_dict["beliefs_plots"] + "example_sra_evolution_no_increase.png",
            bbox_inches="tight",
        )
    if show:
        plt.show()
