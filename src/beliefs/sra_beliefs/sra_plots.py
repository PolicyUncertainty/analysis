import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from beliefs.sra_beliefs.random_walk import filter_df
from beliefs.sra_beliefs.truncated_normals import estimate_truncated_normal_parameters
from set_styles import set_colors
JET_COLOR_MAP, LINE_STYLES = set_colors()


def plot_truncated_normal_for_response(
    responses: list[float],
    options,
    mu: float = None,
    sigma: float = None,
    upper_trunc_limit: float = None,
    show: bool = False,
    save: bool = False,
    path_dict: dict = None,
) -> None:
    """
    Plot the truncated normal subjective expectation distribution for a given set (triple) of SRA responses. If parameters are not provided, the truncated normal distribution has to be estimated. In that case, there is the possibility to overwrite upper truncation limit.

    Args:
        responses (list[float]): List of response values.
        options (dict): Dictionary containing the lower truncation limit and other options.
        mu (float, optional): location parameter of the distribution. Defaults to None.
        scale (float, optional): scale parameter of the distribution. Defaults to None.
    """
    if sum(responses) != 100:
        raise ValueError("Responses must sum to 100.")
    if len(responses) != 3:
        raise ValueError("Responses must be a list of three values.")

    if mu is None:
        # create df
        df = pd.DataFrame(
            {
                "pol_unc_stat_ret_age_67": [responses[0]],
                "pol_unc_stat_ret_age_68": [responses[1]],
                "pol_unc_stat_ret_age_69": [responses[2]],
            }
        )

        # create function_spec dict
        function_spec = {
            "ll": options["lower_limit"],
            "ul": (
                options["upper_limit"]
                if upper_trunc_limit is None
                else upper_trunc_limit
            ),
            "first_cdf_point": options["first_cdf_point"],
            "second_cdf_point": options["second_cdf_point"],
        }
        # estimate parameters
        df = estimate_truncated_normal_parameters(df, function_spec)
        mu = df["mu"].values[0]
        sigma = df["sigma"].values[0]
        expected_sra = df["ex_val"].values[0]
        variance_sra = df["var"].values[0]
    else:
        # check if loc and scale are valid
        if mu <= 0 or sigma <= 0:
            raise ValueError("mu and sigma must be positive real values.")

    # Parameters for the truncated normal distribution and the plot
    lower = options["lower_limit"]
    upper = options["upper_limit"] if upper_trunc_limit is None else upper_trunc_limit
    a, b = (lower - mu) / sigma, (upper - mu) / sigma

    # plot pdf
    x = np.linspace(lower, upper, 1000)
    pdf = truncnorm.pdf(x, a, b, loc=mu, scale=sigma)
    fig, ax = plt.subplots()
    ax.plot(x, pdf, color=JET_COLOR_MAP[0], label="Truncated Normal PDF")

    # Divide responses by 100
    # Normalize the probability mass for the range 69+ because the domain is larger than 1 (unlike fot the other two responses)
    responses = np.array(responses) / 100.0
    responses[2] = responses[2] / (upper - 68.5)

    # Define the x positions and widths for the bars
    bar_positions = [66.5, 67.5, 68.5]
    bar_widths = [1.0, 1.0, upper - 68.5]
    bar_heights = responses

    # Plot the bars
    for i, (pos, width, height) in enumerate(
        zip(bar_positions, bar_widths, bar_heights)
    ):
        ax.bar(
            pos,
            height,
            width=width,
            align="edge",
            color=JET_COLOR_MAP[1],
            alpha=0.6,
            edgecolor="black",
            label=(
                f"Response: ({responses[0]*100:.0f}, {responses[1]*100:.0f}, {responses[2]*(upper-68.5)*100:.0f})"
                if i == 0
                else ""
            ),
        )

    # Add vertical line at the mean
    ax.axvline(
        x=mu,
        color=JET_COLOR_MAP[3],
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mu:.2f}",
    )

    # Add labels and legend
    ax.set_title("Truncated Normal Distribution with Probabilities")
    ax.set_xlabel("SRA at Time of Retirement")
    ax.set_ylabel("Probability Density")
    ax.set_xlim(lower, upper)
    ax.set_ylim(0, max(max(pdf) * 1.1, max(np.array(bar_heights)) * 1.1))
    ax.legend(loc="upper right")

    # # add the expected value and variance as text on the plot
    # plt.text(
    #     x=0.05 * (upper - lower) + lower,  # x position
    #     y= 0.8 * max(pdf),  # y position
    #     s=f"Expected Value: {expected_sra:.2f}\nVariance: {variance_sra:.2f}",
    #     fontsize=10,
    #     color="black",
    #     bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
    # )
    if save:
        if path_dict is None:
            raise ValueError("path_dict must be provided when save=True")
        response_str = "_".join(map(str, responses))
        plt.savefig(path_dict["beliefs_plots"] + f"truncated_normal_for_response_{response_str}.png", bbox_inches="tight")
    if show:
        plt.show()


def plot_expected_sra_vs_birth_year(
    df: pd.DataFrame = None, path_dict: dict = None, show: bool = False, save: bool = False
) -> None:
    """
    Plot the expected SRA against the birth year with a linear trendline.

    Args:
        df (pd.DataFrame): DataFrame containing 'gebjahr' and 'ex_val' columns.
    """
    if df is None and path_dict is not None:
        df = pd.read_csv(
            path_dict["intermediate_data"] + "beliefs/soep_is_truncated_normals.csv"
        )
    elif df is None and path_dict is None:
        raise ValueError("Either df or paths_dict must be provided.")

    df = filter_df(df)

    plt.figure()
    plt.scatter(df["gebjahr"], df["ex_val"], alpha=0.5, s=3, color=JET_COLOR_MAP[0])
    plt.title("Expected Value vs Birth Year")
    plt.xlabel("Birth Year")
    plt.ylabel("E[SRA]")
    z = np.polyfit(df["gebjahr"], df["ex_val"], 1)
    p = np.poly1d(z)
    plt.plot(
        df["gebjahr"],
        p(df["gebjahr"]),
        color=JET_COLOR_MAP[3],
        linewidth=2,
        label="OLS fit",
    )
    plt.legend()
    if save:
        if path_dict is None:
            raise ValueError("path_dict must be provided when save=True")
        plt.savefig(path_dict["beliefs_plots"] + "expected_sra_vs_birth_year.png", bbox_inches="tight")
    if show:
        plt.show()


def plot_alpha_heterogeneity_coefficients_combined(
    results_df: pd.DataFrame = None, path_dict: dict = None, show: bool = False, save: bool = False
) -> None:
    """
    Create a coefficient plot showing heterogeneity in alpha (expected SRA increase)
    by demographic covariates with 95% confidence intervals.

    Args:
        results_df (pd.DataFrame): DataFrame with heterogeneity results
        path_dict (dict): Path dictionary to load results if results_df is None
        show (bool): Whether to display the plot
    """
    # Load data if not provided
    if results_df is None and path_dict is not None:
        results_df = pd.read_csv(
            path_dict["intermediate_data"] + "beliefs/alpha_heterogeneity_results.csv"
        )
    elif results_df is None and path_dict is None:
        raise ValueError("Either results_df or path_dict must be provided.")

    # Prepare data for plotting
    covariates = results_df["covariate"].unique()
    specifications = results_df["specification"].unique()

    fig, ax = plt.subplots()

    # Define positions and colors
    x_positions = np.arange(len(covariates))
    width = 0.35
    colors = [JET_COLOR_MAP[0], JET_COLOR_MAP[1]]

    for i, spec in enumerate(["univariate", "with_age_control"]):
        spec_data = results_df[results_df["specification"] == spec]

        coefficients = []
        errors_lower = []
        errors_upper = []

        for covariate in covariates:
            row = spec_data[spec_data["covariate"] == covariate]
            if len(row) > 0:
                coef = row["coefficient"].iloc[0]
                ci_lower = row["ci_lower"].iloc[0]
                ci_upper = row["ci_upper"].iloc[0]

                coefficients.append(coef)
                errors_lower.append(coef - ci_lower)
                errors_upper.append(ci_upper - coef)
            else:
                coefficients.append(0)
                errors_lower.append(0)
                errors_upper.append(0)

        # Plot bars with error bars
        bars = ax.bar(
            x_positions + i * width - width / 2,
            coefficients,
            width,
            yerr=[errors_lower, errors_upper],
            capsize=5,
            color=colors[i],
            alpha=0.7,
            label=spec.replace("_", " ").title(),
            edgecolor="black",
            linewidth=0.5,
        )

    # Add horizontal line at zero
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)

    # Customize plot
    ax.set_xlabel("Demographic Covariates", fontsize=12)
    ax.set_ylabel("Coefficient Estimate", fontsize=12)
    ax.set_title(
        "Heterogeneity in Expected SRA Increase (α) by Demographics\nwith 95% Confidence Intervals",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([cov.capitalize() for cov in covariates])
    ax.legend(fontsize=11)

    # Add value labels on bars
    for i, spec in enumerate(["univariate", "with_age_control"]):
        spec_data = results_df[results_df["specification"] == spec]
        for j, covariate in enumerate(covariates):
            row = spec_data[spec_data["covariate"] == covariate]
            if len(row) > 0:
                coef = row["coefficient"].iloc[0]
                ax.text(
                    j + i * width - width / 2,
                    coef + (0.001 if coef >= 0 else -0.001),
                    f"{coef:.3f}",
                    ha="center",
                    va="bottom" if coef >= 0 else "top",
                    fontsize=9,
                    fontweight="bold",
                )

    plt.tight_layout()

    if save:
        if path_dict is None:
            raise ValueError("path_dict must be provided when save=True")
        plt.savefig(path_dict["beliefs_plots"] + "alpha_heterogeneity_coefficients_combined.png", bbox_inches="tight")
    if show:
        plt.show()


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
        df = pd.read_csv(path_dict["intermediate_data"] + "beliefs/beliefs_parameters.csv")
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
        plt.savefig(path_dict["beliefs_plots"] + "example_sra_evolution_no_increase.png", bbox_inches="tight")
    if show:
        plt.show()
