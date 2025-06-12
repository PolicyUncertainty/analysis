import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from beliefs.sra_beliefs.est_SRA_expectations import estimate_truncated_normal_parameters


def plot_truncated_normal_for_response(
    responses: list[float],
    options,
    mu: float = None,
    sigma: float = None,
    upper_trunc_limit: float = None,
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
        df = pd.DataFrame({
            "pol_unc_stat_ret_age_67": [responses[0]],
            "pol_unc_stat_ret_age_68": [responses[1]],
            "pol_unc_stat_ret_age_69": [responses[2]],
        })

        # create function_spec dict
        function_spec = {
            "ll": options["lower_limit"],
            "ul": options["upper_limit"] if upper_trunc_limit is None else upper_trunc_limit,
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
    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf, color='blue', label="Truncated Normal PDF")

    # Divide responses by 100
    # Normalize the probability mass for the range 69+ because the domain is larger than 1 (unlike fot the other two responses)
    responses = np.array(responses) / 100.0
    responses[2] =  responses[2]/(upper - 68.5)  

    # Define the x positions and widths for the bars
    bar_positions = [66.5, 67.5, 68.5]
    bar_widths = [1.0, 1.0, upper - 68.5]
    bar_heights = responses

    # Plot the bars
    for pos, width, height in zip(bar_positions, bar_widths, bar_heights):
        plt.bar(pos, height, width=width, align='edge', color='orange', alpha=0.6, edgecolor='black', label="Probabilities" if pos == 66.5 else "")

    # Add labels and legend
    plt.title("Truncated Normal Distribution with Probabilities")
    plt.xlabel("SRA at Time of Retirement")
    plt.ylabel("Probability Density")
    plt.xlim(lower, upper)
    plt.ylim(0, max(max(pdf) * 1.1 , max(np.array(bar_heights)) * 1.1))
    plt.legend()

    # add the expected value and variance as text on the plot
    plt.text(
        x=0.05 * (upper - lower) + lower,  # x position
        y= 0.8 * max(pdf),  # y position
        s=f"Expected Value: {expected_sra:.2f}\nVariance: {variance_sra:.2f}",
        fontsize=10,
        color="black",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
    )
    plt.show()


def plot_expected_sra_vs_birth_year(
    df: pd.DataFrame, 
) -> None:
    """
    Plot the expected SRA against the birth year with a linear trendline.

    Args:
        df (pd.DataFrame): DataFrame containing 'gebjahr' and 'ex_val' columns.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(df["gebjahr"], df["ex_val"], alpha=0.5, s=3, color='blue')
    plt.title("Expected Value vs Birth Year")
    plt.xlabel("Birth Year")
    plt.ylabel("E[SRA]")
    z = np.polyfit(df["gebjahr"], df["ex_val"], 1)
    p = np.poly1d(z)
    plt.plot(df["gebjahr"], p(df["gebjahr"]), color='red', linewidth=2, label='OLS fit')
    plt.grid()
    plt.legend()
    plt.show()

