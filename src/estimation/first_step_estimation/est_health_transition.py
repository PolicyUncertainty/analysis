import numpy as np
import pandas as pd
from scipy.stats import norm  # Import norm from scipy.stats for the Gaussian kernel
from specs.derive_specs import read_and_derive_specs


def estimate_health_transitions(paths_dict, specs):
    """Estimate the health state transition matrix."""

    # Load the data
    transition_data = pd.read_pickle(
        paths_dict["intermediate_data"] + "health_transition_estimation_sample.pkl"
    )

    # Define the Epanechnikov kernel function
    def epanechnikov_kernel(distance, bandwidth):
        u = distance / bandwidth
        return 0.75 * (1 - u**2) * (np.abs(u) <= 1)

    # Define the Gaussian kernel function
    def gaussian_kernel(distance, bandwidth):
        # Scale the distance by the bandwidth and use the Gaussian pdf
        return norm.pdf(distance / bandwidth)

    # Function to calculate the weighted mean using a specified kernel
    def kernel_weighted_mean(df, target_age, bandwidth, kernel_type):
        age_distances = np.abs(df['age'] - target_age)
        if kernel_type == 'epanechnikov':
            weights = epanechnikov_kernel(age_distances, bandwidth)
        elif kernel_type == 'gaussian':
            weights = gaussian_kernel(age_distances, bandwidth)
        else:
            raise ValueError("Invalid kernel type. Use 'epanechnikov' or 'gaussian'.")
        
        return np.sum(weights * df['lead_health_state']) / np.sum(weights)

    # Parameters
    kernel_type = specs.get("health_kernel_type", "epanechnikov")  # Default to Epanechnikov
    bandwidth = specs["health_smoothing_bandwidth"]
    
    # Adjust bandwidth for Gaussian kernel to ensure the desired probability mass
    if kernel_type == "gaussian":
        # Compute the bandwidth such that the Gaussian CDF from -infinity to -5 is approximately 1%
        bandwidth = bandwidth / norm.ppf(0.99)

    ages = np.arange(specs["start_age"], specs["end_age"] + 1)

    # Calculate the smoothed probabilities for each education level and health transition to transition to the lead_health_state
    def calculate_smoothed_probabilities(education, health_state):
        smoothed_values = [
            kernel_weighted_mean(
                transition_data[
                    (transition_data["education"] == education)
                    & (transition_data["health_state"] == health_state)
                ],
                age,
                bandwidth,
                kernel_type
            )
            for age in ages
        ]
        return pd.Series(smoothed_values, index=ages)

    # Compute transition probabilities
    transition_probabilities = {
        "hgg_h": calculate_smoothed_probabilities(education=1, health_state=1),
        "hgg_l": calculate_smoothed_probabilities(education=0, health_state=1),
        "hbg_h": calculate_smoothed_probabilities(education=1, health_state=0),
        "hbg_l": calculate_smoothed_probabilities(education=0, health_state=0),
    }

    # Complementary probabilities
    transition_probabilities.update(
        {
            "hgb_h": 1 - transition_probabilities["hgg_h"],
            "hgb_l": 1 - transition_probabilities["hgg_l"],
            "hbb_h": 1 - transition_probabilities["hbg_h"],
            "hbb_l": 1 - transition_probabilities["hbg_l"],
        }
    )

    # Construct the health transition matrix
    rows = []
    for education in [1, 0]:
        for health_state in [1, 0]:
            for lead_health_state, prob_key in zip(
                [1, 0], ["hgg", "hgb"] if health_state else ["hbg", "hbb"]
            ):
                key = f"{prob_key}_{'h' if education == 1 else 'l'}"
                rows.append(
                    {
                        "education": education,
                        "period": ages - specs["start_age"],
                        "health_state": health_state,
                        "lead_health_state": lead_health_state,
                        "transition_prob": transition_probabilities[key],
                    }
                )

    health_transition_matrix = pd.concat(
        [pd.DataFrame(row) for row in rows], ignore_index=True
    )

    # Save the results to a CSV file
    out_file_path = paths_dict["est_results"] + "health_transition_matrix.csv"
    health_transition_matrix.to_csv(out_file_path, index=False)

    return health_transition_matrix
