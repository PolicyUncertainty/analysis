# Description: This file estimates the parameters of the health transition matrix using the SOEP panel data.
# For each education level and age, we estimate P(health_state_(t+1)=a| health_state_t=b, education=c, age_t) non-parametrically.

import numpy as np
import pandas as pd
from specs.derive_specs import read_and_derive_specs

def load_transition_data(paths_dict):
    """Prepare the data for the transition estimation."""
    transition_data = pd.read_pickle(
        paths_dict["intermediate_data"] + "health_transition_estimation_sample.pkl"
    )
    return transition_data

def estimate_health_transitions(paths_dict, specs):
    """Estimate the health state transition matrix."""
    
    import numpy as np
    import pandas as pd

    # Load the data
    transition_data = load_transition_data(paths_dict)

    # Define the Epanechnikov kernel function
    def epanechnikov_kernel(distance, bandwidth):
        u = distance / bandwidth
        return 0.75 * (1 - u**2) * (np.abs(u) <= 1)

    # Function to calculate the weighted mean using the Epanechnikov kernel
    def kernel_weighted_mean(df, target_age, bandwidth):
        age_distances = np.abs(df['age'] - target_age)
        weights = epanechnikov_kernel(age_distances, bandwidth)
        return np.sum(weights * df['health_state']) / np.sum(weights)

    # Parameters
    bandwidth = specs["health_smoothing_bandwidth"]
    ages = np.arange(specs["start_age"] + 1, specs["end_age"] + 2)

    # Calculate the smoothed probabilities for each education level and health transition
    def calculate_smoothed_probabilities(education, lagged_health_state):
        smoothed_values = [
            kernel_weighted_mean(
                transition_data[
                    (transition_data['education'] == education) & 
                    (transition_data['lagged_health_state'] == lagged_health_state)
                ],
                age,
                bandwidth
            )
            for age in ages
        ]
        return pd.Series(smoothed_values, index=ages)

    # Compute transition probabilities
    transition_probabilities = {
        "hgg_h": calculate_smoothed_probabilities(education=1, lagged_health_state=1),
        "hgg_l": calculate_smoothed_probabilities(education=0, lagged_health_state=1),
        "hbg_h": calculate_smoothed_probabilities(education=1, lagged_health_state=0),
        "hbg_l": calculate_smoothed_probabilities(education=0, lagged_health_state=0),
    }

    # Complementary probabilities
    transition_probabilities.update({
        "hgb_h": 1 - transition_probabilities["hgg_h"],
        "hgb_l": 1 - transition_probabilities["hgg_l"],
        "hbb_h": 1 - transition_probabilities["hbg_h"],
        "hbb_l": 1 - transition_probabilities["hbg_l"],
    })

    # Construct the health transition matrix
    rows = []
    for education in [1, 0]:
        for health_state in [1, 0]:
            for lead_health_state, prob_key in zip([1, 0], ["hgg", "hgb"] if health_state else ["hbg", "hbb"]):
                key = f"{prob_key}_{'h' if education == 1 else 'l'}"
                rows.append({
                    "education": education,
                    "period": ages - 1 - specs["start_age"],
                    "health_state": health_state,
                    "lead_health_state": lead_health_state,
                    "transition_prob": transition_probabilities[key]
                })

    health_transition_matrix = pd.concat(
        [pd.DataFrame(row) for row in rows], ignore_index=True
    )

    # Save the results to a CSV file
    out_file_path = paths_dict["est_results"] + "health_transition_matrix.csv"
    health_transition_matrix.to_csv(out_file_path, index=False)

    return health_transition_matrix
