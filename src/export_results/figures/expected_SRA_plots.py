import matplotlib.pyplot as plt
import numpy as np
from model_code.derive_specs import generate_derived_and_data_derived_specs
from model_code.policy_states_belief import update_specs_exp_ret_age_trans_mat


def plot_markov_process(paths_dict):
    specs = generate_derived_and_data_derived_specs(paths_dict)

    specs = update_specs_exp_ret_age_trans_mat(specs, paths_dict)

    belief_trans_mat = specs["beliefs_trans_mat"]

    # Assuming belief_trans_mat is a 2D numpy array

    # Initialize variables
    n_periods = 35
    mean_vals = [67]
    std_vals = [0]

    SRA_values = np.arange(
        specs["min_SRA"],
        specs["max_SRA"] + specs["SRA_grid_size"],
        specs["SRA_grid_size"],
    )

    for i in range(1, n_periods + 1):
        # Calculate the matrix to the power of i
        result_mat = np.linalg.matrix_power(belief_trans_mat, i)

        mean = np.dot(result_mat[8], SRA_values)
        variance = np.dot(result_mat[8], SRA_values**2) - mean**2
        std = np.sqrt(variance)

        # Append to lists
        mean_vals.append(mean)
        std_vals.append(std)

    # Plot the results
    fig, ax = plt.subplots()
    ax.plot(range(0, n_periods + 1), mean_vals)
    ax.fill_between(
        range(0, n_periods + 1),
        np.array(mean_vals) - np.array(std_vals),
        np.array(mean_vals) + np.array(std_vals),
        alpha=0.3,
    )
    ax.set_xlabel("Time to retirement")
    ax.set_ylabel("SRA")
    # ax.legend()
    fig.tight_layout()
    fig.savefig(
        paths_dict["plots"] + "expectation_process.png", transparent=True, dpi=300
    )
