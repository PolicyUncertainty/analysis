import matplotlib.pyplot as plt
import numpy as np
from model_code.derive_specs import generate_derived_and_data_derived_specs
from model_code.policy_states_belief import update_specs_exp_ret_age_trans_mat
from simulation.policy_state_scenarios.step_function import (
    update_specs_for_step_function,
)


def trajectory_plot(path_dict):
    # Load the estimates
    alpha_hat = np.loadtxt(path_dict["est_results"] + "var_params.txt")

    specs = generate_derived_and_data_derived_specs(path_dict)

    specs = update_specs_for_step_function(specs=specs, path_dict=path_dict)
    specs = update_specs_exp_ret_age_trans_mat(specs, path_dict)

    life_span = specs["end_age"] - specs["start_age"] + 1
    policy_state_67 = int((67 - specs["min_SRA"]) / specs["SRA_grid_size"])

    new_value_periods = specs["policy_step_periods"] + 1
    trans_mat = specs["beliefs_trans_mat"]

    policy_states_delta = (np.arange(trans_mat.shape[0]) - policy_state_67) * specs[
        "SRA_grid_size"
    ]

    step_function_vals = np.zeros(life_span) + policy_state_67
    continous_exp_values = np.zeros(life_span)

    for i in range(1, life_span):
        if np.isin(i, new_value_periods):
            step_function_vals[i] = step_function_vals[i - 1] + 1
        else:
            step_function_vals[i] = step_function_vals[i - 1]

        trans_mat_iter = np.linalg.matrix_power(trans_mat, i)

        continous_exp_values[i] = (
            trans_mat_iter[policy_state_67, :] @ policy_states_delta
        )

    step_function_vals = step_function_vals * specs["SRA_grid_size"] + specs["min_SRA"]
    continous_exp_values += 67

    ages = np.arange(life_span) + 30
    fig, ax = plt.subplots()
    ax.plot(ages, step_function_vals, label="Step function")
    ax.plot(ages, continous_exp_values, label="Continous expectation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        path_dict["plots"] + "counterfactual_design_1.png", transparent=True, dpi=300
    )
