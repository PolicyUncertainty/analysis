import numpy as np
from model_code.policy_processes.policy_states_belief import (
    expected_SRA_probs_estimation,
)
from model_code.policy_processes.policy_states_belief import (
    update_specs_exp_ret_age_trans_mat,
)
from model_code.policy_processes.step_function import create_update_function_for_slope
from model_code.policy_processes.step_function import realized_policy_step_function


def select_expectation_functions_and_model_sol_names(
    path_dict, exp_params_sol, sim_alpha=None
):
    if isinstance(exp_params_sol, dict):
        # Check if alpha and sigma_sq are in the exp_params, otherwise raise error
        if "alpha" not in exp_params_sol:
            raise ValueError("exp_params must contain the key 'alpha', if it is a dict")

        if exp_params_sol["alpha"] == "subjective":
            alpha_hat = np.loadtxt(path_dict["est_results"] + "exp_val_params.txt")
            update_func_sol = create_update_function_for_slope(alpha_hat)
            transition_func_sol = realized_policy_step_function
            sol_name = "subj_no_unc"
        # Check if alpha is a float or int
        elif exp_params_sol["alpha"] in [float, int]:
            update_func_sol = create_update_function_for_slope(exp_params_sol["alpha"])
            transition_func_sol = realized_policy_step_function
            # Generate a string with only 3 digits after the comma
            sol_name = "{:.3f}".format(exp_params_sol["alpha"])
        else:
            raise ValueError("exp_params['alpha'] must be a float, int or 'subject'")
    elif not exp_params_sol:
        update_func_sol = update_specs_exp_ret_age_trans_mat
        transition_func_sol = expected_SRA_probs_estimation
        sol_name = "subj_unc"
    else:
        raise ValueError("exp_params must be a dict or False")

    model_sol_names = {
        "solution": "exp_" + sol_name + ".pkl",
    }
    update_funcs = {
        "solution": update_func_sol,
    }
    transition_funcs = {
        "solution": transition_func_sol,
    }
    if sim_alpha is not None:
        if isinstance(sim_alpha, float) or isinstance(sim_alpha, int):
            update_func_sim = create_update_function_for_slope(sim_alpha)
            transition_func_sim = realized_policy_step_function
            subj_alpha = np.loadtxt(path_dict["est_results"] + "exp_val_params.txt")
            if np.allclose(subj_alpha, sim_alpha):
                sim_name = "subj_no_unc"
            else:
                sim_name = "{:.3f}".format(sim_alpha)
        else:
            raise ValueError("sim_alpha must be a float or int")

        model_sol_names["simulation"] = "exp_" + sol_name + "_sim_" + sim_name + ".pkl"
        update_funcs["simulation"] = update_func_sim
        transition_funcs["simulation"] = transition_func_sim

    return update_funcs, transition_funcs, model_sol_names
