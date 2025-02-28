import numpy as np
from model_code.policy_processes.policy_states_belief import (
    expected_SRA_probs_estimation,
)
from model_code.policy_processes.policy_states_belief import (
    expected_SRA_with_resolution,
)
from model_code.policy_processes.policy_states_belief import (
    update_specs_exp_ret_age_trans_mat,
)
from model_code.policy_processes.step_function import create_update_function_for_slope
from model_code.policy_processes.step_function import (
    create_update_function_for_slope_and_resolution,
)
from model_code.policy_processes.step_function import realized_policy_step_function


def select_expectation_functions_and_model_sol_names(
    path_dict, expected_alpha, sim_alpha=None, resolution=False
):
    if isinstance(expected_alpha, float):
        if resolution:
            update_func_sol = create_update_function_for_slope_and_resolution(
                expected_alpha
            )
            name_pre = "res_"
        else:
            update_func_sol = create_update_function_for_slope(expected_alpha)
            name_pre = ""
        transition_func_sol = realized_policy_step_function

        subj_alpha = np.loadtxt(path_dict["est_results"] + "exp_val_params.txt")
        if np.allclose(subj_alpha, expected_alpha):
            sol_name = name_pre + "subj_no_unc"
        else:
            # Generate a string with only 3 digits after the comma
            sol_name = name_pre + "{:.2f}".format(expected_alpha)

    elif not expected_alpha:
        update_func_sol = update_specs_exp_ret_age_trans_mat
        if resolution:
            transition_func_sol = expected_SRA_with_resolution
            sol_name = "res_subj_unc"
        else:
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
        if isinstance(sim_alpha, float):
            if resolution:
                update_func_sim = create_update_function_for_slope_and_resolution(
                    sim_alpha
                )
            else:
                update_func_sim = create_update_function_for_slope(sim_alpha)
            transition_func_sim = realized_policy_step_function
            subj_alpha = np.loadtxt(path_dict["est_results"] + "exp_val_params.txt")
            if np.allclose(subj_alpha, sim_alpha):
                sim_name = "subj_no_unc"
            else:
                sim_name = "{:.2f}".format(sim_alpha)
        else:
            raise ValueError("sim_alpha must be a float or int")

        if resolution:
            model_sol_names["simulation"] = (
                "exp_res_" + sol_name + "_sim_" + sim_name + ".pkl"
            )
        else:
            model_sol_names["simulation"] = (
                "exp_" + sol_name + "_sim_" + sim_name + ".pkl"
            )
        update_funcs["simulation"] = update_func_sim
        transition_funcs["simulation"] = transition_func_sim

    return update_funcs, transition_funcs, model_sol_names
