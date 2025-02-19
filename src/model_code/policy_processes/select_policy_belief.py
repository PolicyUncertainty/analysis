import numpy as np
from model_code.policy_processes.policy_states_belief import (
    expected_SRA_with_resolution,
)
from model_code.policy_processes.policy_states_belief import (
    update_specs_exp_ret_age_trans_mat,
)
from model_code.policy_processes.step_function import realized_policy_step_function
from model_code.policy_processes.step_function import (
    update_specs_step_function_with_slope_and_resolution,
)


def select_transition_func_and_update_specs(
    path_dict, specs, subj_unc, sim_alpha, custom_resolution_age
):
    # Check if subj_unc is bool
    if not isinstance(subj_unc, bool):
        raise ValueError("subj_unc must be a boolean")

    # CHeck if subj_unc is given and sim_alpha is not None
    if subj_unc and sim_alpha is not None:
        raise ValueError("sim_alpha is not available for subjective uncertainty")

    if custom_resolution_age is None:
        specs["resolution_age"] = specs["resolution_age_estimation"]
    else:
        specs["resolution_age"] = custom_resolution_age

    if subj_unc:
        specs = update_specs_exp_ret_age_trans_mat(specs, path_dict)
        transition_func_sol = expected_SRA_with_resolution
    else:
        if sim_alpha is None:
            slope = 0.0
        else:
            slope = sim_alpha

        specs = update_specs_step_function_with_slope_and_resolution(specs, slope)
        transition_func_sol = realized_policy_step_function

    return specs, transition_func_sol

    #
    #     subj_alpha = np.loadtxt(path_dict["est_results"] + "exp_val_params.txt")
    #     if np.allclose(subj_alpha, expected_alpha):
    #         sol_name = name_pre + "subj_no_unc"
    #     else:
    #         # Generate a string with only 3 digits after the comma
    #         sol_name = name_pre + "{:.2f}".format(expected_alpha)
    #
    # elif not expected_alpha:
    #     update_func_sol = update_specs_exp_ret_age_trans_mat
    #     if resolution_age:
    #
    # else:
    #     raise ValueError("exp_params must be a dict or False")
    #
    # model_sol_names = {
    #     "solution": "exp_" + sol_name + ".pkl",
    # }
    # update_funcs = {
    #     "solution": update_func_sol,
    # }
    # transition_funcs = {
    #     "solution": transition_func_sol,
    # }
    # if sim_alpha is not None:
    #     if isinstance(sim_alpha, float):
    #         if resolution_age:
    #             update_func_sim = create_update_function_for_slope_and_resolution(
    #                 sim_alpha
    #             )
    #         else:
    #             update_func_sim = create_update_function_for_slope(sim_alpha)
    #         transition_func_sim = realized_policy_step_function
    #         subj_alpha = np.loadtxt(path_dict["est_results"] + "exp_val_params.txt")
    #         if np.allclose(subj_alpha, sim_alpha):
    #             sim_name = "subj_no_unc"
    #         else:
    #             sim_name = "{:.2f}".format(sim_alpha)
    #     else:
    #         raise ValueError("sim_alpha must be a float or int")
    #
    #     if resolution_age:
    #         model_sol_names["simulation"] = (
    #             "exp_res_" + sol_name + "_sim_" + sim_name + ".pkl"
    #         )
    #     else:
    #         model_sol_names["simulation"] = (
    #             "exp_" + sol_name + "_sim_" + sim_name + ".pkl"
    #         )
    #     update_funcs["simulation"] = update_func_sim
    #     transition_funcs["simulation"] = transition_func_sim
    #
    # return update_funcs, transition_funcs, model_sol_names
