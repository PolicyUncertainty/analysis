import numpy as np

from model_code.policy_processes.announcment import (
    announce_policy_state,
    update_specs_for_policy_announcement,
)
from model_code.policy_processes.policy_states_belief import (
    expected_SRA_with_resolution,
    update_specs_exp_ret_age_trans_mat,
)
from model_code.policy_processes.step_function import (
    realized_policy_step_function,
    update_specs_step_function_with_slope_and_resolution,
)


def select_solution_transition_func_and_update_specs(
    path_dict,
    specs,
    subj_unc,
    custom_resolution_age,
):
    """Select the solution SRA belief transition function and update the model specs accordingly.
    There are only two cases weather subjective uncertainty exists or not.
    """

    # There can be two cases. Either subjective uncertainty exists or there is no uncertainty.
    # First we check if subjective uncertainty is given.
    if subj_unc:
        if custom_resolution_age is None:
            specs["resolution_age"] = specs["resolution_age_estimation"]
        else:
            specs["resolution_age"] = custom_resolution_age

        # Assign subjective expectation funtion and output.
        specs = update_specs_exp_ret_age_trans_mat(specs, path_dict)
        transition_func_sol = expected_SRA_with_resolution
    else:
        # If there is no subjective uncertainty and a resolution age is given, we return an error.
        if custom_resolution_age is not None:
            raise ValueError(
                "custom_resolution_age can only be given in case of subjective uncertainty"
            )

        specs = update_specs_step_function_with_slope_and_resolution(
            specs=specs, slope=0.0
        )
        transition_func_sol = realized_policy_step_function
    return transition_func_sol, specs


def select_transition_func_and_update_specs(
    path_dict,
    specs,
    sim_alpha,
    annoucement_age,
    annoucement_SRA,
):
    announcment_given = annoucement_age is not None and annoucement_SRA is not None
    if sim_alpha is None and announcment_given:
        transition_func_sol = announce_policy_state
        specs = update_specs_for_policy_announcement(
            specs, annoucement_age, annoucement_SRA
        )
    else:
        if sim_alpha is None:
            sim_alpha = 0.0

        specs = update_specs_step_function_with_slope_and_resolution(specs, sim_alpha)
        transition_func_sol = realized_policy_step_function

    return specs, transition_func_sol

    # In the simulation, things can be more difficult. First, suppose
    # agents hold subjective uncertainty. Then in the simulation, there can be two cases:
    # Smooth change of SRA(including no change) and announcment. Determine the relevant parameters
    # for this
    if subj_unc:
        # If there is no announcment, we have a smooth change with sim_alpha
        if annoucement_age is None:
            sim_alpha = (SRA_at_retirement - SRA_at_start) / (
                model_params["resolution_age"] - model_params["start_age"]
            )
            announcment_SRA = None
        else:
            sim_alpha = None
            announcment_SRA = SRA_at_retirement
    else:
        # If there is no uncertainty then we SRA at resolution is the same as SRA at start.
        # We also check that here
        if SRA_at_start != SRA_at_retirement:
            raise ValueError(
                "SRA at start and resolution must be the same when there is no uncertainty"
            )
        # Announcment is not allowed to be given
        if annoucement_age is not None:
            raise ValueError(
                "Announcment age can only be given in case of subjective uncertainty"
            )
        # We set sim_alpha to 0 for the simulation. (For the solution it is clear if subj_exp is False)
        sim_alpha = 0.0
        # We also set the announcment SRA to None
        announcment_SRA = None
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
