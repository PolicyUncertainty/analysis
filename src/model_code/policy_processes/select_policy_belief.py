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


def select_sim_policy_function_and_update_specs(
    specs, subj_unc, annoucement_age, SRA_at_start, SRA_at_retirement
):

    # Check if subjective uncertainty is given and if announcment is given.
    annoucement_given = annoucement_age is not None
    if ~subj_unc & annoucement_given:
        raise ValueError("Announcement can only be given for subjective uncertainty.")

    # If there is no announcement, we have a smooth policy change.  (This also nests no policy change)
    if annoucement_age is None:
        sim_alpha = (SRA_at_retirement - SRA_at_start) / (
            specs["resolution_age"] - specs["start_age"]
        )

        specs = update_specs_step_function_with_slope_and_resolution(
            specs=specs, slope=sim_alpha
        )
        transition_func_sol = realized_policy_step_function
    else:
        transition_func_sol = announce_policy_state
        specs = update_specs_for_policy_announcement(
            specs=specs,
            announcement_age=annoucement_age,
            announced_SRA=SRA_at_retirement,
        )

    return transition_func_sol, specs
