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
    There are only two cases: either subjective uncertainty exists or it does not.
    """

    # There can be two cases. Either subjective uncertainty exists or there is no uncertainty.
    # First we check if subjective uncertainty is given.
    if subj_unc:
        if custom_resolution_age is None:
            specs["resolution_age"] = specs["resolution_age_estimation"]
        else:
            specs["resolution_age"] = custom_resolution_age

        # Assign subjective expectation funtion and output.
        specs = update_specs_exp_ret_age_trans_mat(specs=specs, path_dict=path_dict)
        transition_func_sol = expected_SRA_with_resolution
    else:
        # If there is no subjective uncertainty and a resolution age is given, we return an error.
        if custom_resolution_age is not None:
            raise ValueError(
                "custom_resolution_age can only be given in case of subjective uncertainty"
            )

        specs["resolution_age"] = None
        # Assign empty step periods as we have no increase in expectation, because of the equality of
        # the value function if there is no uncertainty about the future of the SRA.
        specs["policy_step_periods"] = np.array([])
        # Assign policy step function as transition function.
        transition_func_sol = realized_policy_step_function
        # We also assign the resolution age to the start age
    return transition_func_sol, specs


def select_sim_policy_function_and_update_specs(
    specs,
    subj_unc,
    announcement_age,
    SRA_at_start,
    SRA_at_retirement,
    custom_resolution_age,
):

    # Check if subjective uncertainty is given and if announcment is given.
    announcement_given = announcement_age is not None
    if (not subj_unc) & announcement_given:
        raise ValueError("Announcement can only be given for subjective uncertainty.")

    if SRA_at_retirement < SRA_at_start:
        raise ValueError(
            "SRA at retirement must be larger than SRA at start. "
            "This is not a valid policy change."
        )

    # If there is no announcement, we have a smooth policy change.  (This also nests no policy change)
    if announcement_age is None:
        # If there is no subjective uncertainty, there should be no gradual increase.
        if SRA_at_retirement == SRA_at_start:
            specs["policy_step_periods"] = np.array([])
        else:
            if custom_resolution_age is None:
                specs["resolution_age"] = specs["resolution_age_estimation"]
            else:
                specs["resolution_age"] = custom_resolution_age

            # We are in the case of subjective uncertainty and a gradual increase in the SRA.
            sim_alpha = (SRA_at_retirement - SRA_at_start) / (
                specs["resolution_age"] - specs["start_age"]
            )
            specs = update_specs_step_function_with_slope_and_resolution(
                specs=specs, slope=sim_alpha
            )
        transition_func_sim = realized_policy_step_function
    else:
        if "resolution_age" in specs.keys():
            pass
        else:
            specs["resolution_age"] = None
        transition_func_sim = announce_policy_state
        specs = update_specs_for_policy_announcement(
            specs=specs,
            announcement_age=announcement_age,
            announced_SRA=SRA_at_retirement,
        )

    return transition_func_sim, specs
