from zmq.backend.cffi import proxy

from model_code.state_space.choice_set import state_specific_choice_set
from model_code.state_space.experience import get_next_period_experience


def create_state_space_functions():
    return {
        "state_specific_choice_set": state_specific_choice_set,
        "next_period_experience": get_next_period_experience,
        "sparsity_condition": sparsity_condition,
    }


def sparsity_condition(
    period,
    lagged_choice,
    sex,
    informed,
    health,
    partner_state,
    policy_state,
    education,
    options,
):
    start_age = options["start_age"]
    max_ret_age = options["max_ret_age"]
    # Generate last period, because only here are death states
    last_period = options["n_periods"] - 1
    # Degenerated policy state
    degenerate_policy_state = options["n_policy_states"] - 1

    age = start_age + period
    # Men can't have lagged choice part-time.
    if (sex == 0) & (lagged_choice == 2):
        return False
    # After the maximum retirement age, you must be retired. We exclude the states, where
    # the agent is dead. All death states will be proxied later anyways to death states in the last
    # period.
    if (
        (age > max_ret_age)
        & (lagged_choice != 0)
        & (health != options["death_health_var"])
    ):
        return False
    else:
        # Now turn to the states, where it is decided by the value of an exogenous
        # state if it is valid or not. For invalid states we provide a proxy state
        if health == options["death_health_var"]:
            # Lead all states with death to last period death states
            # with job offer 0, as dead agent's only get assigned a dummy choice
            # set, only including 0.

            # For retirement choice last period (lagged_choice=0):
            # You could be in principle die upon retirement for which we need informed and policy state.
            # Note, that in the solution of the model (for the expectations), the agent's do not expect,
            # that their informed state can't change. In the simulation, the agent always
            # gets informed when they choose retirement.
            state_proxy = {
                "period": last_period,
                "lagged_choice": lagged_choice,
                "education": education,
                "health": health,
                "informed": informed,
                "sex": sex,
                "partner_state": partner_state,
                "job_offer": 0,
                "policy_state": policy_state,
            }
            return state_proxy
        elif lagged_choice == 0:
            # If retirement is already chosen we proxy all states to job offer 0. Also, if you are
            # retired, it does not matter if you are in the bad health state or disabled. We proxy
            # the health state of those two by bad health.
            if health == options["good_health_var"]:
                proxy_health = health
            else:
                proxy_health = options["bad_health_var"]

            # Until age max_ret_age + 1 the individual could also be freshly retired.
            # Therefore, we check if the agent already is longer retired. If so, we proxy
            # informed only by 1 and policy state only by the degenerate policy state. Otherwise
            # the agent is freshly retired and we need informed and policy state.
            already_longer_retired = (age > max_ret_age + 1) | (
                policy_state == degenerate_policy_state
            )
            if already_longer_retired:
                informed_proxy = 1
                policy_state_proxy = degenerate_policy_state
            else:
                informed_proxy = informed
                policy_state_proxy = policy_state
            state_proxy = {
                "period": period,
                "lagged_choice": lagged_choice,
                "education": education,
                "health": proxy_health,
                "informed": informed_proxy,
                "sex": sex,
                "partner_state": partner_state,
                "job_offer": 0,
                "policy_state": policy_state_proxy,
            }
            return state_proxy
        else:
            return True
