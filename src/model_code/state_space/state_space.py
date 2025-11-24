import jax.numpy as jnp

from model_code.state_space.choice_set import state_specific_choice_set
from model_code.state_space.experience import get_next_period_experience


def create_state_space_functions():
    return {
        "state_specific_choice_set": state_specific_choice_set,
        "next_period_experience": get_next_period_experience,
        "sparsity_condition": sparsity_condition,
        "next_period_deterministic_state": next_period_deterministic_state,
    }


def next_period_deterministic_state(
    period, lagged_choice, choice, alg_1_claim, model_specs
):
    # First assign standard updates
    next_period = period + 1
    lagged_choice_next = choice

    # Determine age
    age = model_specs["start_age"] + period

    # Create max periods. The first time you can have to periods is with age 58. That
    # means it can first be updated to be calim 2 at age 57
    max_periods_alg_1 = (age < 57) + (age >= 57) * 2

    # Alg 1 claim for longer
    already_alg_1_claim = alg_1_claim > 0
    # Use min to avoid overflow
    alg_1_claim_longer = already_alg_1_claim * (alg_1_claim.clip(min=1) - 1)

    # Alg 1 claim. First check if already unemployed
    not_already_unemployed = lagged_choice != 1
    alg_1_claim_next = (
        not_already_unemployed * max_periods_alg_1
        + (1 - not_already_unemployed) * alg_1_claim_longer
    )
    choose_unemployment = choice == 1
    alg_1_claim_next = alg_1_claim_next * choose_unemployment

    return {
        "period": next_period,
        "lagged_choice": lagged_choice_next,
        "alg_1_claim": alg_1_claim_next,
    }


def sparsity_condition(
    period,
    lagged_choice,
    sex,
    alg_1_claim,
    informed,
    health,
    partner_state,
    policy_state,
    education,
    model_specs,
):
    start_age = model_specs["start_age"]
    max_ret_age = model_specs["max_ret_age"]
    # Generate last period, because only here are death states
    last_period = model_specs["n_periods"] - 1
    # Degenerated policy state
    degenerate_policy_state = model_specs["n_policy_states"] - 1

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
        & (health != model_specs["death_health_var"])
    ):
        return False

    elif (alg_1_claim > 0) & (lagged_choice in [0, 2, 3]):
        # We don't need alg_1 claim if not unemployed in last period
        return False
    elif (alg_1_claim == 2) & (age < 58):
        # We don't need alg_1 claim if not unemployed in last period
        return False
    elif (alg_1_claim == 1) & (age == 58):
        # At age 58 you can not have claim = 1, as if you decided with 57, you get
        # 2 and there was no one in 2 at age 57 who gets substracted 1
        return False
    else:
        # Now turn to the states, where it is decided by the value of an exogenous
        # state if it is valid or not. For invalid states we provide a proxy state
        if health == model_specs["death_health_var"]:
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
                "alg_1_claim": 0,
                "health": health,
                "informed": informed,
                "sex": sex,
                "partner_state": partner_state,
                "job_offer": 0,
                "policy_state": policy_state,
            }
            return state_proxy
        elif (lagged_choice == 0) | (policy_state == degenerate_policy_state):
            # If retirement is already chosen we proxy all states to job offer 0.
            job_offer_proxy = 0

            # If you already longer retired, i.e. either past max_ret_age + 1 or in the
            # degenerate policy state, we do not need informed and policy state anymore.
            # Before you can always be fresh retired, where we need this information.
            # Also now disabled and bad health are the same
            already_longer_retired = (age > max_ret_age + 1) | (
                policy_state == degenerate_policy_state
            )
            if already_longer_retired:
                informed_proxy = 1
                policy_state_proxy = degenerate_policy_state
                # We already know that the agent is not dead
                if health == model_specs["good_health_var"]:
                    proxy_health = health
                else:
                    proxy_health = model_specs["bad_health_var"]
            else:
                informed_proxy = informed
                policy_state_proxy = policy_state
                proxy_health = health
            state_proxy = {
                "period": period,
                "lagged_choice": 0,
                "education": education,
                "health": proxy_health,
                "alg_1_claim": 0,
                "informed": informed_proxy,
                "sex": sex,
                "partner_state": partner_state,
                "job_offer": job_offer_proxy,
                "policy_state": policy_state_proxy,
            }
            return state_proxy
        else:
            return True
