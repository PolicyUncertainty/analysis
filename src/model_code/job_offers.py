import jax.numpy as jnp


def job_offer_process_transition(params, options, education, period, choice):
    """Transition probability for next period job offer state.

    The values of process are the following:
    - 0: No job offer in case of unemployment and job destruction in case of employment
    - 1: Job offer in case of unemployment and no job destruction in case of employment

    """

    unemployment_choice = choice == 0
    labor_choice = choice == 1

    # Probability of job destruction
    job_sep_prob = options["job_sep_probs"][education, period]

    job_finding_prob = calc_job_finding_prob(params, education, period, options)

    # Transition probability
    prob_value_0 = (
        job_sep_prob * labor_choice + (1 - job_finding_prob) * unemployment_choice
    )

    return jnp.array([prob_value_0, 1 - prob_value_0])


def calc_job_finding_prob(params, education, period, options):
    high_edu = education == 1
    age = period + options["start_age"]
    exp_value = jnp.exp(
        params["job_finding_logit_const"]
        + params["job_finding_logit_age"] * age
        + params["job_finding_logit_high_educ"] * high_edu
    )
    prob = exp_value / (1 + exp_value)
    return prob
