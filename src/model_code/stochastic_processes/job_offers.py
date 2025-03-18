import jax
import jax.numpy as jnp


def job_offer_process_transition(params, sex, options, education, period, choice):
    """Transition probability for next period job offer state.

    The values of process are the following:
    - 0: No job offer in case of unemployment and job destruction in case of employment
    - 1: Job offer in case of unemployment and no job destruction in case of employment

    """

    unemployment_choice = choice == 1
    labor_choice = choice >= 2

    # Probability of job destruction
    job_sep_prob = options["job_sep_probs"][sex, education, period]

    job_finding_prob_men = calc_job_finding_prob_men(params, education, period)
    job_finding_prob_women = calc_job_finding_prob_women(params, education, period)
    job_finding_prob = jax.lax.select(
        sex == 0, job_finding_prob_men, job_finding_prob_women
    )

    # Transition probability
    prob_value_0 = (
        job_sep_prob * labor_choice + (1 - job_finding_prob) * unemployment_choice
    )

    return jnp.array([prob_value_0, 1 - prob_value_0])


def calc_job_finding_prob_men(params, education, period):
    exp_value = jnp.exp(
        params["job_finding_logit_const_men"]
        + params["job_finding_logit_period_men"] * period
        + params["job_finding_logit_high_educ_men"] * education
    )
    prob = exp_value / (1 + exp_value)
    return prob


def calc_job_finding_prob_women(params, education, period):
    exp_value = jnp.exp(
        params["job_finding_logit_const_women"]
        + params["job_finding_logit_period_women"] * period
        + params["job_finding_logit_high_educ_women"] * education
    )
    prob = exp_value / (1 + exp_value)
    return prob
