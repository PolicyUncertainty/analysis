import jax
import jax.numpy as jnp

from model_code.stochastic_processes.math_funcs import logit_formula


def job_offer_process_transition(
    params, policy_state, sex, health, model_specs, education, period, choice
):
    """Transition probability for next period job offer state.

    The values of process are the following:
    - 0: No job offer in case of unemployment and job destruction in case of employment
    - 1: Job offer in case of unemployment and no job destruction in case of employment

    """

    unemployment_choice = choice == 1
    labor_choice = choice >= 2

    age = model_specs["start_age"] + period
    good_health = (health == model_specs["good_health_var"]).astype(int)

    job_finding_prob_men = calc_job_finding_prob_men(
        params=params,
        education=education,
        good_health=good_health,
        age=age,
    )
    job_finding_prob_women = calc_job_finding_prob_women(
        params=params,
        education=education,
        good_health=good_health,
        age=age,
    )
    job_finding_prob = jax.lax.select(
        sex == 0, job_finding_prob_men, job_finding_prob_women
    )
    job_sep_prob = job_sep_probability(
        params=params,
        policy_state=policy_state,
        education=education,
        sex=sex,
        age=age,
        good_health=good_health,
        model_specs=model_specs,
    )

    # Transition probability
    prob_value_0 = (
        job_sep_prob * labor_choice + (1 - job_finding_prob) * unemployment_choice
    )

    return jnp.array([prob_value_0, 1 - prob_value_0])


def job_sep_probability(
    params, policy_state, education, sex, age, good_health, model_specs
):

    # Probability of job destruction
    log_job_sep_prob = model_specs["log_job_sep_probs"][
        sex, education, good_health, age
    ]

    SRA_interecpt = (
        params["SRA_firing_logit_intercept_men_low"] * (1 - education) * (1 - sex)
        + params["SRA_firing_logit_intercept_men_high"] * education * (1 - sex)
        + params["SRA_firing_logit_intercept_women_low"] * (1 - education) * sex
        + params["SRA_firing_logit_intercept_women_high"] * education * sex
    )

    policy_state_value = (
        model_specs["min_SRA"] + policy_state * model_specs["SRA_grid_size"]
    )
    at_policy_state = jnp.isclose(age, policy_state_value)
    logit_intercept = log_job_sep_prob + SRA_interecpt * at_policy_state
    job_sep_prob = logit_formula(logit_intercept)
    return job_sep_prob


def calc_job_finding_prob_men(params, education, good_health, age):
    above_55 = age >= 55
    exp_factor = (
        params["job_finding_logit_const_men"]
        + params["job_finding_logit_high_educ_men"] * education
        + params["job_finding_logit_good_health_men"] * good_health
        + params["job_finding_logit_age_men"] * age
        + params["job_finding_logit_age_above_55_men"] * (age - 55) * above_55
    )
    prob = logit_formula(exp_factor)
    return prob


def calc_job_finding_prob_women(params, education, good_health, age):
    above_55 = age >= 55

    exp_factor = (
        params["job_finding_logit_const_women"]
        + params["job_finding_logit_high_educ_women"] * education
        + params["job_finding_logit_good_health_women"] * good_health
        + params["job_finding_logit_age_women"] * age * (1 - above_55)
        + params["job_finding_logit_age_above_55_women"] * (age - 55) * above_55
    )
    prob = logit_formula(exp_factor)
    return prob
