import jax
import jax.numpy as jnp


def health_transition(sex, health, education, period, params, options):
    trans_mat = options["health_trans_mat"]
    prob_vector = trans_mat[sex, education, period, health, :]

    # Generate conditional probability of being disabled.
    # Start by reading out health state values:
    bad_health_var = options["bad_health_var"]
    disabled_health_var = options["disabled_health_var"]

    # If the agent is disabled the conditional probability of being in bad health
    # and remaining in disability is 1, i.e. there is no transition from disability
    # to bad health. Only to good.
    # If you are dead, it does not matter, as you are dead(absorbing).
    cond_prob_disabled = calc_disability_probability(
        params=params, sex=sex, education=education, period=period, options=options
    )
    bad_health_prob = prob_vector[bad_health_var] * (1 - cond_prob_disabled)
    disabled_health_prob = prob_vector[bad_health_var] * cond_prob_disabled

    # Now set the probabilities in the vector(jax.numpy style)
    prob_vector = prob_vector.at[disabled_health_var].set(disabled_health_prob)
    prob_vector = prob_vector.at[bad_health_var].set(bad_health_prob)

    return prob_vector


def calc_disability_probability(params, sex, education, period, options):
    age = options["start_age"] + period

    # Calculate exp value for men and women
    exp_value_men = jnp.exp(
        params["disability_logit_const_men"]
        + params["disability_logit_age_men"] * age
        + params["disability_logit_high_educ_men"] * education
    )
    exp_value_women = jnp.exp(
        params["disability_logit_const_women"]
        + params["disability_logit_age_women"] * age
        + params["disability_logit_high_educ_women"] * education
    )
    # Now select based on sex state
    is_men = sex == 0
    exp_value = jax.lax.select(is_men, on_true=exp_value_men, on_false=exp_value_women)
    prob = exp_value / (1 + exp_value)
    return prob
