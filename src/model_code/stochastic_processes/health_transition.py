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
    cond_prob_disabled = calc_disability_probability(params, education, period, options)
    bad_health_prob = prob_vector[bad_health_var] * (1 - cond_prob_disabled)
    disabled_health_prob = prob_vector[bad_health_var] * cond_prob_disabled

    # Now set the probabilities in the vector(jax.numpy style)
    prob_vector = prob_vector.at[disabled_health_var].set(disabled_health_prob)
    prob_vector = prob_vector.at[bad_health_var].set(bad_health_prob)

    return prob_vector


def calc_disability_probability(params, education, period, options):
    age = options["start_age"] + period
    exp_value = jnp.exp(
        params["disability_logit_const"]
        + params["disability_logit_age"] * age
        + params["disability_logit_high_educ"] * education
    )
    prob = exp_value / (1 + exp_value)
    return prob
