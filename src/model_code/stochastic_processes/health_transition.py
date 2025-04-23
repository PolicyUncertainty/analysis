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
    cond_prob_disabled = calc_disability_probability(params, education, period)
    already_disabled = health == options["disabled_health_var"]
    cond_prob_disabled = cond_prob_disabled * (1 - already_disabled) + already_disabled

    # Now chain the probability of being in bad health with the conditional
    prob_bad_new = prob_vector[bad_health_var] * (1 - cond_prob_disabled)
    prob_disability_new = prob_vector[bad_health_var] * cond_prob_disabled
    # If already disabled, the probability remains the same
    prob_disability_new = jax.lax.select(
        already_disabled,
        on_true=prob_vector[disabled_health_var],
        on_false=prob_disability_new,
    )

    # Now set the probabilities in the vector(jax.numpy style)
    prob_vector = prob_vector.at[disabled_health_var].set(prob_disability_new)
    prob_vector = prob_vector.at[bad_health_var].set(prob_bad_new)

    return prob_vector


def calc_disability_probability(params, education, period):
    exp_value = jnp.exp(
        params["disability_logit_const"]
        + params["disability_logit_period"] * period
        + params["disability_logit_high_educ"] * education
    )
    prob = exp_value / (1 + exp_value)
    return prob
