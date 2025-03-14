import jax.numpy as jnp
from numba.np.linalg import cond_impl


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
    not_disabled = health != options["disabled_health_var"]
    cond_prob_disabled = calc_disability_probability(params, education, period)
    cond_prob_disabled = cond_prob_disabled * not_disabled + (1 - not_disabled)

    # Now chain the probability of being in bad health with the conditional
    prob_bad_health_before = prob_vector[bad_health_var]
    prob_bad_health = prob_bad_health_before * (1 - cond_prob_disabled)
    prob_disability = prob_bad_health_before * cond_prob_disabled

    # Now set the probabilities in the vector(jax.numpy style)
    prob_vector = prob_vector.at[bad_health_var].set(prob_bad_health)
    prob_vector = prob_vector.at[disabled_health_var].set(prob_disability)

    return prob_vector


def calc_disability_probability(params, education, period):
    exp_value = jnp.exp(
        params["disability_logit_const"]
        + params["disability_logit_period"] * period
        + params["disability_logit_period"] * education
    )
    prob = exp_value / (1 + exp_value)
    return prob
